"""Lightweight single-instance MuJoCo gymnasium env for inspection pick-and-place.

This env intentionally avoids the mjlab / mujoco_warp GPU stack.  It wraps a
single ``mujoco.MjModel`` / ``mujoco.MjData`` pair, exposes ``IKEE_Obs`` batched
observations (batch dim = 1), and satisfies the ``GymVectorInterface`` protocol
so ``skill_lib`` factories (``make_pick_skill``, ``make_reach_xyzrpy_skill``, …)
work unchanged.

Observation shape convention: every tensor has a leading batch dim of 1, matching
the (1, ...) shapes the IK-EE policy expects.

Action convention: shape ``(1, 8)`` — first 7 elements are arm joint position
targets in radians; element 7 is gripper_normalized ∈ [0, 1] (0=open, 1=closed),
which is mapped to the MuJoCo ``fingers_actuator`` ctrl range [0, 255].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
import torch
import gymnasium as gym

from skillet.core.math import quat_apply, quat_mul, subtract_frame_transforms
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.specs import IK_EE_SPEC_BATCHED, JOINT_VEL_SPEC

from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_spec import InspectionSceneSpec


class InspectionMjEnv:
    """Lightweight single-instance MuJoCo env for inspection pick-and-place.

    Satisfies ``GymVectorInterface`` with ``num_envs=1`` so it is compatible
    with every ``skill_lib`` factory that calls ``env.coerce_obs_spec("ik_ee")``
    and ``env.action_spec``.

    The caller is responsible for building the ``InspectionSceneSpec`` with
    ``include_robot=True`` before passing it here.

    Args:
        spec: Pre-built scene spec.  Must contain the Gen3 arm
            (``spec.model.nu == 8``).

    """

    num_envs: int = 1

    _N_ARM_JOINTS: int = 7
    _TCP_OFFSET: list[float] = [0.0, 0.0, 0.120, 1.0, 0.0, 0.0, 0.0]
    _GRIPPER_MAX: float = 0.8
    _GRIPPER_CTRL_MAX: float = 255.0
    _EE_LINK_NAME: str = "end_effector_link"
    _BASE_LINK_NAME: str = "base_link"
    _ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"joint_{i}" for i in range(1, 8))
    _DRIVER_JOINT_NAME: str = "right_driver_joint"

    def __init__(self, spec: InspectionSceneSpec) -> None:
        if spec.model.nu != 8:
            raise ValueError(
                f"Expected nu=8 (7 arm actuators + 1 fingers_actuator), got nu={spec.model.nu}. "
                "Pass include_robot=True to make_inspection_scene()."
            )

        self._model = spec.model
        self._data = mujoco.MjData(self._model)
        self._n_blocks = len(spec.blocks)

        # qpos layout: n_blocks freejoints (7 qpos each), then arm (7), then gripper joints
        # qvel layout: n_blocks freejoints (6 vel each), then arm (7), then gripper
        self._arm_qpos_start = 7 * self._n_blocks
        self._arm_qvel_start = 6 * self._n_blocks

        # body IDs for FK / Jacobian computation
        self._ee_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._EE_LINK_NAME)
        self._base_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._BASE_LINK_NAME)

        # Arm joint limits: build (1, 2, 8) tensor [lower; upper] for 7 arm + 1 driver
        arm_jnt_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._ARM_JOINT_NAMES
        ]
        drv_jnt_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, self._DRIVER_JOINT_NAME)
        arm_limits = torch.tensor(
            np.array([self._model.jnt_range[jid] for jid in arm_jnt_ids]), dtype=torch.float32
        )  # (7, 2)
        drv_limits = torch.tensor(self._model.jnt_range[drv_jnt_id].copy(), dtype=torch.float32)  # (2,)
        all_limits = torch.cat([arm_limits, drv_limits.unsqueeze(0)], dim=0)  # (8, 2)
        self._joint_lims = all_limits.T.unsqueeze(0)  # (1, 2, 8): [lower; upper]

        # Driver joint qpos / dofadr (for gripper obs and vel)
        self._drv_qpos_idx = int(self._model.jnt_qposadr[drv_jnt_id])
        self._drv_dofadr = int(self._model.jnt_dofadr[drv_jnt_id])

        # Fixed tensors
        self._tcp_offset = torch.tensor(self._TCP_OFFSET, dtype=torch.float32).unsqueeze(0)  # (1, 7)
        self._gripper_lim = torch.tensor([[0.0, self._GRIPPER_MAX]], dtype=torch.float32)     # (1, 2)

        # Gymnasium spaces
        n_q = self._N_ARM_JOINTS + 1  # 8
        single_obs = gym.spaces.Dict({
            "ee_pose_b":    gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "tcp_pose_b":   gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "jacobians":    gym.spaces.Box(-np.inf, np.inf, (6, self._N_ARM_JOINTS), np.float32),
            "joint_pos":    gym.spaces.Box(-np.pi, np.pi, (n_q,), np.float32),
            "joint_vel":    gym.spaces.Box(-10.0, 10.0, (n_q,), np.float32),
            "tcp_offset":   gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "gripper":      gym.spaces.Box(0.0, self._GRIPPER_MAX, (1,), np.float32),
            "gripper_lim":  gym.spaces.Box(0.0, self._GRIPPER_MAX, (2,), np.float32),
            "joint_lims":   gym.spaces.Box(-np.pi, np.pi, (2, n_q), np.float32),
            # PlaceSkill reads tcp_wrench_b for contact detection; no force sensor
            # in this MJCF so we always return zeros (contact check never fires).
            "tcp_wrench_b": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
        })
        single_act = gym.spaces.Box(-np.pi, np.pi, (n_q,), np.float32)
        self.single_observation_space = single_obs
        self.single_action_space = single_act
        self.observation_space = _batch_space(single_obs)
        self.action_space = _batch_space(single_act)

        # Specs for GymVectorInterface / skill_lib compatibility.
        # Pin device to CPU — MuJoCo always runs on CPU, and DifferentialIKController
        # is constructed from obs_spec.device, so we must keep everything on the same device.
        _cpu = torch.device("cpu")
        self._obs_spec: ObservationSpec = IK_EE_SPEC_BATCHED.bind(
            n_joints=n_q,
            n_arm_joints=self._N_ARM_JOINTS,
            n_gripper_joints=1,
        ).replace(device=_cpu)
        self._action_spec: ActionSpec = JOINT_VEL_SPEC.bind(n_joints=n_q).replace(device=_cpu)

        self._last_obs: dict[str, torch.Tensor] | None = None
        self._step_callback: Callable[[], None] | None = None
        self._reset_to_home()

    # -----------------------------------------------------------------------
    # GymVectorInterface protocol
    # -----------------------------------------------------------------------

    @property
    def unwrapped(self) -> InspectionMjEnv:
        """Return self (no wrapper layers)."""
        return self

    @property
    def obs_spec(self) -> ObservationSpec:
        """IKEE observation spec (batched, n_envs=1)."""
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec:
        """8-D joint-velocity action spec (batched)."""
        return self._action_spec

    def coerce_obs_spec(self, obs_spec: str | ObservationSpec) -> ObservationSpec:
        """Return the obs spec for name ``'ik_ee'``, or pass through an ObservationSpec unchanged."""
        if isinstance(obs_spec, str):
            if obs_spec == "ik_ee":
                return self._obs_spec
            raise ValueError(f"Observation spec {obs_spec!r} not supported; only 'ik_ee' is available.")
        return obs_spec

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        """Return ``True`` iff ``obs_spec.name == 'ik_ee'``."""
        return obs_spec.name == "ik_ee"

    @property
    def robot_base_world_pos(self) -> np.ndarray:
        """World-frame XYZ of the robot base link, shape (3,).

        Read live from MuJoCo data so it stays correct even if the base is
        moved between resets.  Used by InspectSkill to convert block world
        poses to robot-base frame before computing IK targets.
        """
        return self._data.xpos[self._base_body_id].copy()

    def supports_action_spec(self, action_spec: ActionSpec) -> bool:
        """Accept any (1, 8) action tensor; spec is treated as advisory."""
        return True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        """Reset to the Gen3 home keyframe and return the initial obs dict."""
        self._reset_to_home()
        obs = self._get_obs()
        self._last_obs = obs
        return obs, {}

    def step(
        self,
        actions: torch.Tensor | np.ndarray,
        action_spec: ActionSpec | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply a joint-position action and advance the simulation by one step.

        Args:
            actions: Shape ``(1, 8)`` or ``(8,)``.  First 7 values are arm
                joint position targets in radians.  Value 7 is
                ``gripper_normalized ∈ [0, 1]``.
            action_spec: Ignored (present for protocol compatibility).

        Returns:
            ``(obs, reward, terminated, truncated, info)`` where ``reward``,
            ``terminated``, and ``truncated`` are all zero / False — task
            success is determined by the planning layer, not here.

        """
        a = _to_numpy_1d(actions)
        _N_SIM_STEPS = 100
        _MAX_DELTA = 0.05  # rad per IK step
        current_arm = self._data.qpos[
            self._arm_qpos_start : self._arm_qpos_start + self._N_ARM_JOINTS
        ].copy()
        target_arm = a[: self._N_ARM_JOINTS]
        delta = target_arm - current_arm
        max_abs = float(np.abs(delta).max()) + 1e-8
        scale = min(1.0, _MAX_DELTA / max_abs)
        self._data.ctrl[: self._N_ARM_JOINTS] = current_arm + scale * delta
        self._data.ctrl[self._N_ARM_JOINTS] = float(a[self._N_ARM_JOINTS]) * self._GRIPPER_CTRL_MAX
        for _ in range(_N_SIM_STEPS):
            mujoco.mj_step(self._model, self._data)
        # mj_forward brings derived quantities (xpos, xquat, …) up to date.
        mujoco.mj_forward(self._model, self._data)

        obs = self._get_obs()
        self._last_obs = obs
        if self._step_callback is not None:
            self._step_callback()
        return (
            obs,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    def get_observation(self, obs_spec: ObservationSpec | None = None) -> dict[str, torch.Tensor]:
        """Return the latest observation (call ``reset()`` first)."""
        if self._last_obs is None:
            raise ValueError("Call reset() before get_observation().")
        return self._last_obs

    def set_step_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a function called after every physics step (e.g. ``viewer.sync``)."""
        self._step_callback = callback

    def get_block_world_pos(self, block_name: str) -> np.ndarray:
        """Return the world-frame XYZ of a block body, shape ``(3,)``.

        Reads live from ``MjData.xpos`` so it reflects the current physics state.
        Used to sync the scene-graph after a physical place action.
        """
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, block_name)
        if body_id == -1:
            raise ValueError(f"No body named {block_name!r} in MuJoCo model")
        return self._data.xpos[body_id].copy()

    def capture_wrist_cam(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render a frame from the wrist-mounted camera.

        The Gen3 MJCF includes a ``wrist`` camera on the bracelet_link body
        (fovy ≈ 42°, 640×480).  This method renders it off-screen and returns
        an RGB image.  Caller is responsible for saving the array.

        Returns:
            ``(height, width, 3)`` uint8 RGB array.

        """
        renderer = mujoco.Renderer(self._model, height=height, width=width)
        renderer.update_scene(self._data, camera="wrist")
        pixels = renderer.render()
        renderer.close()
        return pixels

    @property
    def mj_model(self) -> mujoco.MjModel:
        """The underlying ``MjModel`` (e.g. for launching a passive viewer)."""
        return self._model

    @property
    def mj_data(self) -> mujoco.MjData:
        """The underlying ``MjData`` (e.g. for launching a passive viewer)."""
        return self._data

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _reset_to_home(self) -> None:
        if self._model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, self._model.key("home").id)
        else:
            mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        # mj_resetDataKeyframe sets qpos but leaves ctrl at zero.  The position
        # servos reference ctrl as their setpoint, so we must sync ctrl to the
        # keyframe joint positions — otherwise the first env.step() clamps a
        # delta relative to 0, driving the arm away from home.
        self._data.ctrl[: self._N_ARM_JOINTS] = self._data.qpos[
            self._arm_qpos_start : self._arm_qpos_start + self._N_ARM_JOINTS
        ].copy()

    def _get_obs(self) -> dict[str, torch.Tensor]:
        """Compute the IKEE_Obs dict from the current MuJoCo state.

        All tensors have a leading batch dimension of 1.
        """
        d = self._data
        m = self._model

        # EE and base poses in world frame (MuJoCo xquat convention: w, x, y, z)
        ee_pos_w = torch.tensor(d.xpos[self._ee_body_id], dtype=torch.float32).unsqueeze(0)    # (1, 3)
        ee_quat_w = torch.tensor(d.xquat[self._ee_body_id], dtype=torch.float32).unsqueeze(0)  # (1, 4)
        base_pos_w = torch.tensor(d.xpos[self._base_body_id], dtype=torch.float32).unsqueeze(0)
        base_quat_w = torch.tensor(d.xquat[self._base_body_id], dtype=torch.float32).unsqueeze(0)

        # EE pose in robot base frame: T_base^-1 * T_ee
        ee_pos_b, ee_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=1)  # (1, 7)

        # TCP pose: compose EE pose with fixed TCP offset
        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, self._tcp_offset[:, :3])
        tcp_quat_b = quat_mul(ee_quat_b, self._tcp_offset[:, 3:])
        tcp_pose_b = torch.cat([tcp_pos_b, tcp_quat_b], dim=1)  # (1, 7)

        # Jacobian for the EE body; extract the 7 arm-DOF columns
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, self._ee_body_id)
        col_s, col_e = self._arm_qvel_start, self._arm_qvel_start + self._N_ARM_JOINTS
        jac_np = np.vstack([jacp[:, col_s:col_e], jacr[:, col_s:col_e]])  # (6, 7)
        jacobians = torch.tensor(jac_np, dtype=torch.float32).unsqueeze(0)  # (1, 6, 7)

        # Joint positions: 7 arm + 1 driver
        arm_q = torch.tensor(
            d.qpos[self._arm_qpos_start : self._arm_qpos_start + self._N_ARM_JOINTS], dtype=torch.float32
        )
        drv_q = torch.tensor([d.qpos[self._drv_qpos_idx]], dtype=torch.float32)
        joint_pos = torch.cat([arm_q, drv_q]).unsqueeze(0)  # (1, 8)

        # Joint velocities: 7 arm + 1 driver
        arm_v = torch.tensor(
            d.qvel[self._arm_qvel_start : self._arm_qvel_start + self._N_ARM_JOINTS], dtype=torch.float32
        )
        drv_v = torch.tensor([d.qvel[self._drv_dofadr]], dtype=torch.float32)
        joint_vel = torch.cat([arm_v, drv_v]).unsqueeze(0)  # (1, 8)

        # Gripper: raw driver joint position in [0, GRIPPER_MAX]
        gripper = drv_q.unsqueeze(0)  # (1, 1)

        return {
            "ee_pose_b":    ee_pose_b,                                        # (1, 7)
            "tcp_pose_b":   tcp_pose_b,                                       # (1, 7)
            "jacobians":    jacobians,                                        # (1, 6, 7)
            "joint_pos":    joint_pos,                                        # (1, 8)
            "joint_vel":    joint_vel,                                        # (1, 8)
            "tcp_offset":   self._tcp_offset,                                 # (1, 7)
            "gripper":      gripper,                                          # (1, 1)
            "gripper_lim":  self._gripper_lim,                                # (1, 2)
            "joint_lims":   self._joint_lims,                                 # (1, 2, 8)
            "tcp_wrench_b": torch.zeros(1, 6, dtype=torch.float32),          # (1, 6)
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch_space(space: gym.Space) -> gym.Space:
    """Return ``space`` with a leading batch dimension of 1."""
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: _batch_space(v) for k, v in space.items()})
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.expand_dims(space.low, 0),
            high=np.expand_dims(space.high, 0),
            dtype=space.dtype,
        )
    raise TypeError(f"Cannot batch space type {type(space)}")


def _to_numpy_1d(actions: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a batched action tensor or array to a 1-D numpy array."""
    if isinstance(actions, torch.Tensor):
        return actions.cpu().numpy().flatten()
    return np.asarray(actions).flatten()
