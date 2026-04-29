"""isaac_env_wrapper.py.

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import Any, overload

import gymnasium as gym
import torch
from jaxtyping import Bool, Float
from typing_extensions import override

from skillet.core import ObservationSpec
from skillet.core.env import BatchedEnvironment, TSpecObs
from skillet.core.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from skillet.core.spaces import ActionSpec
from skillet.envs.compatibility import DirectRlInterface, SkilletGymEnv
from skillet.envs.specs import (
    GRIPPER_SPEC_BATCHED,
    IK_EE_SPEC_BATCHED,
    JOINT_VEL_SPEC,
    JOINT_VEL_SPEC_BATCHED,
    OSC_SPEC_BATCHED,
    RGBD_GRIPPER_SPEC_BATCHED,
    RGBD_SPEC_BATCHED,
    TCP_CART_SPEC,
    TCP_CART_SPEC_BATCHED,
    TCP_QUAT_SPEC,
    TWIST_SPEC_BATCHED,
    TWIST_TCP_SPEC,
    BxM_Action,
    BxN_Obs,
)


class SkilletEnv(
    BatchedEnvironment[BxN_Obs, BxM_Action],
    DirectRlInterface,
):
    """An environment that interfaces with ROS2/Mujoco/IsaacLab and is compatible with Skillet and IsaacLab DirectRlEnv.

    The environment is batched.
    """

    def __init__(self, env: SkilletGymEnv) -> None:
        """Initialize the environment.

        Args:
            env: SkilletGymEnv environment
        Returns:
            None

        """
        env = env.unwrapped
        self._env: SkilletGymEnv = env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.single_observation_space = env.unwrapped.single_observation_space
        self.single_action_space = env.unwrapped.single_action_space

        # Robot specific information
        self._joint_ids = self.cfg.joint_ids
        self._tcp_offset = self.cfg.tcp_offset
        self._ee_link_name = self.cfg.ee_link_name
        self._base_link_name = self.cfg.base_link_name
        self._gripper_joint_names = self.cfg.gripper_joint_names

        self._tcp_offset = torch.as_tensor(self._tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self._last_obs = None

        # Define the obseravation and action specifications
        self._spec_args = {
            "is_torch": True,
            "is_batched": True,
            "n_envs": -1,
            "device": self.device,
        }
        if hasattr(env, "single_observation_space") and hasattr(env, "single_action_space"):
            obs_space = env.single_observation_space
            action_space = env.single_action_space
        else:
            # if using batched spaces, must set n_envs to the number of environments
            obs_space = env.observation_space
            action_space = env.action_space
            self._spec_args["n_envs"] = env.num_envs
        assert isinstance(obs_space, gym.spaces.Dict)
        assert "policy" in obs_space.keys()  # noqa: SIM118
        self.obs_spec_policy = ObservationSpec[Float[torch.Tensor, "b ..."]](
            name="policy",
            space=obs_space["policy"],
        ).replace(**self._spec_args)
        """Specification of the vector observation passed to a low level policy"""
        self.obs_spec_state = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
            name="state",
            space=obs_space,
        ).replace(**self._spec_args)
        """Specification of the raw dictionary environment state"""
        self.obs_spec_rgbd = RGBD_SPEC_BATCHED.bind(height=480, width=640).replace(**self._spec_args)
        self.obs_spec_rgbd_grip = RGBD_GRIPPER_SPEC_BATCHED.bind(
            height=480, width=640, n_gripper_joints=len(self._gripper_joint_names)
        ).replace(**self._spec_args)
        self.obs_spec_gripper = GRIPPER_SPEC_BATCHED.bind(n_gripper_joints=len(self._gripper_joint_names)).replace(
            **self._spec_args
        )
        """Specification of RGB-D observations and metadata. Bound to the height and width of the RGB-D camera."""
        self.obs_spec_ikee = IK_EE_SPEC_BATCHED.bind(
            n_joints=len(self._joint_ids),
            n_arm_joints=len(self._joint_ids[:-1]),
            n_gripper_joints=len(self._gripper_joint_names),
        ).replace(**self._spec_args)
        """Specification of IK-EE observations."""
        self.obs_spec_osc = OSC_SPEC_BATCHED.bind(
            n_joints=len(self._joint_ids),
            n_arm_joints=len(self._joint_ids[:-1]),
            n_gripper_joints=len(self._gripper_joint_names),
        ).replace(**self._spec_args)
        """Specification of OSC observations."""
        self.obs_spec_twist_tcp = TWIST_SPEC_BATCHED.bind(
            n_gripper_joints=len(self._gripper_joint_names), n_joints=len(self._joint_ids)
        ).replace(**self._spec_args)
        """Specification of Twist TCP observations."""
        self.obs_spec_joints_vel = JOINT_VEL_SPEC_BATCHED.bind(
            n_gripper_joints=len(self._gripper_joint_names), n_joints=len(self._joint_ids)
        ).replace(**self._spec_args)
        """Specification for joints"""
        self.obs_spec_tcp_cart = TCP_CART_SPEC_BATCHED.bind(
            n_gripper_joints=len(self._gripper_joint_names), n_joints=len(self._joint_ids)
        ).replace(**self._spec_args)

        self.action_spec_state = ActionSpec[Float[torch.Tensor, "b ..."]](
            name="state",
            space=action_space,
        ).replace(**self._spec_args)

        self.action_spec_joints_vel = (
            JOINT_VEL_SPEC.replace(**self._spec_args).bind(n_joints=len(self._joint_ids)).replace(device=self.device)
        )
        self.action_spec_twist_tcp = (
            TWIST_TCP_SPEC.replace(**self._spec_args)
            .bind(n_gripper_joints=len(self._env.cfg.gripper_joint_names))
            .replace(device=self.device)
        )
        self.action_spec_tcp_quat = (
            TCP_QUAT_SPEC.replace(**self._spec_args)
            .bind(n_gripper_joints=len(self._env.cfg.gripper_joint_names), n_joints=len(self._joint_ids))
            .replace(device=self.device)
        )
        self.action_spec_tcp_cart = (
            TCP_CART_SPEC.replace(**self._spec_args)
            .bind(n_gripper_joints=len(self._env.cfg.gripper_joint_names), n_joints=len(self._joint_ids))
            .replace(device=self.device)
        )

    # ==================== DirectRlInterface ====================
    @property
    @override
    def cfg(self) -> dict | object:
        return self._env.cfg

    @property
    @override
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    @override
    def device(self) -> torch.device | str:
        return self._env.device

    @property
    @override
    def max_episode_length(self) -> int:
        return self._env.max_episode_length

    @property
    @override
    def episode_length_buf(self) -> torch.Tensor:
        return self._env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._env.unwrapped.episode_length_buf = value

    @override
    def _get_observations(self) -> Mapping[str, torch.Tensor]:
        return self.obs_spec_state.cast(self._last_obs)

    @override
    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        self._env._reset_idx(env_ids)
        self._last_obs = None

    @property
    @override
    def unwrapped(self) -> DirectRlInterface:
        # self satisfies the DirectRlInterface
        return self._env

    # ==================== Skillet Environment ====================
    @property
    @override
    def obs_spec(self) -> ObservationSpec[BxN_Obs]:
        return self.obs_spec_state

    @property
    @override
    def action_spec(self) -> ActionSpec[BxM_Action]:
        return self.action_spec_joints_vel

    @override
    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return obs_spec.name in [
            self.obs_spec_policy.name,
            self.obs_spec_state.name,
            self.obs_spec_rgbd.name,
            self.obs_spec_ikee.name,
            self.obs_spec_osc.name,
            self.obs_spec_gripper.name,
            self.obs_spec_rgbd_grip.name,
            self.obs_spec_joints_vel.name,
            self.obs_spec_tcp_cat.name,
            self.obs_spec_twist_tcp.name,
        ]

    @override
    def supports_action_spec(self, action_spec: ActionSpec) -> bool:
        return action_spec.name in [
            self.action_spec_joints_vel.name,
            self.action_spec_twist_tcp.name,
            self.action_spec_tcp_cart.name,
            self.action_spec_tcp_quat.name,
            self.action_spec_state.name,
        ]

    @override
    def coerce_obs_spec(self, obs_spec: str | ObservationSpec[Any]) -> ObservationSpec[Any]:
        for spec in [
            self.obs_spec_policy,
            self.obs_spec_state,
            self.obs_spec_rgbd,
            self.obs_spec_ikee,
            self.obs_spec_osc,
            self.obs_spec_gripper,
            self.obs_spec_rgbd_grip,
            self.obs_spec_joints_vel,
        ]:
            if spec.name == obs_spec:
                return spec
            if isinstance(obs_spec, str) and obs_spec == spec.name:
                return spec
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    # ==================== Public methods ====================

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[BxN_Obs, dict]:
        obs_dict, info = self._env.reset(seed=seed, options=options)
        self._last_obs = obs_dict

        return obs_dict, info

    @overload
    def get_observation(self) -> BxN_Obs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...
    @override
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs] | None = None) -> Any:
        if self._last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is None:
            obs_spec = self.obs_spec
        if obs_spec.is_batched:
            obs_spec = obs_spec.with_n_envs(self.num_envs)

        if obs_spec.name == "policy":
            return obs_spec.cast(self._last_obs["policy"])
        if obs_spec.name == "rgb-d" or obs_spec.name == "rgbd-gripper":
            latest = self._env._get_latest_rgbd()
            if obs_spec.name == "rgbd-gripper":
                latest["tcp_pose_b"] = self._get_tcp_pose_b(ee_link=self._ee_link_name)
                latest["gripper"] = self._get_gripper_state(gripper_joints=self._gripper_joint_names)

            return obs_spec.cast(latest)
        if obs_spec.name == "state":
            return obs_spec.cast(self._last_obs)
        if obs_spec.name == "ik_ee":
            return obs_spec.cast(
                {
                    "joint_pos": self._get_joint_positions(joint_ids=self._joint_ids),
                    "joint_vel": self._get_joint_velocities(joint_ids=self._joint_ids),
                    "tcp_offset": self._tcp_offset,
                    "jacobians": self._get_jacobians(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "ee_pose_b": self._get_ee_pose_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self._gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                    "joint_lims": self._get_joint_lims(joint_ids=self._joint_ids),
                }
            )
        if obs_spec.name == "osc_ee":
            return obs_spec.cast(
                {
                    "joint_pos": self._get_joint_positions(joint_ids=self._joint_ids),
                    "joint_vel": self._get_joint_velocities(joint_ids=self._joint_ids),
                    "ee_vel_b": self._get_ee_vel_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "tcp_offset": self._tcp_offset,
                    "jacobians": self._get_jacobians(
                        ee_link=self._ee_link_name, base_link=self._base_link_name, arm_joint_ids=self._joint_ids[:7]
                    ),
                    "ee_pose_b": self._get_ee_pose_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self._gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                    "joint_lims": self._get_joint_lims(joint_ids=self._joint_ids),
                    "mass_matrix": self._get_mass_matrices(arm_joint_ids=self._joint_ids[:7]),
                    "joint_gravity": self._get_joint_gravity(arm_joint_ids=self._joint_ids[:7]),
                    "joint_centers": self._get_joint_centers(arm_joint_ids=self._joint_ids[:7]),
                }
            )
        if obs_spec.name == "twist_tcp":
            return obs_spec.cast(
                {
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self._gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                    "dt": torch.tensor([self._env.step_dt]).expand(self.num_envs),
                    "ee_vel_b": self._get_ee_vel_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "joint_vel": self._get_joint_velocities(joint_ids=self._joint_ids),
                    "joint_pos": self._get_joint_positions(joint_ids=self._joint_ids),
                    "joint_eff": self._get_joint_efforts(joint_ids=self._joint_ids),
                    "prev_actions": self._env._prev_actions,
                }
            )
        if obs_spec.name == "tcp_cart":
            return obs_spec.cast(
                {
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self._gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                    "dt": torch.tensor([self._env.step_dt]).expand(self.num_envs),
                    "joint_vel": self._get_joint_velocities(joint_ids=self._joint_ids),
                    "joint_eff": self._get_joint_efforts(joint_ids=self._joint_ids),
                }
            )
        if obs_spec.name == "gripper":
            return obs_spec.cast(
                {
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                }
            )
        if obs_spec.name == "joints_vel":
            return obs_spec.cast(
                {
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self._gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                    "dt": torch.tensor([self._env.step_dt]).expand(self.num_envs),
                    "ee_vel_b": self._get_ee_vel_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "joint_vel": self._get_joint_velocities(joint_ids=self._joint_ids),
                    "joint_pos": self._get_joint_positions(joint_ids=self._joint_ids),
                    "joint_eff": self._get_joint_efforts(joint_ids=self._joint_ids),
                    "joint_lims": self._get_joint_lims(joint_ids=self._joint_ids),
                    "prev_actions": self._env._prev_actions,
                }
            )
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    @override
    def get_state(self) -> Mapping[str, torch.Tensor]:
        return self.get_observation(self.obs_spec_state)

    @override
    def step(self, action: BxM_Action, action_spec: ActionSpec[Any] | None = None) -> tuple[
        BxN_Obs,
        Float[torch.Tensor, "b"],  # noqa: F821
        Bool[torch.Tensor, "b"],  # noqa: F821
        Bool[torch.Tensor, "b"],  # noqa: F821
        Mapping[str, torch.Tensor],
    ]:
        """Step through the environment.

        Args:
            action: The action tensor of shape (num_envs, num_actions)
            action_spec: Skillet Action spec of the action: TODO: currently assumes all actions on batch have same spec

        Returns:
            A tuple containing the observation of observations tensor (num_envs, obs_dim) and info dictionary

        """
        action = action.to(self.device)
        obs_dict, reward, term, trunc, info = self._env.step(action, action_spec=action_spec)
        self.last_obs = obs_dict
        self._last_obs: dict[str, torch.Tensor] = obs_dict

        return obs_dict, reward, term, trunc, info

    # ==================== Helper functions ====================

    def _get_joint_positions(self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None) -> torch.Tensor:
        """Return the joint positions.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (num_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if joint_ids is None:
            joint_ids = self._joint_ids
        return self._env._joint_positions[:, joint_ids][env_ids]

    def _get_joint_velocities(self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None) -> torch.Tensor:
        """Return the joint velocities.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (num_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if joint_ids is None:
            joint_ids = self._joint_ids

        return self._env._joint_velocities[:, joint_ids][env_ids]

    def _get_joint_efforts(self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None) -> torch.Tensor:
        """Return the joint efforts.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (num_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if joint_ids is None:
            joint_ids = self._joint_ids
        return self._env._joint_efforts[:, joint_ids][env_ids]

    def _get_jacobians(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "end_effector_link",
        base_link: str = "base_link",
        arm_joint_ids: list | None = None,
    ) -> torch.Tensor:
        """Return the jacobians.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of jacobians of shape (num_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:-1]

        ee_link_idx = self._env._find_link_idx(ee_link)
        base_link_idx = self._env._find_link_idx(base_link)

        robot_base_pose_w = self._env._robot_body_pose_w[env_ids, base_link_idx]

        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))

        jacobian = self._env._jacobians[env_ids, ee_link_idx][:, :, arm_joint_ids]

        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        return jacobian

    def _get_tcp_pose_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "end_effector_link",
    ) -> torch.Tensor:
        """Get the TCP pose of the robot in the robot base frame.

        Args:
            env_ids: environment ids to tcp pose in XYZ
            ee_link: string for the name of the end effector link

        Returns:
            Tensor in shape (N,7) with 7 in (X,Y,Z,Quat)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        ee_link_idx = self._env._find_link_idx(ee_link)

        ee_pose_w = self._env._robot_body_pose_w[env_ids, ee_link_idx]
        root_pose_w = self._env._robot_root_pose_w[env_ids]

        ee_pos_b = quat_apply_inverse(root_pose_w[:, 3:7], ee_pose_w[:, 0:3] - root_pose_w[:, 0:3])
        ee_quat_b = quat_mul(quat_inv(root_pose_w[:, 3:7]), ee_pose_w[:, 3:7])

        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, self._tcp_offset[env_ids, 0:3])
        tcp_quat_b = quat_mul(ee_quat_b, self._tcp_offset[env_ids, 3:7])

        return torch.concatenate((tcp_pos_b, tcp_quat_b), dim=1).to(torch.float32)

    def _get_gripper_state(
        self, env_ids: torch.Tensor | None = None, gripper_joints: str = ["robotiq_85_left_knuckle_joint"]
    ) -> torch.Tensor:
        """Get the gripper state of the robot."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        gripper_joint_idxs = [self._env._find_joint_idx(j) for j in gripper_joints]
        gripper_pos = self._env._joint_positions[env_ids, gripper_joint_idxs]

        return gripper_pos if gripper_pos.ndim == 2 else gripper_pos.unsqueeze(-1)

    def _get_ee_pose_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "end_effector_link",
        base_link: str = "base_link",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the end effector pose of the robot in the robot's base frame.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm

        Returns:
            The robot EE position in shape (N, 3) relative to the base of the robot
            The robot EE orientation in shape (N, 4) relative to the base of the robot

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ee_link_idx = self._env._find_link_idx(ee_link)
        base_link_idx = self._env._find_link_idx(base_link)

        # Get the pose of the end effector and base in the world frame
        # (B, 7) with (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        robot_ee_pose_w = self._env._robot_body_pose_w[env_ids, ee_link_idx]
        robot_base_pose_w = self._env._robot_body_pose_w[env_ids, base_link_idx]

        # Compute the end effector pose in the robot base frame
        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
            robot_base_pose_w[:, :3],
            robot_base_pose_w[:, 3:7],
            robot_ee_pose_w[:, :3],
            robot_ee_pose_w[:, 3:7],
        )

        return torch.cat((robot_ee_pos_b, robot_ee_quat_b), dim=1).to(torch.float32)

    def _get_gripper_lims(
        self, env_ids: torch.Tensor | None = None, gripper_joints: str = ["robotiq_85_left_knuckle_joint"]
    ) -> torch.Tensor:
        """Get the gripper limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ
            gripper_joints: list of strings for the name of the gripper joint

        Returns:
            A tensor of shape (N, 2) for the gripper lower/upper limits.

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        gripper_joint_idxs = [self._env._find_joint_idx(j) for j in gripper_joints]

        gripper_low = self._env._robot_dof_lower_limits[gripper_joint_idxs]
        gripper_high = self._env._robot_dof_upper_limits[gripper_joint_idxs]
        return torch.cat(
            (
                gripper_low.unsqueeze(0).expand(env_ids.shape[0], 1),
                gripper_high.unsqueeze(0).expand(env_ids.shape[0], 1),
            ),
            dim=1,
        ).to(torch.float32)

    def _get_joint_lims(self, env_ids: torch.Tensor | None = None, joint_ids: torch.Tensor = None) -> torch.Tensor:
        """Get the joint limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ
            joint_ids: joint IDs to get the limits of

        Returns:
            A tensor of shape (N, 2) for the gripper lower/upper limits.

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if joint_ids is None:
            joint_ids = self.cfg.joint_ids
        return (
            torch.cat(
                (
                    self._env._robot_dof_lower_limits[joint_ids].unsqueeze(0),
                    self._env._robot_dof_upper_limits[joint_ids].unsqueeze(0),
                ),
                dim=0,
            )
            .unsqueeze(0)
            .repeat(env_ids.shape[0], 1, 1)
        )

    def _get_mass_matrices(
        self,
        env_ids: torch.Tensor | None = None,
        arm_joint_ids: list | None = None,
    ) -> torch.Tensor:
        """Return the mass matrices.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of mass matrices

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:7]

        return self._env._mass_matrices[:, arm_joint_ids, :][:, :, arm_joint_ids][env_ids]

    def _get_joint_gravity(
        self,
        env_ids: torch.Tensor | None = None,
        arm_joint_ids: list | None = None,
    ) -> torch.Tensor:
        """Return the joint gravity of the arm joints.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of mass matrices

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:7]
        return self._env._gravity_vector[:, arm_joint_ids][env_ids]

    def _get_ee_vel_b(
        self, env_ids: torch.Tensor = None, ee_link: str = "end_effector_link", base_link: str = "base_link"
    ) -> torch.Tensor:
        """Compute the velocity of the end effector relative to the robot base.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot

        """
        # Compute the current velocity of the end-effector
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        ee_link_idx = self._env._find_link_idx(ee_link)
        base_link_idx = self._env._find_link_idx(base_link)

        ee_vel_w = self._env._robot_body_vel_w[
            env_ids, ee_link_idx, :
        ]  # Extract end-effector velocity in the world frame
        root_vel_w = self._env._robot_body_vel_w[env_ids, base_link_idx, :]  # Extract root velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_apply_inverse(
            self._env._robot_body_pose_w[env_ids, base_link_idx][:, 3:7],
            relative_vel_w[:, 0:3],
        )  # From world to root frame
        ee_ang_vel_b = quat_apply_inverse(
            self._env._robot_body_pose_w[env_ids, base_link_idx][:, 3:7],
            relative_vel_w[:, 3:6],
        )
        return torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    def _get_joint_centers(self, env_ids: torch.Tensor = None, arm_joint_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the joint centers of the arm.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of joint centers

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:7]
        return self._env._joint_centers[:, arm_joint_ids][env_ids]
