"""InspectSkill: moves arm to an inspection viewpoint above a block  """

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from skillet.core.checked_skill import CheckedSkill, FailureReason, SkillResult
from skillet.core.skill import BatchedSkill, SkillStatusCodes
from skillet.envs.compatibility import GymVectorInterface
from skillet.scene.base import Scene

if TYPE_CHECKING:
    pass


class InspectSkill(CheckedSkill):
    """Moves the arm to a top-face inspection viewpoint above a target block.

    The resulting position must be in the robot workspace.
    """

    _DEFAULT_STANDOFF_M: float = 0.20

    def __init__(
        self,
        scene: Scene,
        env: GymVectorInterface | None = None,
        reach_skill: BatchedSkill | None = None,
        block_half_extents: np.ndarray | None = None,
        standoff_m: float = _DEFAULT_STANDOFF_M,
        robot_base_world_pos: np.ndarray | None = None,
    ) -> None:
        """Initialize.

        Args:
            scene: The world-model scene; used for pose lookups.
            env: Optional MuJoCo env; when provided, execute() runs physical motion.
            reach_skill: Pre-built reach skill; required when env is provided.
            block_half_extents: Half-lengths [hx, hy, hz] of blocks in metres.
                Defaults to [0.022, 0.022, 0.022] (4.4 cm cube, matching CUBE_SIZE).
            standoff_m: Distance from the block top face to the TCP target in metres.
            robot_base_world_pos: World-frame position [x, y, z] of the robot base
                link.  When provided, block world-frame poses are converted to robot
                base frame before computing the IK target.  Required when scene
                objects store poses in world frame (the typical case for MuJoCo envs).

        """
        self._scene = scene
        self._env = env
        self._reach_skill = reach_skill
        self._block_half_extents = (
            block_half_extents if block_half_extents is not None
            else np.array([0.022, 0.022, 0.022])
        )
        self._standoff_m = standoff_m
        self._robot_base_world_pos = robot_base_world_pos
        self._target_block_id: int | None = None
        self._status: int = SkillStatusCodes.UNINITIATED

    def set_target(self, block_id: int) -> None:
        """Set the target block id before calling preconditions or execute."""
        self._target_block_id = block_id

    # ------------------------------------------------------------------
    # CheckedSkill contract
    # ------------------------------------------------------------------

    def preconditions(self, world: Scene) -> bool:
        """Return True iff the target block is reachable from the front."""
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        if not block.is_pose_known():
            return False
        return float(block.pose[0]) > 0.0

    def postconditions(self, world: Scene) -> bool:
        """Return True iff the block is still visible after the approach.
        """
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        return block.is_pose_known()

    # ------------------------------------------------------------------
    # Convenience method (used by tests and task scripts)
    # ------------------------------------------------------------------

    def execute(self, scene: Scene) -> SkillResult:
        """Check preconditions and optionally execute physical motion.

        When env and reach_skill were provided at construction, drives the arm
        to the top-face inspection viewpoint via the reach primitive.
        Without them, returns ok() immediately.
        """
        if not self.preconditions(scene):
            return SkillResult.fail(FailureReason.PRECONDITION_NOT_MET)
        if self._env is None or self._reach_skill is None:
            return SkillResult.ok()
        viewpoint_xyz = self._compute_viewpoint(scene)
        obs = self._env.get_observation()  # type: ignore[attr-defined]
        params = _xyz_to_reach_params(viewpoint_xyz)
        self._reach_skill.initiate(obs, params)
        return self._run_skill_loop()

    # ------------------------------------------------------------------
    # Internal helpers (motion path)
    # ------------------------------------------------------------------

    def _compute_viewpoint(self, scene: Scene) -> np.ndarray:
        """Return TCP target position (robot base frame) above the target block."""
        block = scene.get_objects_from_id([self._target_block_id])[0]
        pos = block.pose[:3].cpu().numpy()
        # Scene objects store poses in world frame; convert to robot base frame
        # so the IK target matches the tcp_pose_b coordinate frame.
        if self._robot_base_world_pos is not None:
            pos = pos - self._robot_base_world_pos
        hz = float(self._block_half_extents[2])
        target = pos.copy()
        target[2] += hz + self._standoff_m
        return target

    def _run_skill_loop(self, max_steps: int = 200) -> SkillResult:
        """Step the env until the reach skill succeeds, fails, or budget runs out."""
        for _ in range(max_steps):
            obs = self._env.get_observation()  # type: ignore[attr-defined]
            action = self._reach_skill.get_action(obs)
            self._env.step(action, None)  # type: ignore[arg-type]
            status_val = int(self._reach_skill.status[0].item())
            if status_val == SkillStatusCodes.SUCCESS:
                return SkillResult.ok()
            if status_val == SkillStatusCodes.FAILED:
                return SkillResult.fail(FailureReason.IK_FAILURE)
        return SkillResult.fail(FailureReason.TIMEOUT)

    # ------------------------------------------------------------------
    # SingleSkill interface — stubs; wired to motion primitive
    # ------------------------------------------------------------------

    @property
    def policy(self) -> Any:
        raise NotImplementedError("motion policy not wired")

    @property
    def status(self) -> int:
        return self._status

    def initiate(self, obs: Any, params: Any) -> None:
        self._target_block_id = int(params)
        self._status = SkillStatusCodes.RUNNING

    def get_action(self, obs: Any) -> Any:
        raise NotImplementedError("motion action not wired")


def _xyz_to_reach_params(xyz: np.ndarray) -> torch.Tensor:
    """Build a ``(1, 6)`` XYZRPY reach parameter tensor.

    The inspection orientation is "pointing down" (roll=pi, pitch=0, yaw=0),
    matching the Gen3 home posture above a horizontal surface.  Using full
    6-DOF IK avoids the TCP-offset drift that occurs when position-only IK
    recomputes the EE target once at reset while the EE orientation changes.
    """
    params = torch.zeros(1, 6, dtype=torch.float32)
    params[0, 0] = float(xyz[0])
    params[0, 1] = float(xyz[1])
    params[0, 2] = float(xyz[2])
    params[0, 3] = math.pi  # roll = pi → gripper pointing down
    return params