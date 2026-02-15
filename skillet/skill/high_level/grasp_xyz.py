"""A grasp skill for grasping an object at an xyz and yaw position."""

from enum import IntEnum
from typing import Generic

import numpy as np
import torch
from jaxtyping import Int

from skillet.core.math import euler_xyz_from_quat, quat_error_magnitude, quat_from_euler_xyz
from skillet.core.policy import BatchedPPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike


class GraspStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    IDLE = 0
    """The skill is idle."""
    REACH = 1
    """The skill for reaching the object."""
    GRASP = 2
    """The skill is grasping the object."""
    DONE = 3
    """The skill has grasped the ojbect."""


class GraspXYZSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A grasp skill for grasping an object at an xyz and yaw position.

    Parameterized by [x,y,z, yaw] the x y z location to perform the grasp action and orientation

    """

    def __init__(
        self,
        reach_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        gripper_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        length: int,
    ) -> None:
        """Initialize the grasp skill.

        Args:
            reach_policy: The policy for reaching.
            gripper_policy: The policy for grasping.
            length: The number of steps to execute the skill for.

        """
        self._name = "grasp_xyz_skill"
        self._reach_policy = reach_policy
        self._gripper_policy = gripper_policy
        self._length = length
        self._status = None
        self._grasp_status = None
        self._params = None

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._reach_policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:
        """Initiate the grasp xyz skill.

        Args:
            obs: The low-level observation for the skill.
            params: The grasp parameters, (x, y, z, yaw) as shape (b, 4)

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._grasp_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._grasp_status[:] = GraspStatusCodes.REACH
        self._params = params
        self._n_steps = 0

        self._pos_threshold = 0.02
        self._quat_threshold = 0.08

        ee_pose_b = obs["tcp_pose_b"]
        roll, pitch, _ = euler_xyz_from_quat(ee_pose_b[:, 3:7])

        # Define the target poses for each stage of the grasp skill, indexed by GraspStatusCodes
        # (n_envs, num_pick_stages, 7)
        target_poses = spec.zeros(shape=(self.n_envs, 4, 7), dtype=float)
        # REACH[1]: Reach the target location (gripper open)
        target_poses[:, GraspStatusCodes.REACH, 0:3] = self._params[:, 0:3]
        target_poses[:, GraspStatusCodes.REACH, 3:7] = quat_from_euler_xyz(roll, pitch, params[:, 3])

        self._target_poses = target_poses

        # Start the skill by going to the ASCEND pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._grasp_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        np.set_printoptions(precision=3, suppress=True)
        print(
            f"[INFO][GRASP XYZ STATUS]: {self._grasp_status.cpu().numpy()[0]} | target pose: {self._current_target_poses.cpu().numpy()[0]} | obs tcp pose: {obs['tcp_pose_b'].cpu().numpy()[0]}"
        )

        ee_pose_b = obs["tcp_pose_b"]
        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._current_target_poses[:, 0:3], dim=1)
            < self._pos_threshold
        )

        reached_quat = (
            quat_error_magnitude(ee_pose_b[:, 3:7], self._current_target_poses[:, 3:7]) < self._quat_threshold
        )
        reached_pose = reached_pos & reached_quat

        if reached_pose.any():
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._grasp_status[valid_idx] += 1
            valid_idx = valid_idx & (self._grasp_status < GraspStatusCodes.DONE)
            print(
                f"[INFO][GRASP XYZ STATUS UPDATE]: {self._grasp_status.cpu().numpy()[0]} | reached_pose: {reached_pose}"
            )
            # Update the target pose based on the new grasp status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._grasp_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            self._grasp_status >= GraspStatusCodes.GRASP,
            torch.ones_like(reach_actions[:, -1]),  # Open gripper
            torch.zeros_like(reach_actions[:, -1]),  # Close gripper
        )

        self._n_steps += 1
        self._status[self._grasp_status == GraspStatusCodes.DONE] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions
