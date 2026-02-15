"""A push skill for pushing an from xyz to dx dy dz."""

from enum import IntEnum
from typing import Generic

import numpy as np
import torch
from jaxtyping import Int

from skillet.core.math import quat_error_magnitude
from skillet.core.policy import BatchedPPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike


class PushStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    IDLE = 0
    """The skill is idle."""
    REACH = 1
    """The skill is reaching the start location"""
    PUSH = 2
    """The skill is pushing the object."""
    DONE = 3
    """The skill has pushed the ojbect."""


class PushSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A push skill for pushing an from xyz to dx dy dz.

    Parameterized by [x,y,z, dx, dy, dz] the x y z location to start pushing from and dx dy dz to push along
    """

    def __init__(
        self,
        reach_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        gripper_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        length: int,
    ) -> None:
        """Initialize the push skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            gripper_policy: The policy for grasping.
            lift_height: The height to lift the object to.
            length: The number of steps to execute the skill for.

        """
        self._name = "push_skill"
        self._reach_policy = reach_policy
        self._gripper_policy = gripper_policy
        self._length = length
        self._status = None
        self._push_status = None
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
        """Initiate the push skill.

        Args:
            obs: The low-level observation for the skill.
            params: The push parameters, (x, y, z, dx, dy, dz) as shape (b, 6)

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._push_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._push_status[:] = PushStatusCodes.REACH
        # self._gripper_policy.reset(obs, params)
        self._params = params
        self._n_steps = 0

        self._pos_threshold = 0.02
        self._quat_threshold = 0.08

        ee_pose_b = obs["tcp_pose_b"]

        # Define the target poses for each stage of the push skill, indexed by PushStatusCodes
        # (n_envs, num_push_stages, 7)
        target_poses = spec.zeros(shape=(self.n_envs, 4, 7), dtype=float)
        # REACH[1]: Go up to lift height (gripper open)
        target_poses[:, PushStatusCodes.REACH, 3:7] = ee_pose_b[:, 3:7]
        target_poses[:, PushStatusCodes.REACH, 0:3] = self._params[:, 0:3]

        # PUSH[2]: Go over to the delta x,y z position
        target_poses[:, PushStatusCodes.PUSH, 0:3] = self._params[:, 0:3] + self._params[:, 3:6]
        target_poses[:, PushStatusCodes.PUSH, 3:7] = ee_pose_b[:, 3:7]

        self._target_poses = target_poses

        # Start the skill by going to the ASCEND pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._push_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        np.set_printoptions(precision=3, suppress=True)
        print(
            f"[INFO][PUSH STATUS]: {self._push_status.cpu().numpy()[0]} | target pose: {self._current_target_poses.cpu().numpy()[0]} | obs tcp pose: {obs['tcp_pose_b'].cpu().numpy()[0]}"
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
            self._push_status[valid_idx] += 1
            valid_idx = valid_idx & (self._push_status < PushStatusCodes.DONE)
            print(f"[INFO][PUSH STATUS UPDATE]: {self._push_status.cpu().numpy()[0]} | reached_pose: {reached_pose}")
            # Update the target pose based on the new push status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._push_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)

        self._n_steps += 1
        self._status[self._push_status == PushStatusCodes.DONE] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions
