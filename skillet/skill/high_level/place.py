"""A place skill for placing an object down at a location and height after lifting to specified height."""

from enum import IntEnum
from typing import Generic

import numpy as np
import torch
from jaxtyping import Int

from skillet.core.math import quat_error_magnitude, quat_from_euler_xyz
from skillet.core.policy import BatchedPPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike


class PlaceStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    IDLE = 0
    """The skill is idle."""
    ASCEND = 1
    """The skill is ascending to the lift height."""
    HOVER = 2
    """The skill is reaching the hovering position."""
    LOWER = 3
    """The skill is lowering to the object."""
    RELEASE = 4
    """The skill is releasing the object."""
    LIFT = 5
    """The skill lifting the object."""
    DONE = 6
    """The skill has lifted the ojbect."""


class PlaceSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A place skill for placing an object down at a location and height after lifting to specified height.

    Parameterized by [x,y,z, yaw] the x y z location to perform the place action and orientation
    """

    def __init__(
        self,
        reach_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        gripper_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        lift_height: float,
        length: int,
    ) -> None:
        """Initialize the place skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            gripper_policy: The policy for grasping.
            lift_height: The height to lift the object to.
            length: The number of steps to execute the skill for.

        """
        self._name = "place_skill"
        self._reach_policy = reach_policy
        self._gripper_policy = gripper_policy
        self._lift_height = lift_height
        self._length = length
        self._status = None
        self._place_status = None
        self._params = None

    @property
    def param_dim(self) -> int:
        return 4

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
        """Initiate the place skill.

        Args:
            obs: The low-level observation for the skill.
            params: The place parameters, (x, y, z, yaw) as shape (b, 4)

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._place_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._place_status[:] = PlaceStatusCodes.ASCEND
        self._params = params
        self._n_steps = 0

        self._pos_threshold = 0.02
        self._quat_threshold = 0.08

        ee_pose_b = obs["tcp_pose_b"]

        # Define the target poses for each stage of the place skill, indexed by PlaceStatusCodes
        # (n_envs, num_pick_stages, 7)
        target_poses = spec.zeros(shape=(self.n_envs, 7, 7), dtype=float)
        # ASCEND[1]: Go up to lift height (gripper open)
        target_poses[:, PlaceStatusCodes.ASCEND, :7] = ee_pose_b
        target_poses[:, PlaceStatusCodes.ASCEND, 2] = self._lift_height

        # HOVER[2]: Go over to the target x,y position, oriented downward (gripper open)
        target_poses[:, PlaceStatusCodes.HOVER, :2] = params[:, :2]  # (x,y) from params
        target_poses[:, PlaceStatusCodes.HOVER, 2] = self._lift_height
        zero_vec = spec.zeros(shape=(self.n_envs,), dtype=float)
        pi_vec = torch.full_like(zero_vec, fill_value=torch.pi)
        target_poses[:, PlaceStatusCodes.HOVER, 3:7] = quat_from_euler_xyz(pi_vec, zero_vec, params[:, 3])
        # LOWER[3]: Go down to the target z position (gripper open)
        target_poses[:, PlaceStatusCodes.LOWER, :7] = target_poses[:, PlaceStatusCodes.HOVER, :7]
        target_poses[:, PlaceStatusCodes.LOWER, 2] = params[:, 2]
        # RELEASE[4]: Open gripper
        target_poses[:, PlaceStatusCodes.RELEASE, :7] = target_poses[:, PlaceStatusCodes.LOWER, :7]
        # LIFT[5]: Lift up to the target z position (gripper closed)
        target_poses[:, PlaceStatusCodes.LIFT, :7] = target_poses[:, PlaceStatusCodes.HOVER, :7]
        self._target_poses = target_poses

        # Start the skill by going to the ASCEND pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._place_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        np.set_printoptions(precision=3, suppress=True)
        print(
            f"[INFO][PLACE STATUS]: {self._place_status.cpu().numpy()[0]} | target pose: {self._current_target_poses.cpu().numpy()[0]} | obs tcp pose: {obs['tcp_pose_b'].cpu().numpy()[0]}"
        )

        # prev_place_status = self._place_status.clone()

        ee_pose_b = obs["tcp_pose_b"]
        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._current_target_poses[:, 0:3], dim=1)
            < self._pos_threshold
        )
        reached_height = self._place_status == PlaceStatusCodes.ASCEND & (
            ee_pose_b[:, 2] >= self._current_target_poses[:, 2]
        )
        reached_quat = (
            quat_error_magnitude(ee_pose_b[:, 3:7], self._current_target_poses[:, 3:7]) < self._quat_threshold
        )
        reached_pose = (reached_pos & reached_quat) | reached_height

        if reached_pose.any():
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._place_status[valid_idx] += 1
            valid_idx = valid_idx & (self._place_status < PlaceStatusCodes.DONE)
            print(f"[INFO][PLACE STATUS UPDATE]: {self._place_status.cpu().numpy()[0]} | reached_pose: {reached_pose}")
            # Update the target pose based on the new place status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._place_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            self._place_status >= PlaceStatusCodes.RELEASE,
            torch.zeros_like(reach_actions[:, -1]),  # Open gripper
            torch.ones_like(reach_actions[:, -1]),  # Close gripper
        )

        self._n_steps += 1
        self._status[self._place_status == PlaceStatusCodes.DONE] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions
