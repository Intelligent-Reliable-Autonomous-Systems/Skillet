"""A rotate skill for picking an object up at a location, rotating it, and placing it back down."""

from enum import IntEnum
from typing import Generic

import torch
from jaxtyping import Float, Int

from skillet.core.math import quat_error_magnitude, quat_from_yaw, quat_mul
from skillet.core.policy import BatchedPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike, SkillParamsSpec
from skillet.envs.specs import IKEE_Obs
from skillet.skill.specs import XYZ_YAW_Params, XYZ_YAW_Params_Spec


class RotateStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    IDLE = 0
    """The skill is idle."""
    ASCEND = 1
    """The skill is ascending to the lift height."""
    HOVER = 2
    """The skill is reaching the hovering position."""
    LOWER = 3
    """The skill is lowering to the object."""
    GRASP = 4
    """The skill is grasping the object."""
    LIFT = 5
    """The skill lifting the object."""
    ROTATE = 6
    """The skill is rotating the object"""
    LOWER2 = 7
    """The skill is lowering the object"""
    RELEASE = 8
    """The skill is releasing the object"""
    HOVER2 = 9
    """The skill is returning to hovering height"""
    DONE = 10
    """The skill has lifted the ojbect."""


class RotateYawSkill(BatchedSkill[IKEE_Obs, TBAction, XYZ_YAW_Params], Generic[TBAction]):
    """A rotate skill for picking an object up, rotating it, and placing it back down.

    Generic Args:
        TBAction: The type of the action for the skill.

    Parameterized by [x,y,z, yaw, yaw] the x y z location to perform the pick action and orientation and the
    amount to rotate the block
    """

    def __init__(
        self,
        reach_policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_YAW_Params],
        gripper_policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_YAW_Params] | None,
        lift_height: float,
        lift_delta: float,
        length: int,
    ) -> None:
        """Initialize the rotate skill.

        Generic Args:
            TBAction: The type of the action for the skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            gripper_policy: The policy for grasping.
            lift_height: The height to hover
            lift_delta: The height to pick the object up to when rotating
            length: The number of steps to execute the skill for.

        """
        self._name = "rotate_yaw_skill"
        self._reach_policy = reach_policy
        self._gripper_policy = gripper_policy
        self._lift_height = lift_height
        self._lift_delta = lift_delta
        self._length = length
        self._status = None
        self._rotate_status = None
        self._params = None

        # 180 degree rotation about X axis + -90 yaw
        self._default_quat = torch.as_tensor([[0.0, 0.7071, -0.7071, 0.0]])

    @property
    def param_dim(self) -> int:
        return 4

    @property
    def params_spec(self) -> SkillParamsSpec[XYZ_YAW_Params]:
        return XYZ_YAW_Params_Spec

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._reach_policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:
        """Initiate the rotate skill.

        Args:
            obs: The low-level observation for the skill.
            params: The rotate parameters, (x, y, z, yaw, yaw_rot) as shape (b, 5)

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._rotate_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._rotate_status[:] = RotateStatusCodes.ASCEND
        self._params = params
        self._n_steps = 0
        self._default_quat = self._default_quat.to(self.obs_spec.device)
        grasp_quat = quat_mul(quat_from_yaw(params[:, 3]), self._default_quat.repeat(self.n_envs, 1))
        rotate_quat = quat_mul(quat_from_yaw(params[:, 4]), grasp_quat)

        self._pos_threshold = 0.01
        self._quat_threshold = 0.1
        self._vel_threshold = 0.001
        self._joint_threshold = 0.001

        ee_pose_b = obs["tcp_pose_b"]

        # Define the target poses for each stage of the rotate skill, indexed by RotateStatusCodes
        # (n_envs, num_rotate_stages, 7)
        target_poses = spec.zeros(shape=(self.n_envs, 11, 7), dtype=float)
        # ASCEND[1]: Go up to lift height (gripper open)
        target_poses[:, RotateStatusCodes.ASCEND, :7] = ee_pose_b
        target_poses[:, RotateStatusCodes.ASCEND, 2] = self._lift_height

        # HOVER[2]: Go over to the target x,y position, oriented downward (gripper open)
        target_poses[:, RotateStatusCodes.HOVER, :2] = params[:, :2]  # (x,y) from params
        target_poses[:, RotateStatusCodes.HOVER, 2] = self._lift_height
        target_poses[:, RotateStatusCodes.HOVER, 3:7] = grasp_quat
        # LOWER[3]: Go down to the target z position (gripper open)
        target_poses[:, RotateStatusCodes.LOWER, :7] = target_poses[:, RotateStatusCodes.HOVER, :7]
        target_poses[:, RotateStatusCodes.LOWER, 2] = params[:, 2]
        # GRASP[4]: Close gripper
        target_poses[:, RotateStatusCodes.GRASP, :7] = target_poses[:, RotateStatusCodes.LOWER, :7]
        # LIFT[5]: Lift up to the target z position (gripper closed)
        target_poses[:, RotateStatusCodes.LIFT, :7] = target_poses[:, RotateStatusCodes.LOWER, :7]
        target_poses[:, RotateStatusCodes.LIFT, 2] = target_poses[:, RotateStatusCodes.LIFT, 2] + self._lift_delta
        # ROTATE[6]: Rotate the block to target yaw
        target_poses[:, RotateStatusCodes.ROTATE, 0:3] = target_poses[:, RotateStatusCodes.LIFT, 0:3]
        target_poses[:, RotateStatusCodes.ROTATE, 3:7] = rotate_quat
        # LOWER2 [7]: Lower the block back down
        target_poses[:, RotateStatusCodes.LOWER2, 0:3] = target_poses[:, RotateStatusCodes.LOWER, 0:3]
        target_poses[:, RotateStatusCodes.LOWER2, 3:7] = rotate_quat
        # RELEASE[8]: Release the block
        target_poses[:, RotateStatusCodes.RELEASE, :7] = target_poses[:, RotateStatusCodes.LOWER2, :7]
        # HOVER2 [9]: Return to hovering position
        target_poses[:, RotateStatusCodes.HOVER2, :7] = target_poses[:, RotateStatusCodes.HOVER, :7]

        self._target_poses = target_poses

        # Start the skill by going to the ASCEND pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._rotate_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        ee_pose_b = obs["tcp_pose_b"]

        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._current_target_poses[:, 0:3], dim=1)
            < self._pos_threshold
        )
        reached_height = self._rotate_status == RotateStatusCodes.ASCEND & (
            ee_pose_b[:, 2] >= self._current_target_poses[:, 2] - self._pos_threshold
        )
        reached_quat = (
            quat_error_magnitude(ee_pose_b[:, 3:7], self._current_target_poses[:, 3:7]) < self._quat_threshold
        )
        reached_pose = (reached_pos & reached_quat) | reached_height
        next_pose = reached_pose

        if next_pose.any():
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._rotate_status[valid_idx] += 1
            valid_idx = valid_idx & (self._rotate_status < RotateStatusCodes.DONE)
            print(
                f"[INFO][ROTATE YAW STATUS UPDATE]: {RotateStatusCodes(self._rotate_status.cpu().numpy()[0]).name} | reached_pose: {reached_pose}"
            )
            # Update the target pose based on the new rotate status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._rotate_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            (self._rotate_status >= RotateStatusCodes.GRASP) & (self._rotate_status <= RotateStatusCodes.RELEASE),
            torch.ones_like(reach_actions[:, -1]) * 0.8,  # Close gripper
            torch.zeros_like(reach_actions[:, -1]) + 0.2,  # Open gripper
        )

        self._n_steps += 1
        self._status[self._rotate_status == RotateStatusCodes.DONE] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...
