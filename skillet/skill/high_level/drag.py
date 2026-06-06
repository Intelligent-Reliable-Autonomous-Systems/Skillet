"""A drag skill for dragging an object to a desired position."""

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
from skillet.skill.high_level.target_manager import TargetReachManager
from skillet.skill.specs import XYZ_Yaw_XYZ_Params, XYZ_Yaw_XYZ_Params_Spec


class DragStatusCodes(IntEnum):
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
    RAISE = 5
    """The skill is dragging the object."""
    DRAG = 6
    """The skill is raising the object."""
    LOWER2 = 7
    """The skill is raising the object."""
    RELEASE = 8
    """The skill is raising the object."""
    RAISE2 = 9
    """The skill is raising"""
    DONE = 10
    """The skill has lifted the ojbect."""


class DragSkill(BatchedSkill[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params], Generic[TBAction]):
    """A drag skill for dragging an object to a desired location.

    Generic Args:
        TBAction: The type of the action for the skill.

    Parameterized by [x,y,z, yaw, xyz] the x y z location to perform the drag action, orientation (yaw)
    of the gripper, and the new xyz position to drag to
    """

    def __init__(
        self,
        reach_policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params],
        lift_height: float,
        gripper_close: float,
        length: int,
        pos_threshold: float = 0.005,
        quat_threshold: float = 0.04,
        max_pos_threshold: float | None = None,
        stop_failure_steps: int = 120,
        stopped_velocity_threshold: float = 0.001,
    ) -> None:
        """Initialize the drag skill.

        Generic Args:
            TBAction: The type of the action for the skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            lift_height: The height to lift the object to.
            gripper_close: How closed the gripper should be
            length: The number of steps to execute the skill for.

        """
        self._name = "drag_skill"
        self._reach_policy = reach_policy
        self._lift_height = lift_height
        self._gripper_close = gripper_close
        self._length = length
        self._status = None
        self._drag_status = None
        self._params = None

        # 180 degree rotation about X axis + -90 degree yaw
        # self._default_quat = torch.as_tensor([[0.0, 0.7071, -0.7071, 0.0]])
        # self._default_quat = torch.as_tensor([[0.7071, 0.0, 0.0, 0.7071]])
        self._default_quat = torch.as_tensor([[0.0, 0.7071, 0.7071, 0.0]])
        self._pos_threshold = pos_threshold
        self._quat_threshold = quat_threshold
        self._target_manager = TargetReachManager(min_pose_threshold=pos_threshold, quat_threshold=quat_threshold,
            max_pose_threshold=max_pos_threshold, stopped_velocity_threshold=stopped_velocity_threshold)
    @property
    def param_dim(self) -> int:
        return 7

    @property
    def params_spec(self) -> SkillParamsSpec[XYZ_Yaw_XYZ_Params]:
        return XYZ_Yaw_XYZ_Params_Spec

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
        """Initiate the drag skill.

        Args:
            obs: The low-level observation for the skill.
            params: The drag parameters, (x, y, z, yaw, heading, dist) as shape (b, 6)

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._drag_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._drag_status[:] = DragStatusCodes.ASCEND
        self._params = params
        self._n_steps = 0
        self._default_quat = self._default_quat.to(self.obs_spec.device)
        goal_quat = quat_mul(quat_from_yaw(params[:, 3]), self._default_quat.repeat(self.n_envs, 1))

        self._vel_threshold = 0.001
        self._joint_threshold = 0.001
        self._target_manager.reset(obs["tcp_pose_b"][:, 0:3], obs["tcp_pose_b"][:, 3:7])
        ee_pose_b = obs["tcp_pose_b"]

        # Define the target poses for each stage of the drag skill, indexed by DragStatusCodes
        # (n_envs, num_drag_stages, 7)
        target_poses = spec.zeros(shape=(self.n_envs, 11, 7), dtype=float)
        # ASCEND[1]: Go up to lift height (gripper open)
        target_poses[:, DragStatusCodes.ASCEND, :7] = ee_pose_b
        target_poses[:, DragStatusCodes.ASCEND, 2] = self._lift_height

        # HOVER[2]: Go over to the target x,y position, oriented downward (gripper open)
        target_poses[:, DragStatusCodes.HOVER, :2] = params[:, :2]  # (x,y) from params
        target_poses[:, DragStatusCodes.HOVER, 2] = self._lift_height
        target_poses[:, DragStatusCodes.HOVER, 3:7] = goal_quat
        # LOWER[3]: Go down to the target z position (gripper open)
        target_poses[:, DragStatusCodes.LOWER, :7] = target_poses[:, DragStatusCodes.HOVER, :7]
        target_poses[:, DragStatusCodes.LOWER, 2] = params[:, 2]
        # GRASP[4]: Close gripper
        target_poses[:, DragStatusCodes.GRASP, :7] = target_poses[:, DragStatusCodes.LOWER, :7]
        # RAISE[5]: Close gripper
        target_poses[:, DragStatusCodes.RAISE, :7] = target_poses[:, DragStatusCodes.GRASP, :7]
        target_poses[:, DragStatusCodes.RAISE, 2] = target_poses[:, DragStatusCodes.GRASP, 2] + 0.02
        # DRAG[6]: Drag the object to the target location
        target_poses[:, DragStatusCodes.DRAG, 3:7] = target_poses[:, DragStatusCodes.RAISE, 3:7]
        target_poses[:, DragStatusCodes.DRAG, 0:3] = params[:, 4:7]
        target_poses[:, DragStatusCodes.DRAG, 2] = params[:, 6] + 0.02
        # LOWER[7]: Go down to the target z position (gripper open)
        target_poses[:, DragStatusCodes.LOWER2, :7] = target_poses[:, DragStatusCodes.DRAG, :7]
        target_poses[:, DragStatusCodes.LOWER2, 2] = params[:, 6]
        # RELEASE[8]
        target_poses[:, DragStatusCodes.RELEASE, :7] = target_poses[:, DragStatusCodes.LOWER2, :7]
        # RAISE[9]
        target_poses[:, DragStatusCodes.RAISE2, :7] = target_poses[:, DragStatusCodes.RELEASE, :7]
        target_poses[:, DragStatusCodes.RAISE2, 2] = target_poses[:, DragStatusCodes.HOVER, 2]
        self._target_poses = target_poses

        # Start the skill by going to the ASCEND pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._drag_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        ee_pose_b = obs["tcp_pose_b"]

        self._target_manager.add_pose(ee_pose_b[:, 0:3], ee_pose_b[:, 3:7])
        reached_pos = self._target_manager.reached_pos(self._current_target_poses[:, 0:3])
        reached_height = self._drag_status == DragStatusCodes.ASCEND & (
            ee_pose_b[:, 2] >= self._current_target_poses[:, 2] - self._pos_threshold
        )
        reached_quat = self._target_manager.reached_quat(self._current_target_poses[:, 3:7])
        reached_pose = (reached_pos & reached_quat) | reached_height
        next_pose = reached_pose

        if next_pose.any():
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._drag_status[valid_idx] += 1
            valid_idx = valid_idx & (self._drag_status < DragStatusCodes.DONE)
            print(
                f"[INFO][DRAG STATUS UPDATE]: {DragStatusCodes(self._drag_status.cpu().numpy()[0]).name} | reached_pose: {reached_pose.cpu().numpy()}"
            )
            # Update the target pose based on the new drag status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._drag_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            (self._drag_status >= DragStatusCodes.GRASP) & (self._drag_status < DragStatusCodes.RELEASE),
            torch.ones_like(reach_actions[:, -1]) * self._gripper_close,  # Close gripper
            torch.zeros_like(reach_actions[:, -1]),  # Open gripper
        )

        stuck = (self._drag_status != DragStatusCodes.RELEASE) & (self._drag_status != DragStatusCodes.GRASP) \
            & self._target_manager.is_stuck()
        if stuck.any():
            self._status[stuck] = SkillStatusCodes.FAILED
        self._n_steps += 1
        self._status[self._drag_status == DragStatusCodes.DONE] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...
