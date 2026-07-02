"""A squeeze skill for squeezing a deformable object."""

import time
from enum import IntEnum
from typing import Generic

import torch
from jaxtyping import Float, Int

from skillet.core.math import quat_from_yaw, quat_mul
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
from skillet.skill.specs import XYZ_Yaw_XYZ_Params, XYZ_Yaw_XYZ_Params_Spec


class SqueezeStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    IDLE = 0
    """The skill is idle."""
    SQUEEZE = 1
    """The skill is squeezing"""
    RELEASE = 2
    """The skill is releasing"""
    DONE = 3
    """The skill is done"""


class SqueezeSkill(BatchedSkill[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params], Generic[TBAction]):
    """A squeeze skill for squeezing a deformable object.

    Generic Args:
        TBAction: The type of the action for the skill.

    Parameterized by []
    """

    def __init__(
        self,
        reach_policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params],
        lift_height: float,
        gripper_close: float,
        timeout: float,
        length: int,
    ) -> None:
        """Initialize the squeeze skill.

        Generic Args:
            TBAction: The type of the action for the skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            lift_height: The height to lift the object to.
            gripper_close: How closed the gripper should be
            length: The number of steps to execute the skill for.

        """
        self._name = "squeeze_skill"
        self._reach_policy = reach_policy
        self._lift_height = lift_height
        self._gripper_close = gripper_close
        self._timeout = timeout
        self._length = length
        self._status = None
        self._squeeze_status = None
        self._params = None

        # 180 degree rotation about X axis + -90 degree yaw
        # self._default_quat = torch.as_tensor([[0.0, 0.7071, -0.7071, 0.0]])
        # self._default_quat = torch.as_tensor([[0.7071, 0.0, 0.0, 0.7071]])
        self._default_quat = torch.as_tensor([[0.0, 0.7071, 0.7071, 0.0]])

    @property
    def param_dim(self) -> int:
        return

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
        """Initiate the squeeze skill.

        Args:
            obs: The low-level observation for the skill.
            params: The squeeze parameters, []

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._squeeze_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._squeeze_status[:] = SqueezeStatusCodes.SQUEEZE
        self._params = params
        self._n_steps = 0
        self._default_quat = self._default_quat.to(self.obs_spec.device)
        goal_quat = quat_mul(quat_from_yaw(params[:, 3]), self._default_quat.repeat(self.n_envs, 1))

        self._pos_threshold = 0.005
        self._quat_threshold = 0.05
        self._vel_threshold = 0.001
        self._joint_threshold = 0.001

        ee_pose_b = obs["tcp_pose_b"]

        # Define the target poses for each stage of the wipe skill, indexed by SqueezeStatusCodes
        # (n_envs, num_wipe_stages, 7)
        target_poses = spec.zeros(shape=(self.n_envs, 4, 7), dtype=float)

        # HOVER[1]: Go over to the target x,y position, oriented downward (gripper open)
        target_poses[:, SqueezeStatusCodes.SQUEEZE, :7] = ee_pose_b  # (x,y) from params
        target_poses[:, SqueezeStatusCodes.RELEASE, :7] = ee_pose_b
        self._target_poses = target_poses

        # Start the skill by going to the ASCEND pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._squeeze_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        self._start_time = time.perf_counter()
        self._squeezed = torch.zeros(self.n_envs, device=target_poses.device)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        ee_pose_b = obs["tcp_pose_b"]

        elapsed_time = time.perf_counter() - self._start_time
        reached_pose = elapsed_time > self._timeout

        if reached_pose:
            self._start_time = time.perf_counter()
            self._squeezed = torch.ones(self.n_envs, ee_pose_b.device)
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._squeeze_status[valid_idx] += 1
            valid_idx = valid_idx & (self._squeeze_status < SqueezeStatusCodes.DONE)
            # print(
            #     f"[INFO][SQUEEZE STATUS UPDATE]: {SqueezeStatusCodes(self._squeeze_status.cpu().numpy()[0]).name} | reached_pose: {reached_pose.cpu().numpy()}"
            # )
            # Update the target pose based on the new squeeze status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._squeeze_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            self._place_status >= SqueezeStatusCodes.RELEASE,
            torch.ones_like(reach_actions[:, -1]) * self._gripper_close,  # Keep gripper in marginally closed position
            torch.ones_like(reach_actions[:, -1]),  # Close gripper
        )

        self._n_steps += 1
        self._status[self._squeeze_status == SqueezeStatusCodes.DONE] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...
