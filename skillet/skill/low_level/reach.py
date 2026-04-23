"""A skill that moves the end effector to a pose."""

from typing import Generic

import numpy as np
import torch
from jaxtyping import Float, Int
from typing_extensions import override

from skillet.core.math import quat_error_magnitude, quat_from_euler_xyz
from skillet.core.policy import BatchedPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike
from skillet.envs.specs import IKEE_Obs
from skillet.skill.specs import XYZ_QUAT_Params


class ReachPoseSkill(BatchedSkill[IKEE_Obs, TBAction, XYZ_QUAT_Params], Generic[TBAction]):
    """A skill that moves the end effector to a specified pose.

    ParamSpace:
        - (B, 7): batched pose position xyz + orientation quat (wxyz)
    """

    def __init__(self, name: str, policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_QUAT_Params], length: int) -> None:
        """Initialize the fixed length skill.

        Args:
            name: The name of the skill.
            policy: The policy for the skill.
            length: The number of steps to execute the skill for.

        """
        self._name = name
        self._policy = policy
        self._length = length
        self._status = None
        self._params = None

    @property
    def param_dim(self) -> int:
        return 7

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPolicy[IKEE_Obs, TBAction, XYZ_QUAT_Params]:
        """The policy for the skill."""
        return self._policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    @override
    def initiate(self, obs: IKEE_Obs, params: XYZ_QUAT_Params) -> None:
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._params = params
        self._n_steps = 0

        self._pos_threshold = 0.02
        self._quat_threshold = 0.08

        target_poses = spec.zeros(shape=(self.n_envs, 7), dtype=float)
        target_poses[:, 0:7] = params[:, 0:7]

        self._target_poses = target_poses

        self.policy.reset(obs, self._target_poses)

    @override
    def get_action(self, obs: IKEE_Obs) -> TBAction:

        self._n_steps += 1

        tcp_pose_b = obs["tcp_pose_b"]
        reached_pos = (
            torch.linalg.vector_norm(tcp_pose_b[:, 0:3] - self._target_poses[:, 0:3], dim=1) < self._pos_threshold
        )
        reached_quat = quat_error_magnitude(tcp_pose_b[:, 3:7], self._target_poses[:, 3:7]) < self._quat_threshold
        reached_pose = reached_pos & reached_quat

        self._status = torch.where(
            reached_pose,
            SkillStatusCodes.SUCCESS,
            self._status,
        )
        action = self.policy.get_action(obs)

        if self._n_steps >= self._length:
            self._status[:] = SkillStatusCodes.FAILED

        return action

    def reward(self, obs: IKEE_Obs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...


class ReachXYZSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A skill that moves the end effector to a specified location."""

    def __init__(
        self, name: str, policy: BatchedPolicy[TBSkillObs, TBAction, TBSkillParams], length: int, clip: bool = False
    ) -> None:
        """Initialize the fixed length skill.

        Args:
            name: The name of the skill.
            policy: The policy for the skill.
            length: The number of steps to execute the skill for.
            clip: if to clip params to (-1, 1), for RL

        """
        self._name = name
        self._policy = policy
        self._length = length
        self._status = None
        self._params = None
        self._clip = clip

    @property
    def param_dim(self) -> int:
        return 3

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:  # noqa: D102
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._params = params
        self._n_steps = 0

        self._pos_threshold = 0.02
        ee_pose_b = obs["tcp_pose_b"]
        target_poses = spec.zeros(shape=(self.n_envs, 7), dtype=float)
        target_poses[:, 3:7] = ee_pose_b[:, 3:7]
        if self._clip:
            min_xyz = torch.as_tensor([0.0, -1.0, 0.02], device=ee_pose_b.device)
            max_xyz = torch.as_tensor([1.0, 1.0, 1.0], device=ee_pose_b.device)
            target_poses[:, 0:3] = min_xyz + ((params[:, 0:3].clip(-1, 1) + 1) / 2) * (max_xyz - min_xyz)
        else:
            target_poses[:, 0:3] = params[:, 0:3]
        self._target_poses = target_poses

        self.policy.reset(obs, self._target_poses)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102

        self._n_steps += 1

        ee_pose_b = obs["tcp_pose_b"]
        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._target_poses[:, 0:3], dim=1) < self._pos_threshold
        )

        self._status = torch.where(
            reached_pos,
            SkillStatusCodes.SUCCESS,
            self._status,
        )
        action = self.policy.get_action(obs)

        if self._n_steps >= self._length:
            self._status[:] = SkillStatusCodes.FAILED

        return action

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ee_pose_b = obs["tcp_pose_b"]
        dist = torch.clip(
            torch.norm(ee_pose_b[:, 0:3] - self._target_poses[:, 0:3], dim=1) - self._pos_threshold, 0, None
        )
        return 1.0 - torch.tanh(dist)


class ReachXYZRPYSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A skill that moves the end effector to a specified location and orientation."""

    def __init__(self, name: str, policy: BatchedPolicy[TBSkillObs, TBAction, TBSkillParams], length: int) -> None:
        """Initialize the fixed length skill.

        Args:
            name: The name of the skill.
            policy: The policy for the skill.
            length: The number of steps to execute the skill for.

        """
        self._name = name
        self._policy = policy
        self._length = length
        self._status = None
        self._params = None

    @property
    def param_dim(self) -> int:
        return 6

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:  # noqa: D102
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._params = params
        self._n_steps = 0

        self._pos_threshold = 0.02
        self._quat_threshold = 0.08

        target_poses = spec.zeros(shape=(self.n_envs, 7), dtype=float)
        target_poses[:, 3:7] = quat_from_euler_xyz(self._params[:, 3], self._params[:, 4], self._params[:, 5])
        target_poses[:, 0:3] = params[:, 0:3]

        self._target_poses = target_poses

        self.policy.reset(obs, params[:, :6])

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102

        self._n_steps += 1

        ee_pose_b = obs["tcp_pose_b"]
        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._target_poses[:, 0:3], dim=1) < self._pos_threshold
        )
        reached_quat = quat_error_magnitude(ee_pose_b[:, 3:7], self._target_poses[:, 3:7]) < self._quat_threshold
        reached_pose = reached_pos & reached_quat

        self._status = torch.where(
            reached_pose,
            SkillStatusCodes.SUCCESS,
            self._status,
        )
        action = self.policy.get_action(obs)

        if self._n_steps >= self._length:
            self._status[:] = SkillStatusCodes.FAILED

        return action

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...
