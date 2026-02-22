"""A skill that moves the end effector to a pose."""

from typing import Generic

import numpy as np
import torch
from jaxtyping import Float, Int

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


class ReachPoseSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A skill that moves the end effector to a specified pose."""

    def __init__(self, name: str, policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams], length: int) -> None:
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
    def policy(self) -> BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams]:
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
        target_poses[:, 0:7] = params[:, 0:7]

        self._target_poses = target_poses

        self.policy.reset(obs, self._target_poses)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        np.set_printoptions(precision=3, suppress=True)
        if False:
            print(
                f"[INFO][REACH POSE]: {self._status.cpu().numpy()[0]} | target pose: {self._target_poses.cpu().numpy()[0]} | obs ee pose: {obs['ee_pose_b'].cpu().numpy()[0]}"
            )

        self._n_steps += 1

        ee_pose_b = obs["ee_pose_b"]
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
        pass
