"""A skill that moves the end effector to a specified location."""

from typing import Generic

import torch
from jaxtyping import Int

from skillet.core.policy import BatchedPPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike


class ReachXYZSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A skill that moves the end effector to a specified location."""

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
        n_envs = self.obs_spec.n_envs_from(obs)
        self._status = self.policy.obs_spec.with_n_envs(n_envs).zeros(shape=(n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self.policy.reset(obs, params)
        self._params = params
        self._n_steps = 0

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        action = self.policy.get_action(obs, self._params)
        self._n_steps += 1
        self._status = torch.where(
            torch.linalg.vector_norm(obs["tcp_xyz_b"][:, 0:3] - self._params[:, 0:3], dim=1) < 0.02,
            SkillStatusCodes.SUCCESS,
            self._status,
        )
        if self._n_steps >= self._length:
            self._status[:] = SkillStatusCodes.FAILED
        return action
