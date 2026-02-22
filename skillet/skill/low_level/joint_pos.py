"""A skill that handles atomic joint positions."""

from typing import Generic

import numpy as np
import torch
from jaxtyping import Float, Int

from skillet.core.policy import BatchedPPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike


class JointPosSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A skill that handles atomic joint positions."""

    def __init__(self, name: str, policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams], length: int) -> None:
        """Initialize the fixed length skill.

        Args:
            name: The name of the skill.
            policy: The policy for the skill (None).
            length: The number of steps to execute the skill for.

        """
        self._name = name
        self._policy = policy
        self._length = length
        self._status = None
        self._params = None

    @property
    def param_dim(self) -> int:
        return 8  # TODO: address this based on env

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

        self._pos_threshold = 0.01
        joint_lims = obs["joint_lims"]
        self.goal_joint_pos = params[:, : joint_lims.shape[2]].clamp(
            min=joint_lims[:, 0],
            max=joint_lims[:, 1],
        )

        self.policy.reset(obs, self.goal_joint_pos)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        np.set_printoptions(precision=3, suppress=True)
        print(
            f"[INFO][JOINT POS]: {self._status.cpu().numpy()[0]} | target pos: {self.goal_joint_pos.cpu().numpy()[0]} | current pos: {obs['joint_pos'].cpu().numpy()[0]}"
        )

        self._n_steps += 1

        self._status = torch.where(
            torch.linalg.vector_norm(obs["joint_pos"] - self.goal_joint_pos, dim=1) < self._pos_threshold,
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
