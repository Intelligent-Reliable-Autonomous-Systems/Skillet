"""A skill that opens or closes the gripper."""

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


class GripperSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A skill that opens or closes the gripper."""

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
        gripper_lim = obs["gripper_lim"]
        gripper_pos = obs["joint_pos"][:, -1]
        goal_gripper_pos = (self._params[:, 0] - gripper_lim[:, 0]) / (gripper_lim[:, 1] - gripper_lim[:, 0])
        action = torch.cat((obs["joint_pos"][:, :-1], goal_gripper_pos), dim=1)

        self._n_steps += 1

        self._status = torch.where(
            torch.linalg.vector_norm(gripper_pos - goal_gripper_pos, dim=1) < 0.02,
            SkillStatusCodes.SUCCESS,
            self._status,
        )
        if self._n_steps >= self._length:
            self._status[:] = SkillStatusCodes.FAILED
        return action
