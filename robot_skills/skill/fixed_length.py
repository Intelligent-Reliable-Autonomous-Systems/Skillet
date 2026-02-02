

from typing import Generic

from jaxtyping import Int
from robot_skills.core.policy import BatchedPolicy
from robot_skills.core.skill import BatchedSkill, Skill, SkillStatus, SkillStatusCodes, TBSkillObs, TBAction, TBSkillParams
from robot_skills.core.spaces import ArrayLike


class FixedLengthSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A skill that executes for a fixed number of steps."""

    def __init__(self, name: str, policy: BatchedPolicy[TBSkillObs, TBAction, TBSkillParams], length: int) -> None:
        self._name = name
        self._policy = policy
        self._length = length
        self._status = None
        self._params = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:
        n_envs = self.obs_spec.n_envs_from(obs)
        self._status = self.policy.obs_spec.with_n_envs(n_envs).zeros(shape=(n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self.policy.reset(obs, params)
        self._params = params
        self._n_steps = 0

    def get_action(self, obs: TBSkillObs) -> TBAction:
        action = self.policy.get_action(obs, self._params)
        self._n_steps += 1
        if self._n_steps >= self._length:
            self._status[:] = SkillStatusCodes.SUCCESS
        return action