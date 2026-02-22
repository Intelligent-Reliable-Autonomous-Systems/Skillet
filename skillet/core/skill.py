import abc
from collections.abc import Sequence
from enum import IntEnum
from typing import Generic, TypeAlias, TypeVar

import numpy as np
from jaxtyping import Bool, Float, Int

from skillet.core.policy import BatchedPolicy, Policy
from skillet.core.spaces import (
    Action,
    ActionSpec,
    ArrayLike,
    BatchedAction,
    BatchedObservation,
    BatchedSkillParams,
    Observation,
    ObservationSpec,
    SkillParams,
    SkillParamsSpec,
)

TSkillObs = TypeVar("TSkillObs", bound=Observation)
"""The type of the observation for the skill.

int | float | bool | Sequence[int | float | bool] | Mapping[str, int | float | bool | Sequence[int | float | bool]]"""
TSkillParams = TypeVar("TSkillParams", bound=SkillParams)
"""The type of the parameters for the skill.

int | float | bool | Sequence[int | float | bool] | Mapping[str, int | float | bool | Sequence[int | float | bool]]"""
TAction = TypeVar("TAction", bound=Action)
"""The type of the action for the skill.

int | float | bool | Sequence[int | float | bool]"""
TBSkillObs = TypeVar("TBSkillObs", bound=BatchedObservation)
"""The type of the batched observation for the skill.

Sequence[int | float | bool] | Mapping[str, Sequence[int | float | bool]]"""
TBSkillParams = TypeVar("TBSkillParams", bound=BatchedSkillParams)
"""The type of the batched parameters for the skill.

Sequence[int | float | bool] | Mapping[str, Sequence[int | float | bool]]"""
TBAction = TypeVar("TBAction", bound=BatchedAction)
"""The type of the batched action for the skill.

Sequence[int | float | bool]"""
TBPolicy = TypeVar("TBPolicy", bound=BatchedPolicy)
"""The type of the batched policy for the skill."""

TOneOrManyBool = TypeVar("TOneOrManyBool", bound=bool | Sequence[bool])

SkillStatus: TypeAlias = int
"""The status code for a skill execution.

0: The skill is not initiated.
1: The skill is initiated.
2: The skill is running.
3: The skill is successful.
4: The skill is failed.
"""


class SkillStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    UNINITIATED = 0
    """The skill is not initiated."""
    RUNNING = 1
    """The skill is running."""
    SUCCESS = 2
    """The skill is successful."""
    FAILED = 3
    """The skill is failed."""


class Skill(abc.ABC, Generic[TSkillObs, TAction, TSkillParams]):
    """A skill that represents a high-level action in the environment.

    Generic type parameters:
        TObs: The type of the observation from the environment.
        TParams: The type of the parameters of the skill.
    """

    @property
    @abc.abstractmethod
    def param_dim(self) -> int:
        """The number of parameters expected by the skill."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """The name of the skill."""
        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def policy(self) -> Policy[TSkillObs, TAction, TSkillParams]:
        """The policy for the skill."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> SkillStatus | Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skill."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, obs: TSkillObs) -> TAction:
        """Get the next action for the skill based on the observation. Return the action if the skill is not terminated, otherwise return the result of the skill."""
        raise NotImplementedError

    @property
    def obs_spec(self) -> ObservationSpec[TSkillObs]:
        """The specification of the observation space for the skill."""
        return self.policy.obs_spec

    @property
    def action_spec(self) -> ActionSpec[TAction]:
        """The specification of the action space for the skill."""
        return self.policy.action_spec

    @property
    def params_spec(self) -> SkillParamsSpec[TSkillParams]:
        """The specification of the parameters space for the skill."""
        return self.policy.params_spec

    def initiate(self, obs: TSkillObs, params: TSkillParams) -> None:
        """Initiate the skill with the given observation."""
        pass

    def can_initiate(self, obs: TSkillObs) -> bool | Bool[ArrayLike, "b"]:  # noqa: F821
        """Check if the skill can be initiated with the given observation."""
        return self.status != SkillStatusCodes.RUNNING

    def is_terminated(self, obs: TSkillObs) -> bool | Bool[ArrayLike, "b"]:  # noqa: F821
        """Check if the skill is terminated with the given observation."""
        return (self.status == SkillStatusCodes.SUCCESS) | (self.status == SkillStatusCodes.FAILED)

    def is_success(self, obs: TSkillObs) -> bool | Bool[ArrayLike, "b"]:  # noqa: F821
        """Check if the skill is successful."""
        return self.status == SkillStatusCodes.SUCCESS

    def is_fail(self, obs: TSkillObs) -> bool | Bool[ArrayLike, "b"]:  # noqa: F821
        """Check if the skill failed."""
        return self.status == SkillStatusCodes.FAILED

    @abc.abstractmethod
    def reward(self, obs: TSkillObs) -> float | Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the dense reward of the skill for reward shaping."""
        raise NotImplementedError


class SingleSkill(
    Skill[TSkillObs, TAction, TSkillParams],
    abc.ABC,
    Generic[TSkillObs, TAction, TSkillParams],
):
    """A single skill that takes a single observation and outputs a single action."""

    @property
    @abc.abstractmethod
    def status(self) -> SkillStatus:
        """The status of the skills."""
        raise NotImplementedError

    def can_initiate(self, obs: TSkillObs) -> bool:
        return super().can_initiate(obs)

    def is_terminated(self, obs: TSkillObs) -> bool:
        return super().is_terminated(obs)


class BatchedSkill(
    Skill[TBSkillObs, TBAction, TBSkillParams],
    abc.ABC,
    Generic[TBSkillObs, TBAction, TBSkillParams],
):
    """A batched skill that takes a batched observation and outputs a batched action."""

    @property
    @abc.abstractmethod
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills. Must call initiate() before using this property."""
        raise NotImplementedError

    def can_initiate(self, obs: TBSkillObs) -> Bool[ArrayLike, "b"]:  # noqa: F821
        return super().can_initiate(obs)

    def is_terminated(self, obs: TBSkillObs) -> Bool[ArrayLike, "b"]:  # noqa: F821
        return super().is_terminated(obs)


class CompositeSkill(
    BatchedSkill[TBSkillObs, TBAction, TBSkillParams],
    abc.ABC,
    Generic[TBSkillObs, TBAction, TBSkillParams],
):
    """Group of skills as a high level skill controller."""

    def __init__(
        self,
        skills: Sequence[BatchedSkill[TBSkillObs, TBAction, TBSkillParams]],
        env_indices: Int[ArrayLike, "b"] | None = None,  # noqa: F821
    ) -> None:
        """Initialize the composite skill."""
        if not hasattr(skills, "__len__") or len(skills) == 0:
            raise ValueError("The skills must be a non-empty sequence.")
        if not all(isinstance(skill, BatchedSkill) for skill in skills):
            raise ValueError("All skills must be batched skills.")

        for skill in skills:
            if skill.obs_spec.n_envs != -1:
                raise ValueError(
                    f"The observation specification of the skills must support variable batch sizes (n_envs = -1): {skill.obs_spec}"
                )
            if skill.action_spec.n_envs != -1:
                raise ValueError(
                    f"The action specification of the skills must support variable batch sizes (n_envs = -1): {skill.action_spec}"
                )
            if skill.params_spec.n_envs != -1:
                raise ValueError(
                    f"The parameters specification of the skills must support variable batch sizes (n_envs = -1): {skill.params_spec}"
                )
        self.skills = skills
        self.env_indices = env_indices
        # copy and mutate the specifications of the first skill to set n_envs
        # self._params_spec = replace(self.skills[0].params_spec, n_envs=len(env_indices))
        # self._observation_spec = replace(self.skills[0].obs_spec, n_envs=len(env_indices))
        # self._action_spec = replace(self.skills[0].action_spec, n_envs=len(env_indices))
        self._status: Int[ArrayLike, b] | None = None  # noqa: F821

    @property
    def param_dim(self) -> int:
        """Return the maximum number of allowable parameters."""
        return int(np.sum(s.param_dim for s in self.skills))

    @property
    def name(self) -> str:
        """The name of the composite skill."""
        return f"CompositeSkill[{', '.join([skill.name for skill in self.skills])}]"

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """Returns the policy from the first skill in the composite skill."""
        return self.skills[0].policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the composite skill."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        for idx, skill in enumerate(self.skills):
            env_ids = self.env_indices == idx
            if not env_ids.any():
                continue
            self._status[env_ids] = skill.status
        return self._status

    @property
    def policies(self) -> Sequence[BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]]:
        """Returns the policies from the skills in the composite skill."""
        return [skill.policy for skill in self.skills]

    @property
    def obs_spec(self) -> ObservationSpec[TBSkillObs]:
        """The observation specification is the same as the batched observation specification, but with a fixed n_envs."""
        # TODO really need to handle multiple obs specifications
        return self.skills[0].obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:
        """The action specification is the same as the batched action specification, but with a fixed n_envs."""
        return self.skills[0].action_spec

    @property
    def params_spec(self) -> SkillParamsSpec[TBSkillParams]:
        """The parameters specification is the same as the batched skill parameters specification, but with a fixed n_envs."""
        return self.skills[0].params_spec

    def can_initiate(self, obs: TBSkillObs) -> Bool[ArrayLike, "b"]:  # noqa: F821
        """Check if the composite skill can be initiated with the given observation."""
        can_initiate = self.action_spec.zeros(shape=(-1,), dtype=bool)  # initialize (B,) array
        for idx, skill in enumerate(self.skills):
            env_ids = self.env_indices == idx
            if not env_ids.any():
                continue
            can_initiate[env_ids] = skill.can_initiate(self.obs_spec.index(obs, env_ids))
        return can_initiate

    def is_terminated(self, obs: TBSkillObs) -> Bool[ArrayLike, "b"]:  # noqa: F821
        """Check if the composite skill is terminated with the given observation."""
        n_envs = self.obs_spec.n_envs_from(obs)
        terminated = self.action_spec.zeros(shape=(n_envs,), dtype=bool)  # initialize (B,) array
        for idx, skill in enumerate(self.skills):
            env_ids = self.env_indices == idx
            if not env_ids.any():
                continue
            obs_idx = self.obs_spec.index(obs, env_ids)
            term_idx = skill.is_terminated(obs_idx)
            terminated[env_ids] = term_idx
        return terminated

    def initiate(
        self,
        obs: TBSkillObs,
        params: TBSkillParams,
        env_ids: Int[ArrayLike, "b"] | None = None,  # noqa: F821
    ) -> None:
        """Initiate the composite skill with the given observation and parameters.

        Optionally select the skills for each environment.
        """
        if env_ids is not None:
            self.env_indices = env_ids
        n_envs = self.obs_spec.n_envs_from(obs)
        self._status = self.action_spec.with_n_envs(n_envs).zeros(shape=(n_envs,), dtype=int)  # initialize (B,) array

        for idx, skill in enumerate(self.skills):
            env_ids = self.env_indices == idx
            if not env_ids.any():
                continue
            skill.initiate(
                self.obs_spec.index(obs, env_ids),
                self.params_spec.index(params, env_ids),
            )

    def get_action(self, obs: TBSkillObs) -> TBAction:
        """Get the next action for the composite skill based on the observation."""
        n_envs = self.obs_spec.n_envs_from(obs)
        actions = self.action_spec.with_n_envs(n_envs).zeros()
        for idx, skill in enumerate(self.skills):
            env_ids = self.env_indices == idx
            if not env_ids.any():
                continue
            actions[env_ids] = skill.get_action(self.obs_spec.index(obs, env_ids))
        return actions
