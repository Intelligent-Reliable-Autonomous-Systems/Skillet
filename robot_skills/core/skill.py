import abc
from dataclasses import replace
from enum import IntEnum, StrEnum
from typing import Any, Generic, NamedTuple, Optional, Sequence, TypeAlias, TypeVar

from robot_skills.core.policy import BatchedPolicy, Policy
from robot_skills.core.spaces import Action, ActionSpec, BatchedAction, BatchedObservation, BatchedSkillParams, CommonSpecs, ObservationSpec, SkillParamsSpec, Observation, SkillParams

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
    INITIATED = 1
    """The skill is initiated."""
    RUNNING = 2
    """The skill is running."""
    SUCCESS = 3
    """The skill is successful."""
    FAILED = 4
    """The skill is failed."""

class Skill(abc.ABC, Generic[TSkillObs, TSkillParams, TAction]):
    """A skill that represents a high-level action in the environment.
    
    Generic type parameters:
        TObs: The type of the observation from the environment.
        TParams: The type of the parameters of the skill.
    """

    @property
    @abc.abstractmethod
    def policy(self) -> Policy[TSkillObs, TSkillParams, TAction]:
        """The policy for the skill."""
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
        return CommonSpecs.ArrayEmpty

    @property
    @abc.abstractmethod
    def status(self) -> SkillStatus | Sequence[SkillStatus]:
        """The status of the skill."""
        raise NotImplementedError

    @abc.abstractmethod
    def can_initiate(self, obs: TSkillObs) -> bool | Sequence[bool]:
        """Check if the skill can be initiated with the given observation."""
        raise NotImplementedError

    def initiate(self, obs: TSkillObs, params: TSkillParams) -> None:
        """Initiate the skill with the given observation."""
        pass

    @abc.abstractmethod
    def get_action(self, obs: TSkillObs) -> TAction:
        """Get the next action for the skill based on the observation. Return the action if the skill is not terminated, otherwise return the result of the skill."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminated(self, obs: TSkillObs) -> bool | Sequence[bool]:
        """Check if the skill is terminated with the given observation."""
        raise NotImplementedError

class SingleSkill(Skill[TSkillObs, TSkillParams, TAction], abc.ABC, Generic[TSkillObs, TSkillParams, TAction]):
    """A single skill that takes a single observation and outputs a single action."""
    
    @property
    @abc.abstractmethod
    def status(self) -> SkillStatus:
        """The status of the skills."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def can_initiate(self, obs: TSkillObs) -> bool:
        """Check if the skill can be initiated with the given observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminated(self, obs: TSkillObs) -> bool:
        """Check if the skill is terminated with the given observation."""
        raise NotImplementedError

class BatchedSkill(Skill[TBSkillObs, TBSkillParams, TBAction], abc.ABC, Generic[TBSkillObs, TBSkillParams, TBAction]):
    """A batched skill that takes a batched observation and outputs a batched action."""
    
    @property
    @abc.abstractmethod
    def policy(self) -> BatchedPolicy[TBSkillObs, TBSkillParams, TBAction]:
        """The policy for the skill."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> Sequence[SkillStatus]:
        """The status of the skills."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def can_initiate(self, obs: TBSkillObs) -> Sequence[bool]:
        """Check if the skill can be initiated with the given observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminated(self, obs: TBSkillObs) -> Sequence[bool]:
        """Check if the skills are terminated with the given observation."""
        raise NotImplementedError

class CompositeSkill(BatchedSkill[TBSkillObs, TBSkillParams, TBAction], abc.ABC, Generic[TBSkillObs, TBSkillParams, TBAction]):

    def __init__(self, skills: Sequence[BatchedSkill[TBSkillObs, TBSkillParams, TBAction]], env_indices: Sequence[int]) -> None:
        """Initialize the composite skill."""
        if not hasattr(skills, "__len__") or len(skills) == 0:
            raise ValueError("The skills must be a non-empty sequence.")
        if not all(isinstance(skill, BatchedSkill) for skill in skills):
            raise ValueError("All skills must be batched skills.")
        self.skills = skills
        self.env_indices = env_indices
        # copy and mutate the specifications of the first skill to set n_envs
        self._params_spec = replace(self.skills[0].params_spec, n_envs=len(env_indices))
        self._observation_spec = replace(self.skills[0].obs_spec, n_envs=len(env_indices))
        self._action_spec = replace(self.skills[0].action_spec, n_envs=len(env_indices))
        if not self._params_spec.is_batched:
            raise ValueError("The parameters specification must be batched.")

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBSkillParams, TBAction]:
        """Returns the policy from the first skill in the composite skill."""
        return self.skills[0].policy

    @property
    def policies(self) -> Sequence[BatchedPolicy[TBSkillObs, TBSkillParams, TBAction]]:
        """Returns the policies from the skills in the composite skill."""
        return [skill.policy for skill in self.skills]

    @property
    def obs_spec(self) -> ObservationSpec[TBSkillObs]:
        """The observation specification is the same as the batched observation specification, but with a fixed n_envs."""
        return self._observation_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:
        """The action specification is the same as the batched action specification, but with a fixed n_envs."""
        return self._action_spec

    @property
    def params_spec(self) -> SkillParamsSpec[TBSkillParams]:
        """The parameters specification is the same as the batched skill parameters specification, but with a fixed n_envs."""
        return self._params_spec

    def can_initiate(self, obs: TBSkillObs) -> Sequence[bool]:
        """Check if the composite skill can be initiated with the given observation."""
        return [skill.can_initiate(self.observation_spec.index(obs, self.env_indices == idx)) for idx, skill in enumerate(self.skills)]

    def is_terminated(self, obs: TBSkillObs) -> Sequence[bool]:
        """Check if the composite skill is terminated with the given observation."""
        return [skill.is_terminated(self.observation_spec.index(obs, self.env_indices == idx)) for idx, skill in enumerate(self.skills)]

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:
        """Initiate the composite skill with the given observation and parameters."""
        for idx, skill in enumerate(self.skills):
            env_ids = self.env_indices == idx
            if not env_ids.any():
                continue
            skill.initiate(self.obs_spec.index(obs, env_ids), self.params_spec.index(params, env_ids))

    def get_action(self, obs: TBSkillObs) -> TBAction:
        """Get the next action for the composite skill based on the observation."""
        actions = self.action_spec.zeros()
        for idx, skill in enumerate(self.skills):
            env_ids = self.env_indices == idx
            if not env_ids.any():
                continue
            actions[env_ids] = skill.policy.get_action(self.obs_spec.index(obs, env_ids))
        return actions