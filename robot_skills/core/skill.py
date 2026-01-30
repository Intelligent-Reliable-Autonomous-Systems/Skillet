import abc
from enum import StrEnum
from typing import Generic, TypeVar

from skills.core.policy import Policy
from skills.core.spaces import CommonSpecs, ObservationSpec, SkillParamsSpec, Observation, SkillParams
from skills.core.env import Action

TSkillObs = TypeVar("TSkillObs", bound=Observation)
"""The type of the observation for the skill."""
TSkillParams = TypeVar("TSkillParams", bound=SkillParams)
"""The type of the parameters for the skill."""

class SkillStatus(StrEnum):
    """The status of a skill."""
    UNINITIATED = "uninitiated"
    INITIATED = "initiated"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

class Skill(abc.ABC, Generic[TSkillObs, TSkillParams]):
    """A skill that represents a high-level action in the environment.
    
    Generic type parameters:
        TObs: The type of the observation from the environment.
        TParams: The type of the parameters of the skill.
    """

    @property
    @abc.abstractmethod
    def policy(self) -> Policy:
        """The policy for the skill."""
        raise NotImplementedError

    @property
    def observation_spec(self) -> ObservationSpec[TSkillObs]:
        """The specification of the observation space for the skill."""
        return self.policy.observation_spec

    @property
    def params_spec(self) -> SkillParamsSpec[TSkillParams]:
        """The specification of the parameters space for the skill."""
        return CommonSpecs.ArrayEmpty

    @property
    @abc.abstractmethod
    def status(self) -> SkillStatus:
        """The status of the skill."""
        raise NotImplementedError

    @abc.abstractmethod
    def can_initiate(self, obs: TSkillObs) -> bool:
        """Check if the skill can be initiated with the given observation."""
        raise NotImplementedError

    def initiate(self, obs: TSkillObs, params: TSkillParams) -> None:
        """Initiate the skill with the given observation."""
        pass

    @abc.abstractmethod
    def get_action(self, obs: TSkillObs) -> Action | SkillStatus:
        """Get the next action for the skill based on the observation. Return the action if the skill is not terminated, otherwise return the result of the skill."""
        raise NotImplementedError
