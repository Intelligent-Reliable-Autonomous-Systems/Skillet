import abc
from typing import Generic, TypeVar

from robot_skills.core.spaces import Action, ActionSpec, BatchedAction, BatchedObservation, BatchedSkillParams, ObservationSpec, Observation

### Policy type aliases
TPolicyObs = TypeVar("TPolicyObs", bound=Observation)
"""The type of the observation for the policy.

int | float | bool | Sequence[int | float | bool] | Mapping[str, int | float | bool | Sequence[int | float | bool]]"""
TPolicyParams = TypeVar("TPolicyParams")
"""The type of the parameters for the policy."""
TAction = TypeVar("TAction", bound=Action)
"""The type of the action for the policy: int | float | bool | Sequence[int | float | bool]"""

TBPolicyObs = TypeVar("TBPolicyObs", bound=BatchedObservation)
"""The type of the batched observation for the policy.

Sequence[int | float | bool] | Mapping[str, Sequence[int | float | bool]]"""
TBPolicyParams = TypeVar("TBPolicyParams", bound=BatchedSkillParams)
"""The type of the batched parameters for the policy.

Sequence[int | float | bool] | Mapping[str, Sequence[int | float | bool]]"""
TBAction = TypeVar("TBAction", bound=BatchedAction)
"""The type of the batched action for the policy: 

Sequence[int | float | bool]"""

### Policy classes
class Policy(abc.ABC, Generic[TPolicyObs, TPolicyParams, TAction]):
    """A policy that takes an observation and outputs an action."""

    @property
    @abc.abstractmethod
    def obs_spec(self) -> ObservationSpec[TPolicyObs]:
        """The specification of the observation space for the policy."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_spec(self) -> ActionSpec[TAction]:
        """The specification of the action space for the policy."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, obs: TPolicyObs, params: TPolicyParams) -> TAction:
        """Get the next low-level action for the robot based on the observation and parameters."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the policy. Useful if policy is stateful."""
        pass

class BatchedPolicy(Policy[TBPolicyObs, TBPolicyParams, TBAction], abc.ABC, Generic[TBPolicyObs, TBPolicyParams, TBAction]):
    """A batched policy that takes a batched observation and outputs a batched action."""
    pass