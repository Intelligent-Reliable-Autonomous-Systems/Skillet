"""Module for defining and working with policy classes."""

import abc
from dataclasses import replace
from typing import Any, Generic, TypeVar

from skillet.core.spaces import (
    Action,
    ActionSpec,
    ArrayEmpty,
    BatchedAction,
    BatchedArrayEmpty,
    BatchedObservation,
    BatchedSkillParams,
    CommonSpecs,
    Observation,
    ObservationSpec,
    SkillParamsSpec,
)

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

Unparameterized = ArrayEmpty
"""The type of the unparameterized parameters for the policy."""
BUnparameterized = BatchedArrayEmpty
"""The type of the batched unparameterized parameters for the policy."""


### Policy classes
class Policy(abc.ABC, Generic[TPolicyObs, TAction, TPolicyParams]):
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

    @property
    @abc.abstractmethod
    def params_spec(self) -> SkillParamsSpec[TPolicyParams]:
        """The specification of the parameters space for the policy."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, obs: TPolicyObs, params: TPolicyParams) -> TAction:
        """Get the next low-level action for the robot based on the observation and parameters."""
        raise NotImplementedError

    def reset(self, obs: TPolicyObs, params: TPolicyParams) -> None:
        """Reset the policy. Useful if policy is stateful."""
        pass


class BatchedPolicy(
    Policy[TBPolicyObs, TBAction, TBPolicyParams],
    abc.ABC,
    Generic[TBPolicyObs, TBAction, TBPolicyParams],
):
    """A batched policy that takes a batched observation and outputs a batched action."""

    pass


class UPolicy(Policy[TPolicyObs, TAction, Unparameterized], abc.ABC, Generic[TPolicyObs, TAction]):
    """An unparameterized policy that has an empty parameters space."""

    @property
    def params_spec(self) -> SkillParamsSpec[Unparameterized]:
        """The specification of the parameters space for the policy."""
        return replace(CommonSpecs.ArrayEmpty, is_torch=self.action_spec.is_torch)

    def get_action(self, obs: TPolicyObs, params: Any = None) -> TAction:  # noqa: ANN401
        """Get the next low-level action for the robot based on the observation. The parameters are ignored."""
        raise NotImplementedError


class BatchedUPolicy(
    Policy[TBPolicyObs, TBAction, BUnparameterized],
    abc.ABC,
    Generic[TBPolicyObs, TBAction],
):
    """An unparameterized batched policy that takes a batched observation and outputs a batched action."""

    @property
    def params_spec(self) -> SkillParamsSpec[BUnparameterized]:
        """The specification of the parameters space for the policy."""
        return replace(CommonSpecs.BatchedArrayEmpty, is_torch=self.action_spec.is_torch)
