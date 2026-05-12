"""CheckedSkill ABC for checking pre/post conditions on scene"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic

from skillet.core.skill import SingleSkill, TAction, TSkillObs, TSkillParams
from skillet.scene.base import Scene


class FailureReason(Enum):
    """Typed failure reasons for CheckedSkill execution."""

    PRECONDITION_NOT_MET = auto()
    POSTCONDITION_NOT_MET = auto()
    IK_FAILURE = auto()
    GRASP_FAILURE = auto()
    PLACE_FAILURE = auto()
    INSPECTION_FAILURE = auto()
    COLLISION = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class SkillResult:
    """Structured result returned by a CheckedSkill after execution"""

    success: bool
    failure_reason: FailureReason | None = None
    message: str | None = None

    @classmethod
    def ok(cls) -> SkillResult:
        """Return a successful result."""
        return cls(success=True)

    @classmethod
    def fail(cls, reason: FailureReason, message: str | None = None) -> SkillResult:
        """Return a failed result with a typed reason."""
        return cls(success=False, failure_reason=reason, message=message)


class CheckedSkill(
    SingleSkill[TSkillObs, TAction, TSkillParams],
    ABC,
    Generic[TSkillObs, TAction, TSkillParams],
):
    """A SingleSkill that declares world-model preconditions and postconditions.

    Subclass this instead of SingleSkill for any skill that must verify scene
    state before and after execution.
    """

    @abstractmethod
    def preconditions(self, world: Scene) -> bool:
        """Return True iff the world state satisfies this skill's preconditions."""
        raise NotImplementedError

    @abstractmethod
    def postconditions(self, world: Scene) -> bool:
        """Return True iff the world state satisfies this skill's postconditions."""
        raise NotImplementedError
