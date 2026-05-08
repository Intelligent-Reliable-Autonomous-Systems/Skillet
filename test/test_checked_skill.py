"""Tests for CheckedSkill ABC, SkillResult, and FailureReason"""

from __future__ import annotations

import pytest

from skillet.core.checked_skill import CheckedSkill, FailureReason, SkillResult
from skillet.core.skill import SingleSkill, Skill
from skillet.scene.base import Scene


# ---------------------------------------------------------------------------
# Minimal concrete CheckedSkill for structural tests
# ---------------------------------------------------------------------------

class _StubCheckedSkill(CheckedSkill):
    """Minimal concrete implementation that satisfies all abstract methods."""

    @property
    def policy(self):
        raise NotImplementedError

    @property
    def status(self):
        return 0

    def get_action(self, obs):
        raise NotImplementedError

    def preconditions(self, world: Scene) -> bool:
        return True

    def postconditions(self, world: Scene) -> bool:
        return True


# ---------------------------------------------------------------------------
# CheckedSkill inheritance
# ---------------------------------------------------------------------------

def test_checked_skill_is_single_skill() -> None:
    """CheckedSkill subclasses SingleSkill."""
    assert issubclass(CheckedSkill, SingleSkill)


def test_checked_skill_is_skill() -> None:
    """CheckedSkill is in the Skill hierarchy."""
    assert issubclass(CheckedSkill, Skill)


def test_concrete_subclass_instantiates() -> None:
    """A fully-implemented subclass can be instantiated."""
    skill = _StubCheckedSkill()
    assert isinstance(skill, CheckedSkill)
    assert isinstance(skill, SingleSkill)


def test_missing_preconditions_raises() -> None:
    """Omitting preconditions prevents instantiation."""

    class _Bad(CheckedSkill):
        @property
        def policy(self): raise NotImplementedError
        @property
        def status(self): return 0
        def get_action(self, obs): raise NotImplementedError
        def postconditions(self, world: Scene) -> bool: return True

    with pytest.raises(TypeError):
        _Bad()


def test_missing_postconditions_raises() -> None:
    """Omitting postconditions prevents instantiation."""

    class _Bad(CheckedSkill):
        @property
        def policy(self): raise NotImplementedError
        @property
        def status(self): return 0
        def get_action(self, obs): raise NotImplementedError
        def preconditions(self, world: Scene) -> bool: return True

    with pytest.raises(TypeError):
        _Bad()

# ---------------------------------------------------------------------------
# SkillResult
# ---------------------------------------------------------------------------

def test_skill_result_ok() -> None:
    result = SkillResult.ok()
    assert result.success is True
    assert result.failure_reason is None
    assert result.message is None


def test_skill_result_fail() -> None:
    result = SkillResult.fail(FailureReason.IK_FAILURE)
    assert result.success is False
    assert result.failure_reason is FailureReason.IK_FAILURE
    assert result.message is None


def test_skill_result_fail_with_message() -> None:
    result = SkillResult.fail(FailureReason.GRASP_FAILURE, "block slipped")
    assert result.message == "block slipped"