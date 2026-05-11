"""Tests for InspectSkill, InspectForDefectsSkill, and DiscardSkill."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from skillet.core.checked_skill import CheckedSkill, FailureReason, SkillResult
from skillet.perception.inspection.defect_classifier import DefectResult
from skillet.perception.inspection.mock_defect_classifier import MockDefectClassifier
from skillet.scene.base import Scene
from skillet.scene.objects.discard_location import DiscardLocation
from skillet.scene.objects.inspectable_cube import InspectableCube
from skillet.skill.high_level.inspect import InspectSkill
from skillet.skill.high_level.inspect_for_defects import InspectForDefectsSkill
from skillet.skill.object_level.discard import DiscardSkill

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CUBE_SIZE = 0.05  # 5 cm


def _cube(x: float, y: float, z: float, defective: bool | None = None, name: str = "block") -> InspectableCube:
    pose = torch.tensor([x, y, z, 1.0, 0.0, 0.0, 0.0])
    return InspectableCube(size=_CUBE_SIZE, defective=defective, init_pose=pose, name=name)


def _discard(cx: float = 0.6, cy: float = -0.3, cz: float = 0.0) -> DiscardLocation:
    pose = torch.tensor([cx, cy, cz, 1.0, 0.0, 0.0, 0.0])
    return DiscardLocation(width=0.3, depth=0.3, init_pose=pose, name="discard")


def _blank_image() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# InspectSkill
# ---------------------------------------------------------------------------

class TestInspectSkill:
    def _make_scene(self, block: InspectableCube) -> Scene:
        return Scene(objects=[block])

    def _skill(self, scene: Scene, block_id: int) -> InspectSkill:
        skill = InspectSkill(scene)
        skill.set_target(block_id)
        return skill

    def test_is_checked_skill(self) -> None:
        scene = self._make_scene(_cube(0.4, 0.0, 0.1))
        assert isinstance(self._skill(scene, 0), CheckedSkill)

    def test_preconditions_true_for_forward_block(self) -> None:
        block = _cube(x=0.4, y=0.0, z=0.1)
        scene = self._make_scene(block)
        assert self._skill(scene, block.object_id).preconditions(scene) is True

    def test_preconditions_false_for_behind_block(self) -> None:
        block = _cube(x=-0.4, y=0.0, z=0.1)
        scene = self._make_scene(block)
        assert self._skill(scene, block.object_id).preconditions(scene) is False

    def test_preconditions_false_without_target(self) -> None:
        block = _cube(x=0.4, y=0.0, z=0.1)
        scene = self._make_scene(block)
        skill = InspectSkill(scene)  # no set_target
        assert skill.preconditions(scene) is False

    def test_postconditions_true_when_pose_known(self) -> None:
        block = _cube(x=0.4, y=0.0, z=0.1)
        scene = self._make_scene(block)
        assert self._skill(scene, block.object_id).postconditions(scene) is True

    def test_execute_ok_for_reachable_block(self) -> None:
        block = _cube(x=0.4, y=0.0, z=0.1)
        scene = self._make_scene(block)
        result = self._skill(scene, block.object_id).execute(scene)
        assert result.success is True

    def test_execute_precondition_failure_for_behind_block(self) -> None:
        block = _cube(x=-0.4, y=0.0, z=0.1)
        scene = self._make_scene(block)
        result = self._skill(scene, block.object_id).execute(scene)
        assert result.success is False
        assert result.failure_reason == FailureReason.PRECONDITION_NOT_MET


# ---------------------------------------------------------------------------
# InspectForDefectsSkill
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_classifier() -> MockDefectClassifier:
    return MockDefectClassifier({
        "good_block": DefectResult(defective=False, confidence=0.95),
        "bad_block": DefectResult(defective=True, confidence=0.91),
    })


class TestInspectForDefectsSkill:
    def _make_scene(self, block: InspectableCube, discard: DiscardLocation | None = None) -> Scene:
        objects = [block]
        if discard is not None:
            objects.append(discard)
        return Scene(objects=objects)

    def _skill(
        self,
        scene: Scene,
        block_id: int,
        classifier: MockDefectClassifier,
    ) -> InspectForDefectsSkill:
        skill = InspectForDefectsSkill(scene, classifier)
        skill.set_target(block_id)
        return skill

    def test_is_checked_skill(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, name="good_block")
        scene = self._make_scene(block)
        assert isinstance(self._skill(scene, block.object_id, mock_classifier), CheckedSkill)

    def test_preconditions_true_for_uninspected_cube(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=None, name="good_block")
        scene = self._make_scene(block)
        assert self._skill(scene, block.object_id, mock_classifier).preconditions(scene) is True

    def test_preconditions_false_when_already_inspected(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=False, name="good_block")
        scene = self._make_scene(block)
        assert self._skill(scene, block.object_id, mock_classifier).preconditions(scene) is False

    def test_preconditions_false_without_target(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, name="good_block")
        scene = self._make_scene(block)
        skill = InspectForDefectsSkill(scene, mock_classifier)
        assert skill.preconditions(scene) is False

    def test_postconditions_false_before_execution(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=None, name="good_block")
        scene = self._make_scene(block)
        assert self._skill(scene, block.object_id, mock_classifier).postconditions(scene) is False

    def test_postconditions_true_after_verdict_written(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=False, name="good_block")
        scene = self._make_scene(block)
        assert self._skill(scene, block.object_id, mock_classifier).postconditions(scene) is True

    def test_execute_classifies_non_defective(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=None, name="good_block")
        scene = self._make_scene(block)
        skill = self._skill(scene, block.object_id, mock_classifier)
        result = skill.execute(scene, _blank_image())
        assert result.success is True
        assert block.defective is False

    def test_execute_classifies_defective(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=None, name="bad_block")
        scene = self._make_scene(block)
        skill = self._skill(scene, block.object_id, mock_classifier)
        result = skill.execute(scene, _blank_image())
        assert result.success is True
        assert block.defective is True

    def test_execute_precondition_failure_when_already_inspected(self, mock_classifier: MockDefectClassifier) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=True, name="bad_block")
        scene = self._make_scene(block)
        skill = self._skill(scene, block.object_id, mock_classifier)
        result = skill.execute(scene, _blank_image())
        assert result.success is False
        assert result.failure_reason == FailureReason.PRECONDITION_NOT_MET


# ---------------------------------------------------------------------------
# DiscardSkill
# ---------------------------------------------------------------------------

class TestDiscardSkill:
    def _make_scene(self, block: InspectableCube, discard: DiscardLocation) -> Scene:
        return Scene(objects=[block, discard])

    def _skill(self, scene: Scene, block_id: int) -> DiscardSkill:
        skill = DiscardSkill(scene)
        skill.set_target(block_id)
        return skill

    def test_is_checked_skill(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=True, name="bad_block")
        scene = self._make_scene(block, _discard())
        assert isinstance(self._skill(scene, block.object_id), CheckedSkill)

    def test_preconditions_true_for_defective_block(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=True, name="bad_block")
        scene = self._make_scene(block, _discard())
        assert self._skill(scene, block.object_id).preconditions(scene) is True

    def test_preconditions_false_for_non_defective_block(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=False, name="good_block")
        scene = self._make_scene(block, _discard())
        assert self._skill(scene, block.object_id).preconditions(scene) is False

    def test_preconditions_false_when_defective_is_none(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=None, name="block")
        scene = self._make_scene(block, _discard())
        assert self._skill(scene, block.object_id).preconditions(scene) is False

    def test_preconditions_false_without_discard_location(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=True, name="bad_block")
        scene = Scene(objects=[block])  # no DiscardLocation
        assert self._skill(scene, block.object_id).preconditions(scene) is False

    def test_preconditions_false_without_target(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=True, name="bad_block")
        scene = self._make_scene(block, _discard())
        skill = DiscardSkill(scene)  # no set_target
        assert skill.preconditions(scene) is False

    def test_postconditions_false_before_discard(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=True, name="bad_block")
        discard = _discard(cx=0.6, cy=-0.3)
        scene = self._make_scene(block, discard)
        assert self._skill(scene, block.object_id).postconditions(scene) is False

    def test_postconditions_true_when_block_in_discard_region(self) -> None:
        discard = _discard(cx=0.6, cy=-0.3)
        # Place the block at the discard centre
        block = _cube(0.6, -0.3, 0.01, defective=True, name="bad_block")
        scene = self._make_scene(block, discard)
        assert self._skill(scene, block.object_id).postconditions(scene) is True

    def test_execute_ok_when_preconditions_met(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=True, name="bad_block")
        scene = self._make_scene(block, _discard())
        result = self._skill(scene, block.object_id).execute(scene)
        assert result.success is True

    def test_execute_precondition_failure_for_non_defective(self) -> None:
        block = _cube(0.4, 0.0, 0.1, defective=False, name="good_block")
        scene = self._make_scene(block, _discard())
        result = self._skill(scene, block.object_id).execute(scene)
        assert result.success is False
        assert result.failure_reason == FailureReason.PRECONDITION_NOT_MET
