"""Tests for SkillEventLogger."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from skillet.core.checked_skill import FailureReason, SkillResult
from skillet.logging.event_logger import SkillEventLogger
from skillet.perception.inspection.defect_classifier import DefectResult
from skillet.scene.base import Scene
from skillet.scene.objects.inspectable_cube import InspectableCube
from skillet.scene.objects.discard_location import DiscardLocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def _make_scene() -> Scene:
    cube = InspectableCube(
        size=0.05,
        defective=False,
        init_pose=torch.tensor([0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
        name="block_0",
    )
    discard = DiscardLocation(
        width=0.3, depth=0.3,
        init_pose=torch.tensor([0.6, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0]),
        name="discard_0",
    )
    return Scene(objects=[cube, discard])


# ---------------------------------------------------------------------------
# Construction and file creation
# ---------------------------------------------------------------------------

def test_logger_creates_file(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="test-run"):
        pass
    assert log_file.exists()


def test_run_id_propagates_to_all_events(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="my-run-id") as logger:
        logger.log_skill_start("InspectSkill")
        logger.log_planner_decision("inspect", ["block_0"])
    events = _read_events(log_file)
    assert all(e["run_id"] == "my-run-id" for e in events)


def test_all_events_have_ts_field(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_skill_start("InspectSkill")
    events = _read_events(log_file)
    assert all("ts" in e for e in events)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

def test_skill_start_event(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_skill_start("InspectForDefectsSkill", params={"block_id": 0})
    events = _read_events(log_file)
    assert len(events) == 1
    e = events[0]
    assert e["event"] == "skill_start"
    assert e["skill"] == "InspectForDefectsSkill"
    assert e["params"] == {"block_id": 0}


def test_skill_end_success(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_skill_end("InspectForDefectsSkill", SkillResult.ok())
    events = _read_events(log_file)
    e = events[0]
    assert e["event"] == "skill_end"
    assert e["success"] is True
    assert e["failure_reason"] is None


def test_skill_end_failure(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        result = SkillResult.fail(FailureReason.IK_FAILURE, "unreachable viewpoint")
        logger.log_skill_end("InspectSkill", result)
    events = _read_events(log_file)
    e = events[0]
    assert e["success"] is False
    assert e["failure_reason"] == "IK_FAILURE"
    assert e["message"] == "unreachable viewpoint"


def test_planner_decision_event(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_planner_decision("inspect-for-defects", ["block_0", "table"])
    events = _read_events(log_file)
    e = events[0]
    assert e["event"] == "planner_decision"
    assert e["action"] == "inspect-for-defects"
    assert e["parameters"] == ["block_0", "table"]


def test_classifier_verdict_event(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_classifier_verdict("block_0", defective=True, confidence=0.91)
    events = _read_events(log_file)
    e = events[0]
    assert e["event"] == "classifier_verdict"
    assert e["object_id"] == "block_0"
    assert e["defective"] is True
    assert abs(e["confidence"] - 0.91) < 1e-6


def test_world_model_snapshot_event(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    scene = _make_scene()
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_world_model_snapshot(scene)
    events = _read_events(log_file)
    e = events[0]
    assert e["event"] == "world_model_snapshot"
    names = {obj["name"] for obj in e["objects"]}
    assert "block_0" in names
    assert "discard_0" in names


def test_world_model_snapshot_includes_position(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    scene = _make_scene()
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_world_model_snapshot(scene)
    events = _read_events(log_file)
    cube_entry = next(o for o in events[0]["objects"] if o["name"] == "block_0")
    assert "position" in cube_entry
    assert len(cube_entry["position"]) == 3


# ---------------------------------------------------------------------------
# Order and count
# ---------------------------------------------------------------------------

def test_events_written_in_order(tmp_path: Path) -> None:
    """A fake skill sequence produces events in the expected order."""
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_planner_decision("inspect-for-defects", ["block_0"])
        logger.log_skill_start("InspectForDefectsSkill", params=0)
        logger.log_classifier_verdict("block_0", defective=False, confidence=0.95)
        logger.log_skill_end("InspectForDefectsSkill", SkillResult.ok())
        logger.log_planner_decision("discard", ["block_0"])
        logger.log_skill_start("DiscardSkill", params=0)
        logger.log_skill_end("DiscardSkill", SkillResult.ok())
    events = _read_events(log_file)
    assert len(events) == 7
    event_types = [e["event"] for e in events]
    assert event_types == [
        "planner_decision",
        "skill_start",
        "classifier_verdict",
        "skill_end",
        "planner_decision",
        "skill_start",
        "skill_end",
    ]


def test_each_line_is_valid_json(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_skill_start("InspectSkill")
        logger.log_skill_end("InspectSkill", SkillResult.ok())
    for line in log_file.read_text().strip().splitlines():
        json.loads(line)  # raises if invalid


def test_appends_across_reopen(tmp_path: Path) -> None:
    """Two logger instances writing to the same file accumulate all events."""
    log_file = tmp_path / "events.jsonl"
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_skill_start("InspectSkill")
    with SkillEventLogger(log_file, run_id="r") as logger:
        logger.log_skill_end("InspectSkill", SkillResult.ok())
    events = _read_events(log_file)
    assert len(events) == 2
