"""Tests for the inspection PDDL domain and problem factory (Step 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from unified_planning.io import PDDLReader
from unified_planning.engines import PlanGenerationResultStatus as PGStatus
from unified_planning.shortcuts import OneshotPlanner

from skillet.planning.inspection import make_inspection_problem
from skillet.scene.objects import DiscardLocation, InspectableCube, Platform
from skillet.scene.scene_objs import Table

DOMAIN_FILE = Path(__file__).parent.parent / "skillet" / "planning" / "inspection" / "inspection.domain.pddl"

TABLE_HEIGHT = 0.5
CUBE_SIZE = 0.044
PLATFORM_SIZE = CUBE_SIZE * 3.0
PLATFORM_HEIGHT = CUBE_SIZE * 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table() -> Table:
    return Table(height=TABLE_HEIGHT, name="table")


def _make_platform() -> Platform:
    return Platform(
        width=PLATFORM_SIZE,
        depth=PLATFORM_SIZE,
        height=PLATFORM_HEIGHT,
        name="platform",
        init_pose=torch.tensor([0.45, 0.28, TABLE_HEIGHT + PLATFORM_HEIGHT / 2.0, 1.0, 0.0, 0.0, 0.0]),
    )


def _make_discard() -> DiscardLocation:
    return DiscardLocation(
        width=PLATFORM_SIZE,
        depth=PLATFORM_SIZE,
        name="discard",
        init_pose=torch.tensor([0.45, -0.28, TABLE_HEIGHT + 0.001, 1.0, 0.0, 0.0, 0.0]),
    )


def _make_blocks(defective_flags: list[bool]) -> list[InspectableCube]:
    spacing = CUBE_SIZE * 3.0
    y_start = -(len(defective_flags) - 1) * spacing / 2.0
    return [
        InspectableCube(
            size=CUBE_SIZE,
            defective=flag,
            name=f"block_{i}",
            init_pose=torch.tensor([0.35, y_start + i * spacing, TABLE_HEIGHT + CUBE_SIZE / 2.0, 1.0, 0.0, 0.0, 0.0]),
        )
        for i, flag in enumerate(defective_flags)
    ]


# ---------------------------------------------------------------------------
# Domain parsing
# ---------------------------------------------------------------------------

def test_domain_file_exists() -> None:
    assert DOMAIN_FILE.exists(), f"domain file not found: {DOMAIN_FILE}"


def test_domain_parses_without_error() -> None:
    """PDDLReader must parse the domain without raising."""
    reader = PDDLReader()
    problem = reader.parse_problem(str(DOMAIN_FILE))
    assert problem is not None


def test_domain_has_expected_actions() -> None:
    reader = PDDLReader()
    problem = reader.parse_problem(str(DOMAIN_FILE))
    action_names = {a.name for a in problem.actions}
    assert action_names == {"approach-block", "inspect-for-defects", "pick", "place"}


def test_domain_has_expected_predicates() -> None:
    reader = PDDLReader()
    problem = reader.parse_problem(str(DOMAIN_FILE))
    fluent_names = {f.name for f in problem.fluents}
    assert {"on", "holding", "gripper-empty", "gripper-above", "inspected", "defective", "non-defective"}.issubset(
        fluent_names
    )


# ---------------------------------------------------------------------------
# Problem factory — string structure
# ---------------------------------------------------------------------------

def test_problem_factory_single_clean_block() -> None:
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([False])
    pddl = make_inspection_problem(table, blocks, platform, discard)
    assert "(on block_0 platform)" in pddl
    assert "(non-defective block_0)" in pddl
    assert "(gripper-empty)" in pddl


def test_problem_factory_single_defective_block() -> None:
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([True])
    pddl = make_inspection_problem(table, blocks, platform, discard)
    assert "(on block_0 discard)" in pddl
    assert "(defective block_0)" in pddl


def test_problem_factory_mixed_blocks() -> None:
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([False, True, False])
    pddl = make_inspection_problem(table, blocks, platform, discard)
    assert "(on block_0 platform)" in pddl
    assert "(on block_1 discard)" in pddl
    assert "(on block_2 platform)" in pddl


def test_problem_factory_raises_on_unlabelled_block() -> None:
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = [InspectableCube(size=CUBE_SIZE, defective=None, name="block_0")]
    with pytest.raises(ValueError, match="no defect label"):
        make_inspection_problem(table, blocks, platform, discard)


def test_problem_factory_uses_scene_object_names() -> None:
    """Object names in the PDDL must match the scene-graph names."""
    table = Table(height=TABLE_HEIGHT, name="worktable")
    platform = Platform(
        width=PLATFORM_SIZE, depth=PLATFORM_SIZE, height=PLATFORM_HEIGHT, name="good_pile",
        init_pose=torch.tensor([0.45, 0.28, TABLE_HEIGHT + PLATFORM_HEIGHT / 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    discard = DiscardLocation(
        width=PLATFORM_SIZE, depth=PLATFORM_SIZE, name="bad_bin",
        init_pose=torch.tensor([0.45, -0.28, TABLE_HEIGHT + 0.001, 1.0, 0.0, 0.0, 0.0]),
    )
    blocks = _make_blocks([True])
    pddl = make_inspection_problem(table, blocks, platform, discard)
    assert "worktable - table0" in pddl
    assert "good_pile - platform0" in pddl
    assert "bad_bin - discard0" in pddl
    assert "(on block_0 bad_bin)" in pddl


# ---------------------------------------------------------------------------
# End-to-end planning
# ---------------------------------------------------------------------------

def _parse_full_problem(pddl_str: str):
    reader = PDDLReader()
    with tempfile.NamedTemporaryFile(suffix=".pddl", mode="w", delete=False) as f:
        f.write(pddl_str)
        problem_path = f.name
    return reader.parse_problem(str(DOMAIN_FILE), problem_path)


def test_planner_finds_plan_single_clean_block() -> None:
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([False])
    problem = _parse_full_problem(make_inspection_problem(table, blocks, platform, discard))
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem, timeout=30.0)
    assert result.status in (PGStatus.SOLVED_SATISFICING, PGStatus.SOLVED_OPTIMALLY)


def test_planner_finds_plan_single_defective_block() -> None:
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([True])
    problem = _parse_full_problem(make_inspection_problem(table, blocks, platform, discard))
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem, timeout=30.0)
    assert result.status in (PGStatus.SOLVED_SATISFICING, PGStatus.SOLVED_OPTIMALLY)


def test_plan_sequence_single_clean_block() -> None:
    """Plan for one clean block must be: approach → inspect → pick → place-on-platform."""
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([False])
    problem = _parse_full_problem(make_inspection_problem(table, blocks, platform, discard))
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem, timeout=30.0)
    assert result.status in (PGStatus.SOLVED_SATISFICING, PGStatus.SOLVED_OPTIMALLY)
    action_names = [str(a).split("(")[0].strip() for a in result.plan.actions]
    assert action_names == ["approach-block", "inspect-for-defects", "pick", "place"]


def test_plan_sequence_single_defective_block() -> None:
    """Plan for one defective block must be: approach → inspect → pick → place (to discard)."""
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([True])
    problem = _parse_full_problem(make_inspection_problem(table, blocks, platform, discard))
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem, timeout=30.0)
    assert result.status in (PGStatus.SOLVED_SATISFICING, PGStatus.SOLVED_OPTIMALLY)
    action_names = [str(a).split("(")[0].strip() for a in result.plan.actions]
    assert action_names == ["approach-block", "inspect-for-defects", "pick", "place"]


def test_planner_finds_plan_mixed_three_blocks() -> None:
    """Planner must handle three blocks with mixed defect labels."""
    table, platform, discard = _make_table(), _make_platform(), _make_discard()
    blocks = _make_blocks([False, True, False])
    problem = _parse_full_problem(make_inspection_problem(table, blocks, platform, discard))
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem, timeout=30.0)
    assert result.status in (PGStatus.SOLVED_SATISFICING, PGStatus.SOLVED_OPTIMALLY)
    assert len(result.plan.actions) == 12  # 4 actions per block × 3 blocks
