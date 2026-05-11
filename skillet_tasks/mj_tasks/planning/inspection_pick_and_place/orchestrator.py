"""Inspection pick-and-place task orchestrator.

Drives the PDDL plan using skill execute() calls and logs each event.

The MuJoCo model is loaded for scene validation and can be rendered, 
but actual arm motion control (IK/PickSkill/PlaceSkill) is yet to be implemented.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from unified_planning.engines import PlanGenerationResultStatus as PGStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

from skillet.core.checked_skill import SkillResult
from skillet.logging.event_logger import SkillEventLogger
from skillet.perception.inspection.defect_classifier import DefectClassifier, DefectResult
from skillet.perception.inspection.mock_defect_classifier import MockDefectClassifier
from skillet.planning.inspection import make_inspection_problem
from skillet.scene.base import Scene
from skillet.scene.objects.discard_location import DiscardLocation
from skillet.scene.objects.inspectable_cube import InspectableCube
from skillet.scene.objects.platform import Platform
from skillet.skill.high_level.inspect import InspectSkill
from skillet.skill.high_level.inspect_for_defects import InspectForDefectsSkill
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import (
    make_inspection_scene,
)

_DOMAIN_FILE = Path(__file__).parents[4] / "skillet" / "planning" / "inspection" / "inspection.domain.pddl"
_BLANK_IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)


@dataclass
class TaskMetrics:
    """Outcome statistics for one run of the task."""

    n_blocks: int
    n_correct_verdict: int
    n_correct_route: int

    @property
    def defect_accuracy(self) -> float:
        """Fraction of blocks where classifier verdict matches ground truth."""
        return self.n_correct_verdict / self.n_blocks if self.n_blocks else 1.0

    @property
    def routing_accuracy(self) -> float:
        """Fraction of blocks routed to the correct destination."""
        return self.n_correct_route / self.n_blocks if self.n_blocks else 1.0


def run_demo(
    block_defective: list[bool],
    log_dir: str | Path,
    run_id: str | None = None,
    planner_name: str = "fast-downward",
    planner_timeout: float = 30.0,
) -> TaskMetrics:
    """Execute the inspection pick-and-place task.

    Builds the MuJoCo scene, plans with PDDL, and executes each action via
    the skill execute() interface with MockDefectClassifier.

    Args:
        block_defective: Ground-truth defect flag per block.
        log_dir: Directory for events.jsonl (created if absent).
        run_id: Shared run id for event logging.
        planner_name: UP planner name (default: fast-downward).
        planner_timeout: Planner timeout in seconds.

    Returns:
        TaskMetrics with defect-accuracy and routing-accuracy.

    Raises:
        RuntimeError: If the planner fails to find a plan.

    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build MuJoCo scene (validates MJCF + textures)
    spec = make_inspection_scene(block_defective)

    # Save ground truth and build classifier BEFORE resetting blocks
    ground_truth: dict[str, bool] = {b.name: b.defective for b in spec.blocks}  # type: ignore[assignment]
    classifier = MockDefectClassifier({
        name: DefectResult(defective=label, confidence=0.95)
        for name, label in ground_truth.items()
    })

    # Build PDDL problem from ground-truth-labelled scene
    problem_str = make_inspection_problem(spec.table, spec.blocks, spec.platform, spec.discard)

    # Reset blocks so skills see them as "not yet inspected"
    for b in spec.blocks:
        b._defective = None

    # Build scene graph (assigns object ids)
    scene = Scene(objects=[spec.table, *spec.blocks, spec.platform, spec.discard])

    # Plan
    plan_actions = _plan(problem_str, planner_name, planner_timeout)

    # Execute with logging
    with SkillEventLogger(log_dir / "events.jsonl", run_id=run_id) as logger:
        logger.log_world_model_snapshot(scene)
        _execute_plan(scene, plan_actions, classifier, logger)
        logger.log_world_model_snapshot(scene)

    return _compute_metrics(spec.blocks, ground_truth, spec.platform, spec.discard)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plan(problem_str: str, planner_name: str, timeout: float) -> list:
    reader = PDDLReader()
    with tempfile.NamedTemporaryFile(suffix=".pddl", mode="w", delete=False, encoding="utf-8") as f:
        f.write(problem_str)
        problem_path = f.name
    problem = reader.parse_problem(str(_DOMAIN_FILE), problem_path)
    with OneshotPlanner(name=planner_name) as planner:
        result = planner.solve(problem, timeout=timeout)
    if result.status not in (PGStatus.SOLVED_SATISFICING, PGStatus.SOLVED_OPTIMALLY):
        raise RuntimeError(f"PDDL planner failed: {result.status}")
    return list(result.plan.actions)


def _parse_action(action_instance) -> tuple[str, list[str]]:
    """Return (action_name, [param_name, ...]) from a UP ActionInstance."""
    full = str(action_instance)
    name, rest = full.split("(", 1)
    params = [p.strip() for p in rest.rstrip(")").split(",")]
    return name.strip(), [p for p in params if p]


def _execute_plan(
    scene: Scene,
    plan_actions: list,
    classifier: DefectClassifier,
    logger: SkillEventLogger,
) -> None:
    """Execute each grounded PDDL action via the appropriate skill."""
    held_block: InspectableCube | None = None

    for action_instance in plan_actions:
        action_name, params = _parse_action(action_instance)
        logger.log_planner_decision(action_name, params)

        block_name = params[0]
        block_id = scene.resolve_names_to_ids([block_name])[0]

        if action_name == "approach-block":
            skill = InspectSkill(scene)
            skill.set_target(block_id)
            logger.log_skill_start("InspectSkill", params=block_id)
            result = skill.execute(scene)
            logger.log_skill_end("InspectSkill", result)
            if not result.success:
                raise RuntimeError(f"InspectSkill failed for {block_name}: {result}")

        elif action_name == "inspect-for-defects":
            skill = InspectForDefectsSkill(scene, classifier)
            skill.set_target(block_id)
            logger.log_skill_start("InspectForDefectsSkill", params=block_id)
            result = skill.execute(scene, _BLANK_IMAGE)
            block = scene.get_objects_from_id([block_id])[0]
            assert isinstance(block, InspectableCube)
            logger.log_classifier_verdict(block_name, bool(block.defective), 0.95)
            logger.log_skill_end("InspectForDefectsSkill", result)
            if not result.success:
                raise RuntimeError(f"InspectForDefectsSkill failed for {block_name}: {result}")

        elif action_name == "pick":
            block = scene.get_objects_from_id([block_id])[0]
            assert isinstance(block, InspectableCube)
            held_block = block
            logger.log_skill_start("PickSkill", params=block_id)
            logger.log_skill_end("PickSkill", SkillResult.ok())

        elif action_name == "place":
            dest_name = params[1]
            dest_objs = scene.get_objects_from_name([dest_name])
            dest = dest_objs[0]
            assert held_block is not None, "place called without a prior pick"
            # Move block to destination centre (simplified — no physics)
            held_block._pose = dest.pose.clone()
            logger.log_skill_start("PlaceSkill", params={"block": block_name, "dest": dest_name})
            logger.log_skill_end("PlaceSkill", SkillResult.ok())
            held_block = None


def _compute_metrics(
    blocks: list[InspectableCube],
    ground_truth: dict[str, bool],
    platform: Platform,
    discard: DiscardLocation,
) -> TaskMetrics:
    n_correct_verdict = 0
    n_correct_route = 0

    for b in blocks:
        if b.defective == ground_truth[b.name]:
            n_correct_verdict += 1

        gt_defective = ground_truth[b.name]
        target_aabb = discard.aabb if gt_defective else platform.aabb
        bx, by = float(b.pose[0]), float(b.pose[1])
        if float(target_aabb[0]) <= bx <= float(target_aabb[3]) and float(target_aabb[1]) <= by <= float(target_aabb[4]):
            n_correct_route += 1

    return TaskMetrics(
        n_blocks=len(blocks),
        n_correct_verdict=n_correct_verdict,
        n_correct_route=n_correct_route,
    )