"""Inspection pick-and-place task orchestrator.

Drives the PDDL plan using skill execute() calls and logs each event.

Pass ``env`` (an ``InspectionMjEnv``) to ``run_demo()`` to enable physical arm
motion.  Without it the orchestrator runs in world-model-only mode (Phase 1
behaviour): preconditions are checked, the scene graph is updated, but MuJoCo
never steps.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from unified_planning.engines import PlanGenerationResultStatus as PGStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

from skillet.core.checked_skill import SkillResult
from skillet.core.skill import SkillStatusCodes
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
from skillet.skill.skill_lib import make_pick_skill, make_reach_xyzrpy_skill
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import (
    make_inspection_scene,
)

if TYPE_CHECKING:
    from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.env import InspectionMjEnv

_DOMAIN_FILE = Path(__file__).parents[4] / "skillet" / "planning" / "inspection" / "inspection.domain.pddl"
_BLANK_IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)
# Raise TCP above block centre so the 2F-85 4-bar follower joint clears the table.
# At block-centre z (0.022 m base), the follower reaches ≈ −21 mm (table at 0 m).
# Adding 30 mm shifts it to +9 mm, clearing the table while still grasping the block top.
_PICK_LOWER_Z_OFFSET_M: float = 0.03


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
    env: InspectionMjEnv | None = None,
    screenshot_dir: str | Path | None = None,
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
    _screenshot_dir: Path | None = None
    if screenshot_dir is not None:
        _screenshot_dir = Path(screenshot_dir)
        _screenshot_dir.mkdir(parents=True, exist_ok=True)

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
        _execute_plan(scene, plan_actions, classifier, logger, env, _screenshot_dir)
        logger.log_world_model_snapshot(scene)

    return _compute_metrics(spec.blocks, ground_truth, spec.platform, spec.discard)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_wrist_image(env: InspectionMjEnv, path: Path) -> None:
    """Capture the wrist camera and save to *path* as PNG (falls back to .npy)."""
    pixels = env.capture_wrist_cam()
    try:
        from PIL import Image  # type: ignore[import-untyped]
        Image.fromarray(pixels).save(path)
    except ImportError:
        np.save(path.with_suffix(".npy"), pixels)


def _run_skill_loop(skill: object, env: object, max_steps: int = 200) -> None:
    """Step the env until skill reaches SUCCESS, FAILED, or the budget runs out."""
    for _ in range(max_steps):
        obs = env.get_observation()  # type: ignore[union-attr]
        action = skill.get_action(obs)  # type: ignore[union-attr]
        env.step(action)  # type: ignore[union-attr]
        status_val = int(skill.status[0].item())  # type: ignore[union-attr]
        if status_val == SkillStatusCodes.SUCCESS:
            return
        if status_val == SkillStatusCodes.FAILED:
            raise RuntimeError(f"Skill {skill.name!r} returned FAILED")  # type: ignore[union-attr]
    raise RuntimeError(f"Skill {skill.name!r} timed out after {max_steps} steps")  # type: ignore[union-attr]


def _settle(env: object, n_steps: int = 20) -> None:
    """Hold the arm still for ``n_steps`` to let released objects settle under gravity."""
    obs = env.get_observation()  # type: ignore[union-attr]
    action = obs["joint_pos"].clone()   # repeat last joint targets
    for _ in range(n_steps):
        env.step(action)  # type: ignore[union-attr]


def _physical_place(
    env: InspectionMjEnv,
    dest_x_b: float,
    dest_y_b: float,
    lift_height_b: float = 0.3,
    skill_length: int = 500,
    n_gripper_steps: int = 40,
    n_settle_steps: int = 120,
) -> None:
    """Hover above destination and open gripper so the held block drops onto the surface.

    PlaceSkill's 5 mm IK threshold cannot be met at the extended platform/discard
    positions (x ≈ 0.40 m, y ≈ ±0.28 m in base frame) when the arm must also lower
    to z ≈ 0.14 m.  ReachXYZRPYSkill uses a 2 cm threshold and converges reliably
    at lift height (z = 0.30 m), so we hover there and drop the block.  The settle
    budget (120 steps at 500 Hz) covers a ≈ 0.20 m free-fall.
    """
    reach_skill = make_reach_xyzrpy_skill(env, skill_length=skill_length)
    hover_params = torch.tensor([[dest_x_b, dest_y_b, lift_height_b, math.pi, 0.0, 0.0]], dtype=torch.float32)
    obs = env.get_observation()
    reach_skill.initiate(obs, hover_params)
    _run_skill_loop(reach_skill, env, max_steps=skill_length)

    # Open gripper — held block drops freely to the destination surface
    obs = env.get_observation()
    action = obs["joint_pos"].clone()
    action[:, -1] = 0.0
    for _ in range(n_gripper_steps):
        env.step(action)

    _settle(env, n_steps=n_settle_steps)


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
    env: InspectionMjEnv | None = None,
    screenshot_dir: Path | None = None,
) -> None:
    """Execute each grounded PDDL action via the appropriate skill.

    When ``env`` is provided the arm physically moves in MuJoCo.
    When ``env`` is None (default) only the world model is updated (Phase 1
    behaviour).
    """
    held_block: InspectableCube | None = None

    for action_instance in plan_actions:
        action_name, params = _parse_action(action_instance)
        logger.log_planner_decision(action_name, params)

        block_name = params[0]
        block_id = scene.resolve_names_to_ids([block_name])[0]

        if action_name == "approach-block":
            if env is not None:
                # arm physically moves to the inspection viewpoint above the block.
                # InspectSkill._compute_viewpoint converts the block's world-frame pose to
                # robot-base frame using robot_base_world_pos, then calls reach_skill.
                reach_skill = make_reach_xyzrpy_skill(env, skill_length=200)
                skill = InspectSkill(
                    scene,
                    env=env,
                    reach_skill=reach_skill,
                    robot_base_world_pos=env.robot_base_world_pos,
                )
            else:
                skill = InspectSkill(scene)
            skill.set_target(block_id)
            logger.log_skill_start("InspectSkill", params=block_id)
            result = skill.execute(scene)
            logger.log_skill_end("InspectSkill", result)
            if not result.success:
                raise RuntimeError(f"InspectSkill failed for {block_name}: {result}")
            if env is not None and screenshot_dir is not None:
                _save_wrist_image(env, screenshot_dir / f"inspect_{block_name}.png")

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
            if env is not None:
                base_pos = env.robot_base_world_pos
                skill_params = torch.tensor([[
                    float(block.pose[0]) - float(base_pos[0]),
                    float(block.pose[1]) - float(base_pos[1]),
                    float(block.pose[2]) - float(base_pos[2]) + _PICK_LOWER_Z_OFFSET_M,
                    0.0,
                ]])
                pick_skill = make_pick_skill(env, skill_length=1000)
                obs = env.get_observation()
                pick_skill.initiate(obs, skill_params)
                _run_skill_loop(pick_skill, env, max_steps=1000)
            logger.log_skill_end("PickSkill", SkillResult.ok())

        elif action_name == "place":
            dest_name = params[1]
            dest_objs = scene.get_objects_from_name([dest_name])
            dest = dest_objs[0]
            assert held_block is not None, "place called without a prior pick"
            logger.log_skill_start("PlaceSkill", params={"block": block_name, "dest": dest_name})
            if env is not None:
                base_pos = env.robot_base_world_pos
                dest_x_b = float(dest.pose[0]) - float(base_pos[0])
                dest_y_b = float(dest.pose[1]) - float(base_pos[1])
                _physical_place(env, dest_x_b, dest_y_b)
                # Sync the scene-graph pose from the actual MuJoCo block position.
                world_pos = env.get_block_world_pos(held_block.name)
                held_block._pose = torch.tensor(
                    [world_pos[0], world_pos[1], world_pos[2], 1.0, 0.0, 0.0, 0.0],
                    dtype=torch.float32,
                )
            else:
                held_block._pose = dest.pose.clone()
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