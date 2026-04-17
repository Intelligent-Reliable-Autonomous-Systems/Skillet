"""An agent that uses an LLM planner with scene-predicate verification."""

import time
from typing import Any

import numpy as np
from PIL import Image

from skillet.agents.base_agent import Agent
from skillet.agents.llm_planner import LLMPlanner
from skillet.core.env import Environment
from skillet.core.skill import SingleSkill, SkillStatusCodes
from skillet.perception import SkilletPerception
from skillet.scene.abstract.spatial_grounding import (
    ground_cube_on_relations,
    ground_gripper_relations,
)
from skillet.scene.base import Scene


class LLMPlanningAgent(Agent):
    """An agent that uses an LLM to plan skill sequences.

    Each action is verified against live scene predicates after it executes.
    Execution halts on the first failure — no replanning.

    Args:
        scene: The scene with object poses.
        planner: LLMPlanner instance.
        action_to_skill_map: Maps action names to skill instances.
        goal: Natural language goal.

    """

    def __init__(
        self,
        scene: Scene,
        planner: LLMPlanner,
        action_to_skill_map: dict[str, SingleSkill],
        goal: str,
        perception: SkilletPerception | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            perception: Optional perception instance. Required when the
                planner uses VLM-based predicates (so the agent can capture a
                workspace image to pass into ``planner.plan``).

        """
        super().__init__()
        self._scene = scene
        self._planner = planner
        self._action_to_skill_map = action_to_skill_map
        self._goal = goal
        self._perception = perception

    def _capture_image(self) -> Image.Image | None:
        """Capture the current workspace RGB image from perception."""
        if self._perception is None:
            print("[LLMAgent] No perception attached; cannot capture image.")
            return None

        obs = self._perception.latest_observation
        if obs is None:
            print("[LLMAgent] perception.latest_observation is None; cannot capture image.")
            return None

        try:
            rgb = obs["rgb"]
        except (KeyError, TypeError) as e:
            print(f"[LLMAgent] Could not read 'rgb' from latest_observation: {e}")
            return None

        rgb_np = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        if rgb_np.shape[0] == 3:
            rgb_np = rgb_np.transpose(1, 2, 0)

        return Image.fromarray(rgb_np.astype(np.uint8))

    def _execute_plan(self, env: Environment[Any, Any]) -> bool:
        """Execute the current plan with scene-predicate verification.

        Returns True if every action in the plan completed and verified.
        """
        if self._plan is None:
            return False

        terminated = False

        for ab_action in self._plan.actions:
            if ab_action.action not in self._action_to_skill_map:
                print(f"[LLMAgent] Unknown action: {ab_action.action}")
                return False

            self._selected_skill = self._action_to_skill_map[ab_action.action]
            ids = self._scene.resolve_names_to_ids(ab_action.parameters)

            # PickBlockSkill takes a single Discrete block id (source block).
            # PlaceBlock2Skill takes [source_id, target_id] so it can dispatch
            # on whether the target is a Cube or a Table.
            if ab_action.action == "pick_block":
                skill_args = ids[0]
            elif ab_action.action == "place_block":
                skill_args = ids
            else:
                skill_args = ids

            obs = env.get_observation(self._selected_skill.obs_spec)
            self._selected_skill.initiate(obs, skill_args)

            skill_done = self._selected_skill.is_terminated(
                env.get_observation(self._selected_skill.obs_spec)
            )
            while not skill_done and not bool(terminated):
                action = self._selected_skill.get_action(
                    env.get_observation(self._selected_skill.obs_spec)
                )
                _, r, term, trunc, _ = env.step(
                    action, action_spec=self._selected_skill.action_spec
                )
                terminated = terminated | term | trunc
                skill_done = self._selected_skill.is_terminated(
                    env.get_observation(self._selected_skill.obs_spec)
                )

            if self._selected_skill.status != SkillStatusCodes.SUCCESS:
                print(f"[LLMAgent] Skill {ab_action.action}({ab_action.parameters}) failed (status).")
                return False

            if terminated:
                print("[LLMAgent] Environment terminated.")
                return False

            verified, explanation = self._verify_action_from_scene(
                ab_action.action, ab_action.parameters
            )
            if verified:
                print(f"[LLMAgent] Verified {ab_action.action}({ab_action.parameters}): {explanation}")
            else:
                print(f"[LLMAgent] Verification failed for {ab_action.action}({ab_action.parameters}): {explanation}")
                return False

        return True

    def _verify_action_from_scene(
        self, action: str, params: list[str], timeout_s: float = 3.0, poll_s: float = 0.25
    ) -> tuple[bool, str]:
        """Confirm an action's post-condition by polling scene predicates."""
        start = time.time()
        last_reason = "no predicate check executed"
        while time.time() - start < timeout_s:
            on_preds, _ = ground_cube_on_relations(self._scene)
            _, holding_preds = ground_gripper_relations(self._scene)

            if action == "pick_block" and len(params) >= 1:
                block_name = params[0]
                if any(h[1].name == block_name for h in holding_preds):
                    return True, f"scene shows gripper holding {block_name}"
                currently_holding = [h[1].name for h in holding_preds] or "nothing"
                last_reason = f"gripper is holding {currently_holding}, expected {block_name}"
            elif action == "place_block" and len(params) >= 2:
                block_name, surface_name = params[0], params[1]
                if any(o[1].name == block_name and o[2].name == surface_name for o in on_preds):
                    return True, f"scene shows {block_name} on {surface_name}"
                on_what = next((o[2].name for o in on_preds if o[1].name == block_name), "nothing")
                last_reason = f"{block_name} is on {on_what}, expected {surface_name}"
            else:
                return True, f"no predicate rule for {action}; skipping verification"

            time.sleep(poll_s)

        return False, f"timed out after {timeout_s:.1f}s — {last_reason}"

    def execute(self, env: Environment[Any, Any]) -> None:
        """Plan once and execute. Stop on the first failure."""
        image = self._capture_image()
        success, self._plan = self._planner.plan(self._scene, self._goal, image=image)
        if not success or self._plan is None:
            print("[LLMAgent] Failed to generate plan.")
            return

        print(f"[LLMAgent] Plan: {self._plan}")

        if self._execute_plan(env):
            print("[LLMAgent] Plan executed successfully.")
        else:
            print("[LLMAgent] Execution halted.")