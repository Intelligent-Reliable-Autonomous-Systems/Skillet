"""A Task and Motion Planner executor for running an agent in an environment."""

from typing import Any

from skillet.core.env import Environment
from skillet.core.policy import Unparameterized
from skillet.core.skill import SingleSkill, SkillStatusCodes
from skillet.logging import SkilletDataLogger
from skillet.scene.abstract.abstract_model import AbstractModel
from skillet.scene.base import Scene


class PlanningAgent:
    """A Task-Planning agent that plans a sequence of skills to execute to complete a task."""

    def __init__(
        self,
        scene: Scene,
        abstract_model: AbstractModel,
        action_to_skill_map: dict[str, SingleSkill[Any, Any, Unparameterized]],
    ) -> None:
        """Initialize the planning agent.

        Args:
            scene: The scene to execute the skills in.
            abstract_model: The abstract model of the scene.
            action_to_skill_map: A map of actions to skills.

        """
        self._scene = scene
        self.abstract_model = abstract_model
        self.action_to_skill_map = action_to_skill_map

    def execute(
        self, env: Environment[Any, Any], task: str | None = None, data_logger: SkilletDataLogger | None = None
    ) -> None:
        """Execute the policy over the options configured.

        Args:
            env: The environment to execute the policy over.
            task: The task to execute.

        """
        # Get the current symbolic state
        self.abstract_model.initialize(self._scene, task)

        abstract_state = self.abstract_model.get_abstract_state()
        result, plan = self.abstract_model.plan(abstract_state=abstract_state)

        terminated = False
        cum_reward = 0.0

        if plan is None:
            print("[WARNING][TAMP] Failed to find plan.")
            return
        for ab_action in plan.actions:
            selected_skill = self.action_to_skill_map[ab_action.action]
            args = self._scene.resolve_names_to_ids(ab_action.parameters)

            obs = env.get_observation(selected_skill.obs_spec)
            selected_skill.initiate(obs, args)
            skill_done = selected_skill.is_terminated(env.get_observation(selected_skill.obs_spec))
            while not skill_done and not bool(terminated):
                # Get the next action with the low-level observation
                action = selected_skill.get_action(env.get_observation(selected_skill.obs_spec))
                # Take a step in the environment
                _, r, term, trunc, _ = env.step(action, action_spec=selected_skill.action_spec)
                cum_reward += r
                terminated = terminated | term | trunc
                # Check if the skill is terminated
                skill_done = selected_skill.is_terminated(env.get_observation(selected_skill.obs_spec))
            # Check if the skill was successful
            if selected_skill.status != SkillStatusCodes.SUCCESS:
                break
            if terminated:
                break
