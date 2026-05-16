"""A Task and Motion Planner executor for running an agent in an environment."""

import time
from typing import Any

from skillet.agents.base_agent import Agent
from skillet.core.env import Environment
from skillet.core.policy import Unparameterized
from skillet.core.skill import SingleSkill, SkillStatusCodes
from skillet.logging import SkilletDataLogger
from skillet.perception.perception import SkilletPerception
from skillet.planning import AbstractModel
from skillet.planning.abstract.up_utils import sample_action_from_state
from skillet.scene.base import Scene
from skillet.agents import SkilletModerator


class PlanningAgent(Agent):
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
        super().__init__()

        self._scene = scene
        self.abstract_model = abstract_model
        self.action_to_skill_map = action_to_skill_map

    def execute(self, env: Environment[Any, Any], task: str | None = None) -> None:
        """Execute the policy over the options configured.

        Args:
            env: The environment to execute the policy over.
            task: The task to execute.

        """
        # Get the current symbolic state
        self.abstract_model.initialize(self._scene, task)

        abstract_state = self.abstract_model.get_abstract_state()
        self._result, self._plan = self.abstract_model.plan(abstract_state=abstract_state)

        terminated = False
        cum_reward = 0.0
        if self._plan is None:
            print("[WARNING][TAMP] Failed to find plan.")
            return
        for ab_action in self._plan.actions:
            self._selected_skill = self.action_to_skill_map[ab_action.action]
            args = self._scene.resolve_names_to_ids(ab_action.parameters)

            obs = env.get_observation(self._selected_skill.obs_spec)
            self._selected_skill.initiate(obs, args)
            skill_done = self._selected_skill.is_terminated(env.get_observation(self._selected_skill.obs_spec))
            while not skill_done and not bool(terminated):
                # Get the next action with the low-level observation
                action = self._selected_skill.get_action(env.get_observation(self._selected_skill.obs_spec))
                # Take a step in the environment
                _, r, term, trunc, _ = env.step(action, action_spec=self._selected_skill.action_spec)
                cum_reward += r
                terminated = terminated | term | trunc
                # Check if the skill is terminated
                skill_done = self._selected_skill.is_terminated(env.get_observation(self._selected_skill.obs_spec))
            # Check if the skill was successful
            if self._selected_skill.status != SkillStatusCodes.SUCCESS:
                break
            if terminated:
                break
            time.sleep(2)  # Give the EMA filter time to catch up


class RandomTampAgent(Agent):
    """A Task-Planning agent that randomly executes a sequence of valid actions."""

    def __init__(
        self,
        scene: Scene,
        abstract_model: AbstractModel,
        action_to_skill_map: dict[str, SingleSkill[Any, Any, Unparameterized]],
        perception: SkilletPerception | None = None,
    ) -> None:
        """Initialize the planning agent.

        Args:
            scene: The scene to execute the skills in.
            abstract_model: The abstract model of the scene.
            action_to_skill_map: A map of actions to skills.

        """
        super().__init__()

        self._scene = scene
        self.abstract_model = abstract_model
        self.action_to_skill_map = action_to_skill_map
        self._perception = perception
        self._moderator = SkilletModerator()

    def execute(
        self,
        env: Environment[Any, Any],
        task: str | None = None,
        num_actions: int = 10,
        logger: SkilletDataLogger = None,
    ) -> None:
        """Execute the policy over the options configured.

        Args:
            env: The environment to execute the policy over.
            task: The task to execute.
            num_actions: the number of actions to execute

        """
        # Get the current symbolic state
        self.abstract_model.initialize(self._scene, task)

        terminated = False

        for i in range(num_actions):
            # self._perception.update_state()
            up_state = self.abstract_model.reset_up_problem_state()
            # ab_action, up_action = self.abstract_model.get_random_action(up_state)
            ab_action, up_action = sample_action_from_state(self.abstract_model._problem, up_state)
            self._selected_skill = self.action_to_skill_map[ab_action.action]
            args = self._scene.resolve_names_to_ids(ab_action.parameters)

            terminated = self._moderator.run_skill(env, self._selected_skill, args)

            if logger is not None:
                obs_log = env.get_observation(logger._obs_spec)
                logger.log(
                    save_log=True,
                    rgb=obs_log["rgb"],
                    depth=obs_log["depth"],
                    tcp_pose_b=self._scene.tcp_pose,
                    gripper=self._scene.gripper_pos,
                    parsed_scene=self._scene,
                    skill=ab_action,
                    skill_status=self._selected_skill.status,
                    states=up_state,
                    actions=up_action,
                    executions="applicable",
                )

            if terminated:
                break
            time.sleep(4)
        if logger is not None:
            obs_log = env.get_observation(logger._obs_spec)
            logger.log(
                save_log=True,
                rgb=obs_log["rgb"],
                depth=obs_log["depth"],
                tcp_pose_b=self._scene.tcp_pose,
                gripper=self._scene.gripper_pos,
                parsed_scene=self._scene,
                skill=ab_action,
                skill_status=self._selected_skill.status,
                states=up_state,
                actions=up_action,
                executions="applicable",
            )


class RandomStateAgent(Agent):
    """A Task-Planning agent that randomly executes a sequence of valid actions."""

    def __init__(
        self,
        scene: Scene,
        abstract_model: AbstractModel,
        action_to_skill_map: dict[str, SingleSkill[Any, Any, Unparameterized]],
        perception: SkilletPerception | None = None,
    ) -> None:
        """Initialize the planning agent.

        Args:
            scene: The scene to execute the skills in.
            abstract_model: The abstract model of the scene.
            action_to_skill_map: A map of actions to skills.

        """
        super().__init__()

        self._scene = scene
        self.abstract_model = abstract_model
        self.action_to_skill_map = action_to_skill_map
        self._perception = perception

    def execute(
        self,
        env: Environment[Any, Any],
        task: str | None = None,
        num_actions: int = 10,
        logger: SkilletDataLogger = None,
    ) -> None:
        """Execute the policy over the options configured.

        Args:
            env: The environment to execute the policy over.
            task: The task to execute.
            num_actions: the number of actions to execute

        """
        # Get the current symbolic state
        self.abstract_model.initialize(self._scene, task)

        terminated = False
        cum_reward = 0.0

        for i in range(num_actions):
            # self._perception.update_state()
            up_state = self.abstract_model.get_abstract_state()

            # ab_action, up_action = self.abstract_model.get_random_action(up_state)
            # ab_action, up_action = sample_action_from_state(self.abstract_model._problem, up_state)

            time.sleep(3)
