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
from skillet.planning.abstract import AbstractAction
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
        self._moderator = SkilletModerator()

    def execute(
        self,
        env: Environment[Any, Any],
        task: str | None = None,
        logger: SkilletDataLogger = None,
    ) -> None:
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
        if self._plan is None:
            print("[WARNING][TAMP] Failed to find plan.")
            return
        for ab_action in self._plan.actions:
            up_state = self.abstract_model.reset_up_problem_state()
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
                    skill=self._selected_skill.name,
                    skill_status=self._selected_skill.status,
                    states=up_state,
                    actions=ab_action,
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
                skill=self._selected_skill.name,
                skill_status=self._selected_skill.status,
                states=up_state,
                actions=ab_action,
                executions="applicable",
            )


class RandomTampAgent(Agent):
    """A Task-Planning agent that randomly executes a sequence of valid actions."""

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
            up_state = self.abstract_model.reset_up_problem_state()
            # ab_action, up_action = self.abstract_model.get_random_action(up_state)
            ab_action, _ = sample_action_from_state(self.abstract_model._problem, up_state)
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
                    skill=self._selected_skill.name,
                    skill_status=self._selected_skill.status,
                    states=up_state,
                    actions=ab_action,
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
                skill=self._selected_skill.name,
                skill_status=self._selected_skill.status,
                states=up_state,
                actions=ab_action,
                executions="applicable",
            )


class ActiveLearningAgent(Agent):
    """A Task-Planning agent that actively selects an action."""

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
        self._moderator = SkilletModerator()
        self._learning_agent = None

    def execute(
        self,
        env: Environment[Any, Any],
        task: str | None = None,
        logger: SkilletDataLogger = None,
    ) -> None:
        """Execute the policy over the options configured.

        Args:
            env: The environment to execute the policy over.
            task: The task to execute.

        """
        # Get the current symbolic state
        self.abstract_model.initialize(self._scene, task)

        terminated = False

        while True:
            up_state = self.abstract_model.reset_up_problem_state()

            # TODO Sample an action from the learning agent
            ab_action: AbstractAction = self._learning_agent.sample_action(up_state)

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
                    skill=self._selected_skill.name,
                    skill_status=self._selected_skill.status,
                    states=up_state,
                    actions=ab_action,
                    executions="applicable",
                )

            if terminated:
                break
            time.sleep(4)
            # TODO update the learning agent with the success information of the skill/new model
            self._learning_agent.update()

        if logger is not None:
            obs_log = env.get_observation(logger._obs_spec)
            logger.log(
                save_log=True,
                rgb=obs_log["rgb"],
                depth=obs_log["depth"],
                tcp_pose_b=self._scene.tcp_pose,
                gripper=self._scene.gripper_pos,
                parsed_scene=self._scene,
                skill=self._selected_skill.name,
                skill_status=self._selected_skill.status,
                states=up_state,
                actions=ab_action,
                executions="applicable",
            )
