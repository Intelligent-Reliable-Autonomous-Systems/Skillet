"""A Task and Motion Planner executor for running an agent in an environment."""

import pickle
import time
from typing import Any

from skillet.agents import SkilletModerator
from skillet.agents.base_agent import Agent
from skillet.core.env import Environment
from skillet.core.policy import Unparameterized
from skillet.core.skill import SingleSkill, SkillStatusCodes
from skillet.logging import SkilletDataLogger
from skillet.planning import AbstractModel
from skillet.planning.abstract import AbstractAction
from skillet.planning.abstract.up_utils import sample_action_from_state
from skillet.scene.base import Scene
from skillet.scene import Spill


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
        self._abstract_model = abstract_model
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
        self._abstract_model.initialize(self._scene, task)

        abstract_state = self._abstract_model.get_abstract_state()
        self._result, self._plan, up_actions = self._abstract_model.plan(abstract_state=abstract_state)
        up_state = self._abstract_model.reset_up_problem_state()
        terminated = False
        if self._plan is None:
            print("[WARNING][TAMP] Failed to find plan.")
            return
        print(self._plan)
        for ab_action, up_action in zip(self._plan.actions, up_actions):
            up_state = self._abstract_model.reset_up_problem_state()
            self._selected_skill = self.action_to_skill_map[ab_action.action]
            args = self._scene.resolve_names_to_ids(ab_action.parameters)
            terminated, status = self._moderator.run_skill(env, self._selected_skill, args)

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
                    actions=up_action,
                    executions="applicable",
                )

            if terminated:
                break
            time.sleep(1)
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
                actions=up_action,
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
        self._abstract_model = abstract_model
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
        self._abstract_model.initialize(self._scene, task)

        terminated = False

        for i in range(num_actions):
            up_state = self._abstract_model.reset_up_problem_state()
            # ab_action, up_action = self._abstract_model.get_random_action(up_state)
            ab_action, up_action = sample_action_from_state(self._abstract_model._problem, up_state)
            self._selected_skill = self.action_to_skill_map[ab_action.action]
            args = self._scene.resolve_names_to_ids(ab_action.parameters)

            terminated, status = self._moderator.run_skill(env, self._selected_skill, args)

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
                    actions=up_action,
                    executions="applicable",
                )

            if terminated:
                break
            time.sleep(3)
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
                actions=up_action,
                executions="applicable",
            )


class ActiveLearningAgent(Agent):
    """A Task-Planning agent that actively selects an action."""

    def __init__(
        self,
        scene: Scene,
        abstract_model: AbstractModel,
        action_to_skill_map: dict[str, SingleSkill[Any, Any, Unparameterized]],
        learning_agent: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the planning agent.

        Args:
            scene: The scene to execute the skills in.
            abstract_model: The abstract model of the scene.
            action_to_skill_map: A map of actions to skills.

        """
        super().__init__()

        self._scene = scene
        self._abstract_model = abstract_model
        self.action_to_skill_map = action_to_skill_map
        self._moderator = SkilletModerator()
        self._learning_agent = learning_agent

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
        self._abstract_model.initialize(self._scene, task)

        terminated = False
        up_state = self._abstract_model.reset_up_problem_state()
        up_objects = self._abstract_model._problem.all_objects
        self._learning_agent.reset_problem(self._abstract_model.problem)
        skills_sampled = 0
        skills_failed = 0
        was_paused = True
        trace_file = None
        while True:
            if self._moderator.is_paused:
                up_state = self._abstract_model.reset_up_problem_state()
                up_objects = self._abstract_model._problem.all_objects
                was_paused = True
                for obj in self._scene.objects:
                    if isinstance(obj, Spill) and "marker" in obj.name:
                        obj.wiped = False

                time.sleep(0.1)
                continue
            up_action = self._learning_agent.get_action(up_state, up_objects)
            up_objects = self._abstract_model._problem.all_objects

            if up_action is None:
                print("[WARN][ACTIVE] Invalid action selected/unable to find valid caction")
                time.sleep(0.1)
                continue
            ab_action = AbstractAction(
                action=up_action.action.name, parameters=[p.object().name for p in up_action.actual_parameters]
            )

            self._selected_skill = self.action_to_skill_map[ab_action.action]
            args = self._scene.resolve_names_to_ids(ab_action.parameters)
            terminated, status = self._moderator.run_skill(env, self._selected_skill, args)

            execution = "applicable" if status == SkillStatusCodes.SUCCESS else "inapplicable"
            # Logging and learning
            time.sleep(1.5)  # To let perception update
            if logger is not None:
                obs_log = env.get_observation(logger._obs_spec)
                logger.log(
                    log_dir=self._learning_agent.dataset.experiment_dir,
                    save_log=True,
                    rgb=obs_log["rgb"],
                    depth=obs_log["depth"],
                    tcp_pose_b=self._scene.tcp_pose,
                    gripper=self._scene.gripper_pos,
                    parsed_scene=self._scene,
                    skill=self._selected_skill.name,
                    skill_status=self._selected_skill.status,
                    states=up_state,
                    actions=up_action,
                    executions=execution,
                )
            next_up_state = self._abstract_model.reset_up_problem_state()

            if was_paused:
                trace_file = self._learning_agent.dataset._traces_dir / f"{self._moderator._exp_count}.trace"
                self._learning_agent.dataset._plan_parser.initialize_trace_file(
                    trace_file,
                    up_state,
                    up_objects,
                )
                was_paused = False
            self._learning_agent.dataset._plan_parser.append_trace_step(trace_file, up_action, next_up_state, execution)

            self._learning_agent.reset_problem(self._abstract_model.problem)

            self._learning_agent.learn_step(up_state, up_objects, up_action, next_up_state, execution)
            # with open(f"{self._learning_agent.dataset.experiment_dir}/_agent.pkl", "wb") as f:
            #    pickle.dump(self._learning_agent, f)

            if terminated:
                break

            up_state = next_up_state
            skills_sampled += 1
            if execution == "inapplicable":
                skills_failed += 1

            print(
                f"[INFO] Sampled Skills {skills_sampled} / {self._learning_agent.dataset.max_steps}. Skills Failed: {skills_failed}."
            )

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
                actions=up_action,
                executions=execution,
            )
