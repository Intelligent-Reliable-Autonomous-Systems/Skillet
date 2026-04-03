"""A Task and Motion Planner executor for running an agent in an environment."""
from typing import Any

from skillet.core.env import Environment
from skillet.core.policy import Unparameterized
from skillet.core.skill import SingleSkill, SkillStatus
from skillet.scene.abstract.abstract_model import AbstractModel
from skillet.scene.base import Scene

# class TAMPAgent(Agent):

#     def execute(env: Environment) -> None:
#         """A Task and Motion Planner executor for running an agent in an environment."""

#         # Get the initial observation

#         # 1. Do high level planning -> Get a sequence of skills to execute
#         # 2. Do parameter sampling -> Get a set of parameters for each skill
#         obs = env.get_observation(agent.task_observation_spec)
#         skill_sequence, skill_params: list[Skill], list[SkillParams] = agent.get_skill_sequence(obs, task)
#         skill_params: list[SkillParams] = agent.get_skill_params(skill_sequence, obs, task)

#         # 3. Execute the skills:
#         for skill, params in zip(skill_sequence, skill_params):
#             obs = env.get_observation(skill.observation_spec)
#             #   3.2. Check if skill can be initiated, else fail
#             if not skill.can_initiate(obs):
#                 return SkillStatus.FAILED
#             skill.initiate(obs, params)
#             #   3.3. While not terminated, query the skill controller to get the next action and take a step in the environment
#             action_or_status = skill.get_action(obs)
#             while not isinstance(action_or_status, SkillStatus):
#                 _, reward, term, trunc, info = env.step(action)
#                 obs = env.get_observation(skill.observation_spec)
#                 action_or_status = skill.get_action(obs)
#             #   3.4. If skill reports success, move to next skill, else fail
#             if action_or_status != SkillStatus.SUCCESS:
#                 return SkillStatus.FAILED

class PlanningAgent:
    """A Task-Planning agent that plans a sequence of skills to execute to complete a task."""

    def __init__(self,
            scene: Scene,
            abstract_model: AbstractModel,
            action_to_skill_map: dict[str, SingleSkill[Any, Any, Unparameterized]]) -> None:
        """Initialize the planning agent.

        Args:
            scene: The scene to execute the skills in.
            abstract_model: The abstract model of the scene.
            action_to_skill_map: A map of actions to skills.

        """
        self.scene = scene
        self.abstract_model = abstract_model
        self.action_to_skill_map = action_to_skill_map

    def execute(self, env: Environment[Any, Any], task: str) -> None:
        """Execute the policy over the options configured.

        Args:
            env: The environment to execute the policy over.
            task: The task to execute.

        """
        # Initialize our scene model
        self.scene.reset(task)
        self.scene.perceive()

        # Get the current symbolic state
        self.abstract_model.initialize(self.scene, task)
        actions = self.abstract_model.plan()

        terminated = False
        cum_reward = 0.0

        for action in actions:
            selected_skill = self.action_to_skill_map[action[0]]
            args = action[1:]

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

            # Update the scene state
            self.scene.perceive()

            # Check if the skill was successful
            if selected_skill.status != SkillStatus.SUCCESS:
                break
            if terminated:
                break
