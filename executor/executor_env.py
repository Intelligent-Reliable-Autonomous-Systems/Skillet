"""executor.py.

Main executor class to be run in parallel to the task/policy chain

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from env.skill_env_wrapper import SkillEnvWrapper
from skills.skill_controller.skill_controller import SkillController
from skills.task_policy.reach_task_policy import ReachTaskPolicy
from skills.task_policy.task_policy import TaskPolicy


class SkillExecutor:
    """The main class for executing skills.

    This assumes access to a gymansium environment to be executed in paralell
    """

    env: SkillEnvWrapper
    task_policy: TaskPolicy
    skill_controller: SkillController

    def __init__(self, cfg: dict, env: SkillEnvWrapper) -> None:
        """Initialize the environment.

        Args:
            cfg: Config dictionary
            env: A SkillEnvWrapper environment of a wrapped IsaacLab/ROS2 environment

        Returns:
            None

        """
        self.cfg = cfg
        self.env = env

        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.skill_controller = SkillController(cfg)
        self.task_policy = ReachTaskPolicy(cfg)

    def execute(self) -> None:
        """Execute a run of the environment.

        Args:
            None

        Returns:
            None

        """
        self.task_policy.reset()
        _, _ = self.env.reset()

        dones = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        while not dones.all():
            # skills, params = self.task_policy.get_skills_and_params(obs)
            # self.skill_controller.reset(skills=skills, params=params)
            i = 0
            # while not self.skill_controller.is_done:
            while True:
                # action = self.skill_controller.step(obs)
                action = 2 * torch.rand(self.env.env.action_space.shape, device=self.env.env.unwrapped.device) - 1
                _, _, term, trunc, _ = self.env.step(action)
                if i > 10:
                    break
                i += 1
            dones = torch.logical_or(term, trunc)

        return
