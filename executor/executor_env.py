"""
executor.py

Main executor class to be run in parallel to the task/policy chain

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from skills.task_policy.task_policy import TaskPolicy
from skills.skill_controller.skill_controller import SkillController
from env.skill_env_wrapper import SkillEnvWrapper

class SkillExecutor:

    env: SkillEnvWrapper
    task_policy: TaskPolicy
    skill_controller: SkillController

    def __init__(self, cfg, env: SkillEnvWrapper) -> None:

        self.cfg = cfg
        self.env = env

        self.num_envs = self.env.num_envs
        self.device = self.env.device

    def execute(self) -> None:
        """
        Execute a run of the environment
        """

        self.task_policy.reset()
        obs, _ = self.env.reset()

        dones = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        while not dones.all():

            skills, params = self.task_policy.get_skills_and_params(obs)
            self.skill_controller.reset(skills=skills, params=params)

            while not self.skill_controller.is_done:
                
                action = self.skill_controller.step(obs)
                obs, rew, term, trunc, info = self.env.step(action)

            dones = torch.logical_or(term, trunc)
        
        return 