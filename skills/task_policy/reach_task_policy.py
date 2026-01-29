"""
task_policy.py

The high level controller class for skills 

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from ..skill_controller.skill_controller import SkillController
from .task_policy import TaskPolicy

class ReachTaskPolicy(TaskPolicy):


    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def reset(self) -> None:
        """
        Reset the low level policy
        """
        pass

    def get_skills_and_params(self, obs: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """
        Get the next skill for the robot to execute

        Input: 
            obs: Torch Tensor of shape (num_envs, observation dimension)

        Returns:
            skill dictionary: dict of length (num_envs,)
            params: Torch Tensor of shape (num_envs, max_skill_parameter_dimension)
        """
        
        num_envs = obs.shape[0]

        return ["ReachXYZ"] * num_envs, torch.zeros((num_envs,3),device=obs.device)





