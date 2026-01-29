"""
task_policy.py

The high level controller class for skills

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

import torch

from ..skill_controller.skill_controller import SkillController


class TaskPolicy:

    skill_controller: SkillController
    avail_skills: dict

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def reset(self) -> None:
        """
        Reset the low level policy
        """
        pass

    @abstractmethod
    def get_skills_and_params(self, obs: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """
        Get the next skill for the robot to execute

        Input:
            obs: Torch Tensor of shape (num_envs, observation dimension)

        Returns:
            skill dictionary: dict of length (num_envs,)
            params: Torch Tensor of shape (num_envs, max_skill_parameter_dimension)
        """
        pass
