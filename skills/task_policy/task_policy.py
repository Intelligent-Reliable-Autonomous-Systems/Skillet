"""task_policy.py.

The high level controller class for skills

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

import torch

from skills.skill_controller.skill_controller import SkillController


class TaskPolicy:
    """Superclass for the task policy.

    This class executes each available skill in paralell with the number of environments executing that skill
    """

    skill_controller: SkillController
    avail_skills: dict

    def __init__(self, cfg: dict) -> None:
        """Initialize the skill controller.

        Args:
            cfg: The configuration

        """
        self.cfg = cfg

    @abstractmethod
    def reset(self) -> None:
        """Reset the low level policy."""
        raise NotImplementedError(f"Please implement the 'reset' method for {self.__class__.__name__}.")

    @abstractmethod
    def get_skills_and_params(self, obs: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Get the next skill for the robot to execute.

        Input:
            obs: Torch Tensor of shape (num_envs, observation dimension)

        Returns:
            skill dictionary: dict of length (num_envs,)
            params: Torch Tensor of shape (num_envs, max_skill_parameter_dimension)

        """
        raise NotImplementedError(f"Please implement the 'get_skills_and_params' method for {self.__class__.__name__}.")
