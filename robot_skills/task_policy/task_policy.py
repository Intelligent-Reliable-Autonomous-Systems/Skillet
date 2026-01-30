"""task_policy.py.

The high level controller class for skills

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

import numpy as np
import torch

from robot_skills.skill_controller import SkillController
from robot_skills.task_policy import TaskPolicyCfg


class TaskPolicy:
    """Superclass for the task policy.

    This class executes each available skill in paralell with the number of environments executing that skill
    """

    skill_controller: SkillController
    current_skills: np.ndarray | None
    current_skills_params: torch.Tensor | None

    def __init__(self, cfg: TaskPolicyCfg) -> None:
        """Initialize the skill controller.

        Args:
            cfg: The configuration

        """
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        print(f"[INFO][TaskPolicy] Creating Skill Controller `{cfg.task_policy_name}`")
        self.skill_controller = SkillController(cfg)

    def reset(self) -> None:
        """Reset the current skills."""
        self.current_skills = None
        self.current_skills_params = None
        print("[INFO][TaskPolicy] Resetting the task policy")

    def low_level_step(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the next action to execute from the skill controller.

        Input:
            obs: Torch Tensor of shape (num_envs, observation dimension)

        Returns:
            torch tensor of shape (N, num_joints)

        """
        return self.skill_controller.step(obs)

    @abstractmethod
    def set_skills_and_params(self, high_level_obs: torch.Tensor) -> None:
        """Get the next skill for the robot to execute.

        Input:
            high_level_obs: Torch Tensor of shape (num_envs, high_level_obs_dim)

        Returns:
            skill dictionary: dict of length (num_envs,)
            params: Torch Tensor of shape (num_envs, max_skill_parameter_dimension)

        """
        raise NotImplementedError(f"Please implement the 'set_skills_and_params' method for {self.__class__.__name__}.")

    @property
    def avail_skills(self) -> list[str]:
        """Return the list of available skills."""
        return self.skill_controller.skills_names

    @property
    def is_skills_done(self) -> bool:
        """Returns if all the skills have completed"""
        return self.skill_controller.is_done.all().item()
