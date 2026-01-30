"""dummy_task_policy.py.

The high level controller class for skills

Written by Will Solow & Jeff Jewett, 2026
"""

import numpy as np
import torch

from robot_skills.task_policy import TaskPolicy, TaskPolicyCfg


class DummyTaskPolicy(TaskPolicy):
    """Dummy Task Policy.

    This task policy repeatedly uses the dummy skill
    """

    def __init__(self, cfg: TaskPolicyCfg) -> None:
        """Initialize the skill controller.

        Args:
            cfg: The configuration

        """
        super().__init__(cfg)

    def set_skills_and_params(self, high_level_obs: torch.Tensor) -> None:
        """Set the next skill for the robot to execute.

        Choose randomly from available skills.

        Input:
            high_level_obs: Torch Tensor of shape (num_envs, observation dimension)

        Returns:
            skill dictionary: dict of length (num_envs,)
            params: Torch Tensor of shape (num_envs, max_skill_parameter_dimension)

        """
        self.current_skills = np.random.choice(self.avail_skills, self.num_envs, replace=True)
        self.current_skills_params = torch.rand(size=(self.num_envs, 3), device=self.device)

        self.skill_controller.reset(self.current_skills, self.current_skills_params)

        print(f"[INFO][DummyTaskPolicy] Set current skills: {self.current_skills}")
