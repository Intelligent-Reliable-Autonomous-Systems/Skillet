"""skill_controller.py.

The class for controlling low level skills

Written by Will Solow and Jeff Jewett, 2026
"""

import torch

from .skills.base_skill import BaseSkill
from .skills.reach_xyz import ReachXYZ


class SkillController:
    """Base class for controlling executing of skills.

    This class executes each available skill in paralell with the number of environments executing that skill
    """

    current_skills: list[BaseSkill]

    def __init__(self, cfg: dict) -> None:
        """Initialize the skill controller.

        Args:
            cfg: The configuration

        """
        self.cfg = cfg

    def reset(self, skills: list[str], params: torch.Tensor) -> None:
        """Reset the skill controller.

        Args:
            skills: A list of skills
            params: A parameterization for each of the skills

        """
        self.current_skill_list = skills
        self.current_skills = [ReachXYZ({})]
        self.params = params

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the next low level action for the robot.

        Step through each of the skills for the skill controller

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tensor of shape (N, num_joints)

        """
        self.current_skills.get_action(obs)

        return torch.tensor(0)

    @property
    def is_done(self) -> torch.Tensor:
        """Check if the skill controller is done.

        Returns:
            A tensor of dones for each skill that is running.

        """
        return self.current_skills.is_valid_terminal_state()
