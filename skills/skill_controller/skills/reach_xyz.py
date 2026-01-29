"""reach_xyz.py.

The reach XYZ skill, moving the end effector to a location in XYZ space relative
to the robot base

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from .base_skill import BaseSkill


class ReachXYZ(BaseSkill):
    """The Reach XYZ skill with a Differential IK controller.

    Moves the end effector to a location in XYZ space while preserving orientation
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize Reach XYZ Skill.

        Defines the low level controller

        Args:
            cfg: A configuration dictionary

        """
        super().__init__(cfg)

    def reset(self) -> None:
        """Reset the skill.

        Resets the low level skill policy
        """
        pass

    def is_valid_initial_state(self) -> torch.Tensor:
        """Test if the current state is a valid initial state for the skill."""
        return torch.tensor(0)

    def is_valid_terminal_state(self) -> torch.Tensor:
        """Test if the current state is a valid terminal state."""
        return torch.tensor(0)

    def is_success(self) -> torch.Tensor:
        """Test if the skill successfully completed."""
        return torch.tensor(0)
