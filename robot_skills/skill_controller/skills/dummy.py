"""dummy.py.

The dummy skill, giving random robot actions

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from .base_skill import BaseSkill
from robot_skills.low_level_policy import LowLevelPolicyCfg


class Dummy(BaseSkill):
    """The dummy skill with random joint actions.

    Gives random joint position commands
    """

    _max_steps = 10

    def __init__(self, cfg: LowLevelPolicyCfg, device: str = "cuda") -> None:
        """Initialize Reach XYZ Skill.

        Defines the low level controller

        Args:
            cfg: A configuration dictionary
            device: CUDA device

        """
        super().__init__(cfg, device=device)

    def _get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the skill.

        Randomly select joint positions for the agent

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tuple with a tensor of shape (N, num_joints) and a tensor of shape (N,) if the skill is done

        """
        return 2 * torch.rand((obs.shape[0], self.cfg.output_dim), device=self.device) - 1

    def _is_done(self) -> torch.Tensor:
        """Return a Tensor of the environment IDs that are done."""
        return (self._state == 1) | self._curr_calls > self._max_steps
