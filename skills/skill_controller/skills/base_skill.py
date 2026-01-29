"""base_skill.py.

The abstract class for skills (options) which define their starting/ending conditions,
low level controller, and success condition

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

import torch

from skills.low_level_policy.low_level_policy import LowLevelPolicy


class BaseSkill:
    """Base class for individual skills.

    Provides a framework for the execution of an individual skill.
    """

    low_level_policy: LowLevelPolicy

    def __init__(self, cfg: dict) -> None:
        """Initialize Base skill.

        Defines the low level controller

        Args:
            cfg: A configuration dictionary

        """
        self.cfg = cfg

    @abstractmethod
    def reset(self) -> None:
        """Reset the skill.

        Resets the low level skill policy
        """
        raise NotImplementedError(f"Please implement the 'reset' method for {self.__class__.__name__}.")

    def step(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Step through the skill.

        Step through each of the skills for the skill controller

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tuple with a tensor of shape (N, num_joints) and a tensor of shape (N,) if the skill is done

        """
        action = self.low_level_policy.get_action(obs)

        return action, torch.tensor(0)

    @abstractmethod
    def is_valid_initial_state(self) -> torch.Tensor:
        """Test if the current state is a valid initial state for the skill."""
        raise NotImplementedError(
            f"Please implement the 'is_valid_initial_state' method for {self.__class__.__name__}."
        )

    @abstractmethod
    def is_valid_terminal_state(self) -> torch.Tensor:
        """Test if the current state is a valid terminal state."""
        raise NotImplementedError(
            f"Please implement the 'is_valid_terminal_state' method for {self.__class__.__name__}."
        )

    @abstractmethod
    def is_success(self) -> torch.Tensor:
        """Test if the skill successfully completed."""
        raise NotImplementedError(f"Please implement the 'is_success' method for {self.__class__.__name__}.")
