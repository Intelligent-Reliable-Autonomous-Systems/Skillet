"""low_level_policy.py.

The abstract class for low level control policies.
Subclasses are End Effector Policies and RL Policies

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

import torch


class LowLevelPolicy:
    """Superclass for low level policies.

    This assumes that the low level policy decides the next joint positions based on observations
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize the low level policy.

        Args:
            cfg: The configuration

        """
        self.cfg = cfg

    @abstractmethod
    def reset(self) -> None:
        """Reset the low level policy.

        Perform any resetting actions that need to happen.
        """
        raise NotImplementedError(f"Please implement the 'reset' method for {self.__class__.__name__}.")

    @abstractmethod
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the next low level action for the robot.

        This is on type of controller being used

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tensor of shape (N, num_joints)

        """
        raise NotImplementedError(f"Please implement the 'get_action' method for {self.__class__.__name__}.")
