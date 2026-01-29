"""rl_policy.py.

The low level controller class for a RL controller.

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from .low_level_policy import LowLevelPolicy


class RLPolicy(LowLevelPolicy):
    """Low level policy based on an RL policy.

    This assumes that the joint positions are given by an RL policy
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize the RL policy.

        Args:
            cfg: The configuration

        """
        super().__init__(cfg)

    def reset(self) -> None:
        """Reset the low level policy.

        Resets the RL policy.
        """
        pass

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the next low level action for the robot.

        This is on the forward pass of the RL policy

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tensor of shape (N, num_joints)

        """
        print(obs)
        return torch.tensor(0)
