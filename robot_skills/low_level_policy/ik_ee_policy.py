"""ik_ee_policy.py.

The low level controller class for a Differential IK Controller

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from .low_level_policy import LowLevelPolicy


class IKEEPolicy(LowLevelPolicy):
    """Low level policy based on EE position.

    This assumes that the end effector is controlled by a Differential Inverse Kinematics controller
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize the EE policy.

        Args:
            cfg: The configuration

        """
        super().__init__(cfg)

    def reset(self) -> None:
        """Reset the low level policy.

        Resets the differential IK controller.
        """
        pass

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the next low level action for the robot.

        This is on the inverse kinematics of the robot

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tensor of shape (N, num_joints)

        """
        print(obs)
        return torch.tensor(0)
