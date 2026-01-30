"""skill_env_wrapper.py.

A wrapper around IsaacSim/ROS2/etc environments to handle the passing
of actions to the underlying simulation/hardware

Written by Will Solow and Jeff Jewett, 2026
"""

from abc import abstractmethod

import gymnasium as gym
import torch


class SkillEnvWrapper:
    """The superclass for wrapping IsaacLab/ROS2 environments.

    This assumes that every environment is a Gym environment and provides the correct interfaces accordingly.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the environment.

        Args:
            env: Gymnasium environment of either a ROS2/IsaacLab env.

        Returns:
            None

        """
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.device = self.env.unwrapped.device

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            None

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        raise NotImplementedError(f"Please implement the 'reset' method for {self.__class__.__name__}.")

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step through the environment.

        Args:
            action: The action tensor of shape (N, num_actions)

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        raise NotImplementedError(f"Please implement the 'step' method for {self.__class__.__name__}.")

    def close(self) -> None:
        """Cleanup for the environment."""
        self.env.close()
