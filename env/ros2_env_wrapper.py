"""ros2_env_wrapper.py.

A wrapper around ROS2 Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

import gymnasium as gym
import torch

from .skill_env_wrapper import SkillEnvWrapper


class ROS2EnvWrapper(SkillEnvWrapper):
    """Wrapper for ROS2 Environments.

    This assumes that the environment is either a gym.Env and interfaces directly with ROS2.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the environment.

        Args:
            env: IsaacLab Gymnasium environment

        Returns:
            None

        """
        super().__init__(env)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step through the environment.

        Args:
            action: The action tensor of shape (N, num_actions)

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        print(action)

        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), {}
