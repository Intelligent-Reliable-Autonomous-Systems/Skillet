"""isaac_env_wrapper.py.

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from .skill_env_wrapper import SkillEnvWrapper


class IsaacEnvWrapper(SkillEnvWrapper):
    """Wrapper for IsaacLab Environments.

    This assumes that the environment is either a DirectRLEnv or ManagerBasedRLEnv.
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv) -> None:
        """Initialize the environment.

        Args:
            env: IsaacLab Gymnasium environment

        Returns:
            None

        """
        super().__init__(env)

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            None

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        obs_dict, info = self.env.reset()
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=1)

        return obs, info

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step through the environment.

        Args:
            action: The action tensor of shape (N, num_actions)

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        obs_dict, reward, term, trunc, info = self.env.step(action)
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=1)

        return obs, reward, term, trunc, info
