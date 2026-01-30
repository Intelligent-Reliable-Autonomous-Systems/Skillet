"""base_skill.py.

The abstract class for skills (options) which define their starting/ending conditions,
low level controller, and success condition

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

import torch

from robot_skills.low_level_policy import LowLevelPolicy, LowLevelPolicyCfg


class BaseSkill:
    """Base class for individual skills.

    Provides a framework for the execution of an individual skill.
    """

    low_level_policy: LowLevelPolicy

    def __init__(self, cfg: LowLevelPolicyCfg, device: str = "cuda") -> None:
        """Initialize Base skill.

        Defines the low level controller

        Args:
            cfg: A configuration dictionary
            device: CUDA device

        """
        self.cfg = cfg
        self.device = device

    def reset(self, env_ids: torch.Tensor, skill_params: torch.Tensor) -> None:
        """Reset the skill."""
        self.env_ids = env_ids
        self.num_envs = env_ids.shape[0]
        self.skill_params = skill_params

        self._state = torch.zeros((self.num_envs,), device=self.device)
        self._curr_calls = 0
        self._dones = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def step(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Step through the skill.

        Step through the skill by querying the low level policy

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tuple with a tensor of shape (N, num_joints) and a tensor of shape (N,) if the skill is done

        """
        self._curr_calls += 1
        return self._get_action(obs), self._is_done()

    @abstractmethod
    def _update_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Update the state of the skill."""
        raise NotImplementedError(f"Please implement the '_update_state' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Update the pose goal of the skill."""
        raise NotImplementedError(f"Please implement the '_get_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _is_done(self) -> torch.Tensor:
        """Return a Tensor of the environment IDs that are done."""
        raise NotImplementedError(f"Please implement the '_is_done' method for {self.__class__.__name__}.")

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
