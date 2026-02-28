"""A module specifying the abstract interface for the Isaac Lab DirectRLEnv.

Isaac Lab has two environment APIs: DirectRLEnv and ManagerBasedRLEnv.
This module specifies the abstract interface for the DirectRLEnv, without any implementation details.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import gymnasium as gym
import torch

from skillet.core.spaces import BatchedObservation


class DirectRlInterface(ABC, gym.vector.VectorEnv):
    """An abstract interface for the Isaac Lab DirectRLEnv.

    This interface is used to specify the abstract interface for the DirectRLEnv, without any implementation details.
    It should behave as a vectorized environment

    At minimum, the environment should implement the following properties and methods:
    - cfg
    - num_envs
    - device
    - max_episode_length
    - episode_length_buf
    - _get_observations()
    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""

    @property
    @abstractmethod
    def cfg(self) -> dict | object:
        """Configuration object."""
        raise NotImplementedError(f"Please implement the 'cfg' property for {self.__class__.__name__}.")

    @abstractmethod
    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        raise NotImplementedError(f"Please implement the 'num_envs' property for {self.__class__.__name__}.")

    @abstractmethod
    @property
    def device(self) -> torch.device | str:
        """The device on which the environment is running."""
        raise NotImplementedError(f"Please implement the 'device' property for {self.__class__.__name__}.")

    @abstractmethod
    @property
    def max_episode_length(self) -> int:
        """The maximum episode length in steps adjusted from s."""
        raise NotImplementedError(f"Please implement the 'max_episode_length' property for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def episode_length_buf(self) -> torch.Tensor:
        """Buffer for current episode lengths."""
        raise NotImplementedError(f"Please implement the 'episode_length_buf' property for {self.__class__.__name__}.")

    @property
    def common_step_counter(self) -> int:
        """Step counter common to all environments."""
        raise NotImplementedError(f"Please implement the 'common_step_counter' property for {self.__class__.__name__}.")

    @property
    def reset_terminated(self) -> torch.Tensor:
        """Buffer for terminated resets."""
        raise NotImplementedError(f"Please implement the 'reset_terminated' property for {self.__class__.__name__}.")

    @property
    def reset_time_outs(self) -> torch.Tensor:
        """Buffer for time out resets."""
        raise NotImplementedError(f"Please implement the 'reset_time_outs' property for {self.__class__.__name__}.")

    @property
    def reset_buf(self) -> torch.Tensor:
        """Buffer for resets."""
        raise NotImplementedError(f"Please implement the 'reset_buf' property for {self.__class__.__name__}.")

    @property
    def extras(self) -> dict[str, torch.Tensor]:
        """Dictionary for extra information."""
        raise NotImplementedError(f"Please implement the 'extras' property for {self.__class__.__name__}.")

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        raise NotImplementedError(f"Please implement the 'physics_dt' property for {self.__class__.__name__}.")

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        raise NotImplementedError(f"Please implement the 'step_dt' property for {self.__class__.__name__}.")

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        raise NotImplementedError(f"Please implement the 'max_episode_length_s' property for {self.__class__.__name__}.")

# - event_manager
# - has_debug_vis_implementation
# - scene
# - sim
# - viewport_camera_controller
    @property
    def scene(self) -> None:
        raise AttributeError(f"The 'scene' attribute is not available for {self.__class__.__name__}.")

    @property
    def sim(self) -> None:
        raise AttributeError(f"The 'sim' attribute is not available for {self.__class__.__name__}.")

    @property
    def viewport_camera_controller(self) -> None:
        raise AttributeError(f"The 'viewport_camera_controller' attribute is not available for {self.__class__.__name__}.")

    @property
    def event_manager(self) -> None:
        raise AttributeError(f"The 'event_manager' attribute is not available for {self.__class__.__name__}.")

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the environment has a debug visualization implementation."""
        raise AttributeError(f"The 'has_debug_vis_implementation' attribute is not available for {self.__class__.__name__}.")

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset

        """
        raise NotImplementedError(f"Please implement the '_reset_idx' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self) -> BatchedObservation:
        """Compute and return the observations for the environment.

        Returns:
            The batched observations for the environment.

        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    def _get_states(self) -> BatchedObservation | None:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.

        """
        raise NotImplementedError(f"Please implement the '_get_states' method for {self.__class__.__name__}.")

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).

        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).

        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")
