"""A module specifying the abstract interface for the Isaac Lab DirectRLEnv.

Isaac Lab has two environment APIs: DirectRLEnv and ManagerBasedRLEnv.
This module specifies the abstract interface for the DirectRLEnv, without any implementation details.

Many properties are marked as deprecated, because they are not required for skillet.
If we need to use them, we will need to implement them (e.g. in ROS2SkilletEnv).
"""

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from typing_extensions import deprecated

from skillet.core.spaces import BatchedObservation
from skillet.envs.compatibility.gymnasium import GymVectorInterface


class IsaacLabInterface(GymVectorInterface):
    """An abstract interface for the Isaac Lab environment.

    The properties and methods are the intersection of the DirectRLEnv and ManagerBasedRLEnv interfaces.

    This interface is used to specify the abstract interface for the Isaac Lab environment, without any implementation.
    It should behave as a vectorized environment

    At minimum, the environment should implement the following properties and methods:
    - cfg
    - device
    - max_episode_length
    - num_envs
    """

    @property
    def unwrapped(self) -> "IsaacLabInterface":
        """Return the base environment."""
        ...

    @property
    def cfg(self) -> dict | object:
        """Configuration object."""
        ...

    @property
    def device(self) -> torch.device | str:
        """The device on which the environment is running."""
        ...

    @property
    @deprecated("event_manager is not currently supported across all environments.")
    def event_manager(self) -> None:
        """The Isaac Lab event manager object."""
        ...

    @property
    @deprecated("extras is not currently supported across all environments.")
    def extras(self) -> dict[str, torch.Tensor]:
        """Dictionary for extra information."""
        ...

    @property
    def max_episode_length(self) -> int:
        """The maximum episode length in steps adjusted from s."""
        ...

    @property
    @deprecated("max_episode_length_s is not currently supported across all environments.")
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        ...

    # @property
    # def num_envs(self) -> int:
    #     """The number of instances of the environment that are running."""
    #     ...

    @property
    @deprecated("physics_dt is not currently supported across all environments.")
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        ...

    @property
    @deprecated("scene is not currently supported across all environments.")
    def scene(self) -> None:
        """The Isaac Lab scene object."""
        ...

    @property
    @deprecated("sim is not currently supported across all environments.")
    def sim(self) -> None:
        """The Isaac Lab sim object."""
        ...

    @property
    @deprecated("step_dt is not currently supported across all environments.")
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        ...

    @property
    @deprecated("viewport_camera_controller is not currently supported across all environments.")
    def viewport_camera_controller(self) -> None:
        """The Isaac Lab viewport camera controller object."""
        ...


class DirectRlInterface(IsaacLabInterface):
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

    @property
    def unwrapped(self) -> "DirectRlInterface":
        """Return the base environment."""
        ...

    @property
    @deprecated("common_step_counter is not currently supported across all environments.")
    def common_step_counter(self) -> int:
        """Step counter common to all environments."""
        ...

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Buffer for current episode lengths."""
        ...

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Set the episode length buffer."""
        ...

    @property
    @deprecated("has_debug_vis_implementation is not currently supported across all environments.")
    def has_debug_vis_implementation(self) -> bool:
        """Whether the environment has a debug visualization implementation."""
        ...

    @property
    @deprecated("reset_buf is not currently supported across all environments.")
    def reset_buf(self) -> torch.Tensor:
        """Buffer for resets."""
        ...

    @property
    @deprecated("reset_terminated is not currently supported across all environments.")
    def reset_terminated(self) -> torch.Tensor:
        """Buffer for terminated resets."""
        ...

    @property
    @deprecated("reset_time_outs is not currently supported across all environments.")
    def reset_time_outs(self) -> torch.Tensor:
        """Buffer for time out resets."""
        ...

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset

        """
        ...

    @deprecated("_get_observations is not currently supported across all environments.")
    def _get_observations(self) -> BatchedObservation:
        """Compute and return the observations for the environment.

        Returns:
            The batched observations for the environment.

        """
        ...

    @deprecated("_get_states is not currently supported across all environments.")
    def _get_states(self) -> BatchedObservation | None:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.

        """
        ...

    @deprecated("_get_rewards is not currently supported across all environments.")
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).

        """
        ...

    @deprecated("_get_dones is not currently supported across all environments.")
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).

        """
        ...


class ManagerBasedRlInterface(IsaacLabInterface):
    """An abstract interface for the Isaac Lab ManagerBasedRLEnv.

    This interface is used to specify the abstract interface for the ManagerBasedRLEnv, without any implementation.
    It should behave as a vectorized environment

    At minimum, the environment should implement the following properties and methods:
    - cfg
    - device
    - max_episode_length
    - num_envs
    """

    @property
    def unwrapped(self) -> "ManagerBasedRlInterface":
        """Return the base environment."""
        ...

    @deprecated("export_IO_descriptors is not currently supported across all environments.")
    def export_IO_descriptors(self, output_dir: str | None = None) -> None:  # noqa: N802
        """Export the IO descriptors for the environment.

        Args:
            output_dir: The directory to export the IO descriptors to.

        """
        ...

    @deprecated("load_managers is not currently supported across all environments.")
    def load_managers(self) -> None:
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        ...

    @property
    @deprecated("obs_buf is not currently supported across all environments.")
    def obs_buf(self) -> dict[str, torch.Tensor]:
        """The observation buffer for the environment."""
        ...

    @property
    @deprecated("observation_manager is not currently supported across all environments.")
    def observation_manager(self) -> Any:  # noqa: ANN401
        """The observation manager for the environment."""
        ...

    @deprecated("reset_to is not currently supported across all environments.")
    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> None:
        """Reset specified environments to provided states.

        This function resets the environments to the provided states. The state is a dictionary
        containing the state of the scene entities. Please refer to :meth:`InteractiveScene.get_state`
        for the format.

        The function is different from the :meth:`reset` function as it resets the environments to specific states,
        instead of using the randomization events for resetting the environments.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.

        """
        ...

    @deprecated("setup_manager_visualizers is not currently supported across all environments.")
    def setup_manager_visualizers(self) -> None:
        """Create live visualizers for manager terms."""
        ...
