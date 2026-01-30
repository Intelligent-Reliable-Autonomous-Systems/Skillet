"""ros2_rl_env.py.

Main ROS2 RL Env Runner

Written by Will Solow, 2026

"""

from typing import Any, Sequence, SupportsFloat
from abc import abstractmethod

import gymnasium as gym
import torch

from .ros2_rl_env_cfg import ROS2RLEnvCfg
from cfg import configure_seed


class ROS2RLEnv(gym.Env):
    """The superclass for the ROS2 workflow to design environments.

    This class implements the core functionality for reinforcement learning (RL)
    environments. However, it is designed to interface with ROS2 and thus calls the corresponding
    action servers, etc.

    """

    def __init__(self, cfg: ROS2RLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """

        self.cfg = cfg
        self.render_mode = render_mode

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        # return observations
        return self._get_observations(), self.extras

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:  # type: ignore
        """Execute one time-step of the ROS2 robot.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Convert the action to NumPy
        2. Publish the action to the robot
        3. Compute the rewards
        4. Get the observations
        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.cpu().numpy()

        self._publish_action_to_robot(action)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for torch and other libraries
        return configure_seed(seed)

    def render(self) -> None:
        """Run rendering by visualizing with RViz."""
        pass

    def close(self):
        """Cleanup for the environment."""
        pass

    """
    Helper functions.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

    """
    Implementation-specific functions.
    """

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self) -> torch.Tensor:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment.
        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")
