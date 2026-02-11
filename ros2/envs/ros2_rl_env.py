"""ros2_rl_env.py.

Main ROS2 RL Env Runner

Written by Will Solow, 2026

"""

import math
import time
from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from roslibpy import Ros

from ros2.envs.utils import configure_seed

from .ros2_rl_env_cfg import ROS2RLEnvCfg


class ROS2RLEnv(gym.Env):
    """The superclass for the ROS2 workflow to design environments.

    This class implements the core functionality for reinforcement learning (RL)
    environments. However, it is designed to interface with ROS2 and thus calls the corresponding
    action servers, etc.

    """

    _current_joint_positions: np.ndarray
    _current_joint_velocities: np.ndarray
    _robot_links: list[str]
    _robot_joints: list[str]
    _current_robot_body_pose_w: np.ndarray
    _current_robot_root_pose_w = np.ndarray
    _current_jacobians: np.ndarray
    _current_upper_joint_limits = np.ndarray
    _current_lower_joint_limits = np.ndarray

    def __init__(self, cfg: ROS2RLEnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment
            ros: roslibpy object
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
            kwargs: Additoinal arguments

        """
        self.cfg = cfg
        self.ros = ros
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        self.render_mode = render_mode

        self._sim_step_counter = 0
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = 0
        self.reset_terminated = False
        self.reset_time_outs = False
        self.reset_buf = False
        # allocate dictionary to store metrics
        self.extras: dict[str, Any] = {}

        # setup the action and observation spaces for Gym

        print("[INFO][ROS2RLEnv] Completed Environment Setup")

    def __del__(self) -> None:
        """Cleanup for the environment."""
        self.close()

    """
    Properties.
    """

    @property
    def _joint_positions(self) -> np.ndarray:
        """Return current joint positions."""
        return self._current_joint_positions

    @property
    def _joint_velocities(self) -> np.ndarray:
        """Return current joint_velocities."""
        return self._current_joint_positions

    @property
    def _robot_body_pose_w(self) -> np.ndarray:
        """Return the body pose information in XYZ + Quaternion."""
        return self._current_robot_body_pose_w

    @property
    def _robot_root_pose_w(self) -> np.ndarray:
        """Return the body pose information in XYZ + Quaternion."""
        return self._current_robot_root_pose_w

    @property
    def _jacobians(self) -> np.ndarray:
        """Return the jacobian frame transforms of the robot."""
        return self._current_jacobians

    @property
    def _robot_lower_joint_limits(self) -> np.ndarray:
        """Return the lower limits of the robot joints."""
        return self._current_lower_joint_limits

    @property
    def _robot_upper_joint_limits(self) -> np.ndarray:
        """Return the upper limits of the robot joints."""
        return self._current_upper_joint_limits

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which ROS2 is publishing.
        """
        return self.cfg.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        # return float(self.cfg.dt * self.cfg.decimation)
        return 4.0

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> float:
        """The maximum episode length in steps adjusted from s."""
        return float(math.ceil(self.max_episode_length_s / (self.cfg.dt * self.cfg.decimation)))

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations happen during reset() which are not repeated

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
        self._reset_idx()

        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # return observations
        return self._get_observations(), self.extras

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, bool, bool, dict[str, Any]]:  # type: ignore
        """Execute one time-step of the ROS2 robot.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre process the action and store it
        2. Publish the action to the robot
        3. Compute the rewards
        4. Get the observations
        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.

        """
        # Pre process the robot action
        joint_pos = self._pre_process_action(action)

        # Send the robot action to hardware
        self._publish_action_to_robot(joint_pos, duration=self.step_dt)
        time.sleep(self.step_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        if self.reset_buf:
            self._reset_idx()

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

    def close(self) -> None:
        """Cleanup for the environment."""
        pass

    """
    Helper functions.
    """

    def _reset_idx(self) -> None:
        """Reset environments based on specified indices."""
        # reset the episode length buffer
        self.episode_length_buf = 0

    def _find_link_idx(self, link: str) -> int:
        """Find the link index in the robot urdf.

        Args:
            link: string of link name

        Returns:
            int of index

        """
        return self._robot_links.index(link)

    def _find_joint_idx(self, joint: str) -> int:
        """Find the joint index in the robot urdf.

        Args:
            joint: string of joint name

        Returns:
            int of index

        """
        return self._robot_joints.index(joint)

    """
    Implementation-specific functions.
    """

    @abstractmethod
    def _pre_process_action(self, actions: np.ndarray) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).

        Returns:
            The joint positions to publish to the robot

        """
        raise NotImplementedError(f"Please implement the '_pre_process_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _publish_action_to_robot(self, actions: np.ndarray, duration: float = 1) -> None:
        """Publish action to robot controller.

        This function is responsible publishing joint positions and velocities to
        the correct robot target. These positions are set by _pre_process_action()

        Args:
            actions: joint positions to publish to the robot
            duration: duration of trajectory

        """
        raise NotImplementedError(
            f"Please implement the '_publish_action_to_robot' method for {self.__class__.__name__}."
        )

    @abstractmethod
    def _get_observations(self) -> dict[str, np.ndarray]:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment in the form {positions: [], velocities: []}.

        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self) -> np.ndarray:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).

        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[bool, bool]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Assumed to not be batched

        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")

    """
    Inverse kinematics related helper functions
    """
