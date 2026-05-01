"""Define a gymnasium environment interface for compatibility across different environments."""

from abc import abstractmethod
from typing import Any, Protocol

import gymnasium as gym
import torch
from jaxtyping import Bool, Float

from skillet.core.spaces import ActionSpec, BatchedAction, BatchedObservation, BatchedSpaceValue


class GymVectorInterface(Protocol):
    """An abstract interface for a Gymnasium vectorized environment."""

    observation_space: gym.Space
    action_space: gym.Space
    single_observation_space: gym.Space
    single_action_space: gym.Space

    num_envs: int

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset all parallel environments and return a batch of initial observations and info.

        Args:
            seed: The environment reset seed
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.

        Example:
            >>> import gymnasium as gym
            >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
            >>> observations, infos = envs.reset(seed=42)
            >>> observations
            array([[ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ],
                   [ 0.01522993, -0.04562247, -0.04799704,  0.03392126],
                   [-0.03774345, -0.02418869, -0.00942293,  0.0469184 ]],
                  dtype=float32)
            >>> infos
            {}

        """
        ...

    def step(
        self, actions: BatchedAction, action_spec: ActionSpec
    ) -> tuple[
        BatchedObservation, Float[torch.Tensor, "b"], Bool[torch.Tensor, "b"], Bool[torch.Tensor, "b"], dict[str, Any]
    ]:
        """Take an action for each parallel environment.

        Args:
            actions: Batch of actions with the :attr:`action_space` shape.
            action_spec: Action specification

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)

        Note:
            As the vector environments autoreset for a terminating and truncating sub-environments, this will occur on
            the next step after `terminated or truncated is True`.

        Example:
            >>> import gymnasium as gym
            >>> import numpy as np
            >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
            >>> _ = envs.reset(seed=42)
            >>> actions = np.array([1, 0, 1], dtype=np.int32)
            >>> observations, rewards, terminations, truncations, infos = envs.step(actions)
            >>> observations
            array([[ 0.02727336,  0.18847767,  0.03625453, -0.26141977],
                   [ 0.01431748, -0.24002443, -0.04731862,  0.3110827 ],
                   [-0.03822722,  0.1710671 , -0.00848456, -0.2487226 ]],
                  dtype=float32)
            >>> rewards
            array([1., 1., 1.])
            >>> terminations
            array([False, False, False])
            >>> terminations
            array([False, False, False])
            >>> infos
            {}

        """
        ...

    @property
    def unwrapped(self) -> "GymVectorInterface":
        """Return the base environment."""
        ...


class AsGymVectorEnv(gym.vector.VectorEnv):
    """A wrapper for gym.Env environments that already have a vectorized interface to the gymnasium 1.0 vector API.

    THIS ASSUMES THAT THE ENVIRONMENT WAS DESIGNED AS A VECTOR ENVIRONMENT BEFORE GYMNASIUM 1.0.

    This assumes that the environment is a gym.Env (not a gym.vector.VectorEnv)
    and that it has self.single_observation_space and self.single_action_space attributes.
    """

    def __init__(self, env: gym.Env | GymVectorInterface, num_envs: int | None = None) -> None:
        """Initialize the environment.

        Args:
            env: The gym.Env to wrap. Must have a single observation space and action space.
            num_envs: Optionally, the number of environments to wrap.
                If not provided, it requires env.get_wrapper_attr("num_envs") to be set.

        """
        self.env = env

        def env_attr(env: gym.Env | GymVectorInterface, attr: str) -> Any:
            if hasattr(env, attr):
                return getattr(env, attr)
            if isinstance(env, gym.Env) and env.has_wrapper_attr(attr):
                return env.get_wrapper_attr(attr)
            return None

        self.num_envs = num_envs or env_attr(env, "num_envs")
        if self.num_envs is None:
            raise ValueError("The environment does not have a number of environments .num_envs")
        if not env_attr(env, "single_observation_space") or not env_attr(env, "single_action_space"):
            raise ValueError("The environment does not have a single observation space or action space.")
        self.single_observation_space = env_attr(env, "single_observation_space")
        self.single_action_space = env_attr(env, "single_action_space")
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(  # noqa: D102
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[BatchedSpaceValue, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(  # noqa: D102
        self, actions: BatchedAction, action_spec: ActionSpec
    ) -> tuple[
        BatchedObservation, Float[torch.Tensor, "b"], Bool[torch.Tensor, "b"], Bool[torch.Tensor, "b"], dict[str, Any]
    ]:
        return self.env.step(actions, action_spec=action_spec)

    def render(self):  # noqa: ANN201, D102
        return self.env.render()

    def close(self, **kwargs: Any) -> None:  # noqa: D102
        return self.env.close(**kwargs)


class SkilletGymEnv(gym.Env):
    """The Skillet Gymnasium environment that all Rl environments inherit from.

    In addition to the standard Gym functions, it includes properties that enable interfacing
    with the SkilletEnvWrapper across IsaacLab, Mujoco, and ROS2 environments.
    """

    @abstractmethod
    def _find_link_idx(self, link: str) -> int:
        """Find the link index of the robot link."""
        raise NotImplementedError

    @abstractmethod
    def _find_joint_idx(self, joint: str) -> int:
        """Find the joint index robot joint."""
        raise NotImplementedError

    @abstractmethod
    def _get_latest_rgbd() -> torch.Tensor:
        """Get the latest RGBD information from the camera in the environment."""
        raise NotImplementedError

    @property
    def _prev_actions(self) -> torch.Tensor:
        """Return the previous action taken in the environment."""
        raise NotImplementedError

    @property
    def robot(self) -> Any:
        """Returns the robot articulation for Isaac and Mj environments. Useful for helper functions."""
        raise NotImplementedError

    @property
    def _tcp_pose_b(self) -> torch.Tensor:
        """Return current tool frame position."""
        raise NotImplementedError

    @property
    def _ee_vel_b(self) -> torch.Tensor:
        """Return current EE velocity."""
        raise NotImplementedError

    @property
    def _joint_positions(self) -> torch.Tensor:
        """Return current joint positions."""
        raise NotImplementedError

    @property
    def _joint_velocities(self) -> torch.Tensor:
        """Return current joint velocities."""
        raise NotImplementedError

    @property
    def _joint_efforts(self) -> torch.Tensor:
        """Return current joint efforts (torques)."""
        raise NotImplementedError

    @property
    def _robot_body_pose_w(self) -> torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _robot_root_pose_w(self) -> torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _jacobians(self) -> torch.Tensor:
        """Return the jacobian frame transforms of the robot."""
        raise NotImplementedError

    @property
    def _robot_dof_lower_limits(self) -> torch.Tensor:
        """Return the lower limits of the robot joints."""
        raise NotImplementedError

    @property
    def _robot_dof_upper_limits(self) -> torch.Tensor:
        """Return the upper limits of the robot joints."""
        raise NotImplementedError

    @property
    def _gravity_vector(self) -> torch.Tensor:
        """Return the gravity compenstation vector."""
        raise NotImplementedError

    @property
    def _mass_matrices(self) -> torch.Tensor:
        """Return the mass matrices."""
        raise NotImplementedError

    @property
    def _robot_body_vel_w(self) -> torch.Tensor:
        """Return body velocity in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _joint_centers(self) -> torch.Tensor:
        """Return joint centers."""
        raise NotImplementedError
