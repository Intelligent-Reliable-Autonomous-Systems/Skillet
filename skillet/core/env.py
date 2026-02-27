"""The environment interface that adds explicit obs and action specifications.

'Environment' class wraps a gym.Env and adds explicit obs and action specifications.
'BatchedEnvironment' class wraps a gym.vector.VectorEnv and adds explicit obs and action specifications.
'BasicEnvironment' class is a minimal implementation that supports raw state observations (full observability).
'BasicBatchedEnvironment' class is a minimal batched implementation that supports batched observations and actions.
"""

import abc
from typing import Any, Generic, TypeVar, overload

import gymnasium as gym
import torch
from jaxtyping import Bool, Float

from skillet.core.spaces import (
    Action,
    ActionSpec,
    ArrayLike,
    BatchedAction,
    BatchedObservation,
    BatchedSpaceValue,
    Observation,
    ObservationSpec,
    State,
)

TSpecObs = TypeVar("TSpecObs", bound=Observation)
"""A generic type of the requested observation specification"""
TObs = TypeVar("TObs", bound=Observation)
"""A generic type of the observation returned by the environment."""
TBObs = TypeVar("TBObs", bound=BatchedObservation)
"""A generic type of the batched observation returned by the environment."""
TAction = TypeVar("TAction", bound=Action)
"""A generic type of the action returned by the environment."""
TBAction = TypeVar("TBAction", bound=BatchedAction)
"""A generic type of the batched action returned by the environment."""


class _EnvironmentBase(abc.ABC, Generic[TObs, TAction]):
    """An environment interface for the Robot Skills framework.

    Generic type parameters:
        TEnvObs: The type of the environment observation. e.g. np.ndarray[(8,), float]
        TAction: The type associated with the action spec
    """

    @property
    def obs_spec(self) -> ObservationSpec[TObs]:
        """The default observation specification for the environment, for observations returned by step()."""
        raise NotImplementedError

    @property
    def action_spec(self) -> ActionSpec[TAction]:
        """The default action specification for the environment, for actions consumed by step()."""
        raise NotImplementedError

    @abc.abstractmethod
    def supports_observation_spec(self, obs_spec: ObservationSpec[Any]) -> bool:
        """Check if the environment supports a specific observation type."""
        raise NotImplementedError

    @overload
    def get_observation(self) -> TObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...

    @abc.abstractmethod
    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:
        """Get the latest observation from the environment, optionally querying a specific observation type."""
        raise NotImplementedError

    def get_state(self) -> State:
        """Get the latest state from the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[TObs, dict]:
        """Reset the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: TAction) -> tuple[TObs, float, bool, bool, dict]:
        """Step the environment."""
        raise NotImplementedError


class Environment(_EnvironmentBase[TObs, TAction], gym.Wrapper[TObs, TAction, TObs, TAction], Generic[TObs, TAction]):
    """An environment interface for the Robot Skills framework.

    Generic type parameters:
        TObs: The type of the environment observation. e.g. np.ndarray[(8,), float]
        TAction: The type associated with the action spec
    """


class BatchedEnvironment(_EnvironmentBase[TBObs, TBAction], gym.vector.VectorWrapper, Generic[TBObs, TBAction]):
    """A batched environment that supports batched observations and actions.

    Generic type parameters:
        TBObs: The type of the batched environment observation. e.g. torch.Tensor[(b, 8), float]
        TBAction: The type associated with the batched action spec
    """


class BasicEnvironment(Environment[TObs, TAction], Generic[TObs, TAction]):
    """A basic environment that supports raw state observations (full observability)."""

    def __init__(self, env: gym.Env, is_torch: bool = False, *args: Any, **kwargs: Any) -> None:
        """Initialize a basic environment wrapper.

        Args:
            env: The gym.Env environment to wrap
            is_torch: Whether the environment uses PyTorch tensors
            *args: Additional arguments to pass to the gym.Wrapper constructor
            **kwargs: Additional keyword arguments to pass to the gym.Wrapper constructor

        Returns:
            None

        """
        super().__init__(env, *args, **kwargs)
        self.last_obs = None
        self._obs_spec = ObservationSpec[TObs](
            space=env.observation_space,
            name="obs",
            is_torch=is_torch,
            is_batched=False,
        )
        self._action_spec = ActionSpec[TAction](
            space=env.action_space,
            name="action",
            is_torch=is_torch,
            is_batched=False,
        )

    @property
    def obs_spec(self) -> ObservationSpec[TObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TAction]:  # noqa: D102
        return self._action_spec

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:  # noqa: D102
        del obs_spec
        return True  # TODO: compare obs_spec.space with self.env.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[TObs, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        return obs, info

    def step(self, action: TAction) -> tuple[TObs, float, bool, bool, dict]:  # noqa: D102
        obs, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, term, trunc, info

    @overload
    def get_observation(self) -> TObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...

    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:  # noqa: D102
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is not None and not self.supports_observation_spec(obs_spec):
            raise ValueError(f"Observation spec {obs_spec} not supported by environment.")
        return self.last_obs

    def get_state(self) -> TObs:  # noqa: D102
        return self.get_observation()


class BasicBatchedEnvironment(BatchedEnvironment[TBObs, TBAction], Generic[TBObs, TBAction]):
    """A batched environment that supports batched observations and actions.

    Generic type parameters:
        TBObs: The type of the batched environment observation. e.g. torch.Tensor[(batch_size, 8), float]
        TBAction: The type associated with the batched action spec
    """

    def __init__(self, env: gym.vector.VectorEnv, is_torch: bool = False, *args: Any, **kwargs: Any) -> None:
        """Initialize a basic batched environment wrapper.

        Args:
            env: The gym.vector.VectorEnv environment to wrap
            is_torch: Whether the environment uses PyTorch tensors
            *args: Additional arguments to pass to the gym.vector.VectorWrapper constructor
            **kwargs: Additional keyword arguments to pass to the gym.vector.VectorWrapper constructor

        """
        super().__init__(env, *args, **kwargs)
        self.last_obs = None
        self._obs_spec = ObservationSpec[TObs](
            space=env.single_observation_space,
            name="obs",
            is_torch=is_torch,
            is_batched=True,
            n_envs=-1,  # Variable batch size is more flexible
        )
        self._action_spec = ActionSpec[TAction](
            space=env.single_action_space,
            name="action",
            is_torch=is_torch,
            is_batched=True,
            n_envs=-1,
        )

    @property
    def obs_spec(self) -> ObservationSpec[TObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TAction]:  # noqa: D102
        return self._action_spec

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:  # noqa: D102
        return True  # TODO: compare obs_spec.space with self.env.observation_space

    def reset(  # noqa: D102
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TBObs, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        return obs, info

    def step(  # noqa: D102
        self, actions: TBAction
    ) -> tuple[TBObs, Float[ArrayLike, "b"], Bool[ArrayLike, "b"], Bool[ArrayLike, "b"], dict]:  # noqa: F821
        obs, reward, term, trunc, info = self.env.step(actions)
        self.last_obs = obs
        return obs, reward, term, trunc, info

    @overload
    def get_observation(self) -> TObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...

    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:  # noqa: D102
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is not None and not self.supports_observation_spec(obs_spec):
            raise ValueError(f"Observation spec {obs_spec} not supported by environment.")
        return self.last_obs

    def get_state(self) -> TObs:  # noqa: D102
        return self.get_observation()


class AsGymVectorEnv(gym.vector.VectorEnv):
    """A wrapper for gym.Env environments that already have a vectorized interface to the new gymnasium vector environment."""

    def __init__(self, env: gym.Env, num_envs: int | None = None) -> None:
        """Initialize the environment.

        Args:
            env: The gym.Env to wrap. Must have a single observation space and action space.
            num_envs: Optionally, the number of environments to wrap

        """
        self.env = env
        self.num_envs = num_envs or env.get_wrapper_attr("num_envs")
        if self.num_envs is None:
            raise ValueError("The environment does not have a number of environments .num_envs")
        if not env.has_wrapper_attr("single_observation_space") or not env.has_wrapper_attr("single_action_space"):
            raise ValueError("The environment does not have a single observation space or action space.")
        self.single_observation_space = env.get_wrapper_attr("single_observation_space")
        self.single_action_space = env.get_wrapper_attr("single_action_space")
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = env.unwrapped.device if hasattr(env.unwrapped, "device") else None
        self.max_episode_length = (
            env.unwrapped.max_episode_length if hasattr(env.unwrapped, "max_episode_length") else None
        )
        self.cfg = env.unwrapped.cfg if hasattr(env.unwrapped, "cfg") else None

    def reset(  # noqa: D102
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[BatchedSpaceValue, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(  # noqa: D102
        self, actions: BatchedAction
    ) -> tuple[BatchedSpaceValue, Float[ArrayLike, "b"], Bool[ArrayLike, "b"], Bool[ArrayLike, "b"], dict[str, Any]]:
        return self.env.step(actions)

    def render(self):  # noqa: ANN201, D102
        return self.env.render()

    def close(self, **kwargs: Any) -> None:  # noqa: D102
        return self.env.close(**kwargs)

    @property
    def episode_length_buf(self):
        buf = self.env.unwrapped.episode_length_buf if hasattr(self.env.unwrapped, "episode_length_buf") else None
        if isinstance(buf, torch.Tensor):
            return buf
        if isinstance(buf, int):
            return torch.as_tensor([buf], device=self.device).unsqueeze(0)
        raise ValueError(f"Unsupported `episode_length_buf` type {type(buf)}")

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        """Set the episode length buffer.

        This is needed to perform random initialization of episode lengths in RSL-RL.

        """
        if not hasattr(self.env.unwrapped, "episode_length_buf"):
            raise ValueError(f"`{self.env.unwrapped}` has no attribute `episode length buf`.")
        if isinstance(self.env.unwrapped.episode_length_buf, torch.Tensor):
            self.env.unwrapped.episode_length_buf = value
        elif isinstance(self.env.unwrapped.episode_length_buf, int):
            self.env.unwrapped.episode_length_buf = value.squeeze().item()
        else:
            raise ValueError(f"Unsupported `episode_length_buf` type {type(self.env.unwrapped.episode_length_buf)}")
