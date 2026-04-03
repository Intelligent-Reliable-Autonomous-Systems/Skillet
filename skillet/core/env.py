"""The environment interface that adds explicit obs and action specifications.

'Environment' class wraps a gym.Env and adds explicit obs and action specifications.
'BatchedEnvironment' class wraps a gym.vector.VectorEnv and adds explicit obs and action specifications.
'BasicEnvironment' class is a minimal implementation that supports raw state observations (full observability).
'BasicBatchedEnvironment' class is a minimal batched implementation that supports batched observations and actions.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, overload

import gymnasium as gym
import torch
from jaxtyping import Bool, Float
from typing_extensions import override

from skillet.core.spaces import (
    Action,
    ActionSpec,
    ArrayLike,
    BatchedAction,
    BatchedObservation,
    Observation,
    ObservationSpec,
    SpaceSpecification,
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


class _EnvironmentBase(ABC, Generic[TObs, TAction]):
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

    @abstractmethod
    def supports_observation_spec(self, obs_spec: ObservationSpec[Any]) -> bool:
        """Check if the environment supports a specific observation type."""
        raise NotImplementedError

    @abstractmethod
    def supports_action_spec(self, action_spec: ActionSpec[Any]) -> bool:
        """Check if the environment supports a specific action type."""
        raise NotImplementedError

    def coerce_obs_spec(self, obs_spec: str | ObservationSpec[Any]) -> ObservationSpec[Any]:
        """Coerce an observation specification or name to a compatible ObservationSpec.

        Raises ValueError if the observation specification is not supported by the environment.
        """
        if isinstance(obs_spec, str):
            if obs_spec == self.obs_spec.name:
                return self.obs_spec
            raise ValueError(f"Observation spec {obs_spec} not supported by environment.")
        return obs_spec

    def coerce_action_spec(self, action_spec: str | ActionSpec[Any]) -> ActionSpec[Any]:
        """Coerce an action specification or name to a compatible ActionSpec.

        Raises ValueError if the action specification is not supported by the environment.
        """
        if isinstance(action_spec, str):
            if action_spec == self.action_spec.name:
                return self.action_spec
            raise ValueError(f"Action spec {action_spec} not supported by environment.")
        return action_spec

    @overload
    def get_observation(self) -> TObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...

    @abstractmethod
    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:
        """Get the latest observation from the environment, optionally querying a specific observation type."""
        raise NotImplementedError

    def get_state(self) -> State:
        """Get the latest state from the environment."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[TObs, dict]:
        """Reset the environment."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: TAction, action_spec: ActionSpec[Any] | None = None) -> \
            tuple[TObs, float | Float[ArrayLike, " b"], bool | Bool[ArrayLike, " b"], \
                bool | Bool[ArrayLike, " b"], dict]:
        """Step the environment.

        Args:
            action: The action to take
            action_spec: Optional action specification to use for the action. If not provided, uses self.action_spec.

        Returns:
            A tuple containing the observation, reward, done, truncated, and info

        """
        raise NotImplementedError


class Environment(_EnvironmentBase[TObs, TAction], gym.Env[TObs, TAction], Generic[TObs, TAction]):
    """An environment interface for the Robot Skills framework.

    Generic type parameters:
        TObs: The type of the environment observation. e.g. np.ndarray[(8,), float]
        TAction: The type associated with the action spec
    """

    @override
    @abstractmethod
    def step(self, action: TAction, action_spec: ActionSpec[Any] | None = None) -> \
            tuple[TObs, float, bool, bool, dict]:
        raise NotImplementedError


class BatchedEnvironment(
    _EnvironmentBase[TBObs, TBAction], gym.vector.VectorEnv[TBObs, TBAction, ArrayLike], Generic[TBObs, TBAction]
):
    """A batched environment that supports batched observations and actions.

    Generic type parameters:
        TBObs: The type of the batched environment observation. e.g. torch.Tensor[(b, 8), float]
        TBAction: The type associated with the batched action spec
    """

    @abstractmethod
    @override
    def step(self, action: TBAction, action_spec: ActionSpec[Any] | None = None) -> \
            tuple[TBObs, Float[ArrayLike, " b"], Bool[ArrayLike, " b"], Bool[ArrayLike, " b"], dict]:
        raise NotImplementedError


class BasicEnvironment(Environment[TObs, TAction], gym.Wrapper[TObs, TAction, TObs, TAction], Generic[TObs, TAction]):
    """A basic environment that supports raw state observations (full observability)."""

    def __init__(
        self, env: gym.Env, is_torch: bool = False, device: torch.device | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize a basic environment wrapper.

        Args:
            env: The gym.Env environment to wrap
            is_torch: Whether the environment uses PyTorch tensors
            device: The device to use for the environment. If not provided, uses the device of the environment.
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
            device=device,
        )
        self._action_spec = ActionSpec[TAction](
            space=env.action_space,
            name="action",
            is_torch=is_torch,
            is_batched=False,
            device=device,
        )

    @property
    def obs_spec(self) -> ObservationSpec[TObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TAction]:  # noqa: D102
        return self._action_spec

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:  # noqa: D102
        return obs_spec.name == self.obs_spec.name

    def supports_action_spec(self, action_spec: ActionSpec) -> bool:  # noqa: D102
        return action_spec.name == self.action_spec.name

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[TObs, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        return obs, info

    @override
    def step(self, action: TAction, action_spec: ActionSpec[Any] | None = None) -> tuple[TObs, float, bool, bool, dict]:
        if action_spec is not None and not self.supports_action_spec(action_spec):
            raise ValueError(f"Action spec {action_spec} not supported by environment.")
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


class BasicBatchedEnvironment(BatchedEnvironment[TBObs, TBAction], gym.vector.VectorWrapper, Generic[TBObs, TBAction]):
    """A simple BatchedEnvironment implementation that wraps a standard gym.vector.VectorEnv.

    Generic type parameters:
        TBObs: The type of the batched environment observation. e.g. torch.Tensor[(batch_size, 8), float]
        TBAction: The type associated with the batched action spec
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        is_torch: bool = False,
        device: torch.device | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a basic batched environment wrapper.

        Args:
            env: The gym.vector.VectorEnv environment to wrap
            is_torch: Whether the environment uses PyTorch tensors
            device: The device to use for the environment. If not provided, uses the device of the environment.
            *args: Additional arguments to pass to the gym.vector.VectorWrapper constructor
            **kwargs: Additional keyword arguments to pass to the gym.vector.VectorWrapper constructor

        """
        super().__init__(env, *args, **kwargs)
        self.last_obs = None
        self._obs_spec = ObservationSpec[TBObs](
            space=env.single_observation_space,
            name="obs",
            is_torch=is_torch,
            is_batched=True,
            n_envs=-1,  # Variable batch size is more flexible
            device=device,
        )
        self._action_spec = ActionSpec[TBAction](
            space=env.single_action_space,
            name="action",
            is_torch=is_torch,
            is_batched=True,
            n_envs=-1,
            device=device,
        )

    @property
    def obs_spec(self) -> ObservationSpec[TBObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:  # noqa: D102
        return obs_spec.name == self.obs_spec.name

    def supports_action_spec(self, action_spec: ActionSpec) -> bool:  # noqa: D102
        return action_spec.name == self.action_spec.name

    @override
    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TBObs, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        return self.obs_spec.cast(obs), info

    @override
    def step(
        self, actions: TBAction
    ) -> tuple[TBObs, Float[ArrayLike, " b"], Bool[ArrayLike, " b"], Bool[ArrayLike, " b"], dict]:
        obs, reward, term, trunc, info = self.env.step(actions)
        self.last_obs = obs
        return self.obs_spec.cast(obs), self.obs_spec.cast(reward, False), self.obs_spec.cast(term, False), \
            self.obs_spec.cast(trunc, False), info

    @overload
    def get_observation(self) -> TBObs: ...
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


class BatchToSingleWrapper(Environment[TObs, TAction], Generic[TObs, TAction]):
    """A wrapper that converts a batched environment to a single environment."""

    def __init__(self, env: BatchedEnvironment[TObs, TAction]) -> None:
        """Initialize a batch to single environment wrapper.

        Args:
            env: The batched environment to wrap

        """
        self.batched_env = env
        """The wrapped batched environment."""

    @property
    def unwrapped(self) -> Environment:
        return self.batched_env.unwrapped

    @property
    @override
    def obs_spec(self) -> ObservationSpec[TObs]:
        return self.batched_env.obs_spec.unbatched()

    @property
    @override
    def action_spec(self) -> ActionSpec[TAction]:
        return self.batched_env.action_spec.unbatched()

    @override
    def supports_observation_spec(self, obs_spec: ObservationSpec[Any]) -> bool:
        return self.batched_env.supports_observation_spec(obs_spec.unbatched())

    @override
    def supports_action_spec(self, action_spec: ActionSpec[Any]) -> bool:
        return self.batched_env.supports_action_spec(action_spec.batched())

    @override
    def coerce_obs_spec(self, obs_spec: str | ObservationSpec[Any]) -> ObservationSpec[Any]:
        if isinstance(obs_spec, SpaceSpecification):
            obs_spec = obs_spec.batched()
        return self.batched_env.coerce_obs_spec(obs_spec).unbatched()

    @override
    def coerce_action_spec(self, action_spec: str | ActionSpec[Any]) -> ActionSpec[Any]:
        if isinstance(action_spec, SpaceSpecification):
            action_spec = action_spec.batched()
        return self.batched_env.coerce_action_spec(action_spec).unbatched()

    @override
    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:
        if obs_spec is not None:
            obs_spec = obs_spec.batched()
        batched_obs = self.batched_env.get_observation(obs_spec)
        return obs_spec.cast(batched_obs)

    @override
    def get_state(self) -> State:
        batched_state = self.batched_env.get_state()
        try:
            obs_spec = self.coerce_obs_spec("state")
        except ValueError:
            obs_spec = self.obs_spec
        return obs_spec.cast(batched_state)

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[TObs, dict]:
        batched_obs, info = self.batched_env.reset(seed=seed, options=options)
        return self.obs_spec.cast(batched_obs), self._unbatch_info(info)

    @override
    def step(self, action: TAction, action_spec: ActionSpec[Any] | None = None) -> tuple[TObs, float, bool, bool, dict]:
        if action_spec is not None:
            action_spec = action_spec.batched()
        batched_obs, reward, term, trunc, info = self.batched_env.step(action, action_spec=action_spec)
        return self.obs_spec.cast(batched_obs), reward[0], term[0], trunc[0], self._unbatch_info(info)

    def _unbatch_info(self, batched_info: dict) -> dict:
        """Unbatch the info dictionary."""
        return {k: v[0] for k, v in batched_info.items()}
