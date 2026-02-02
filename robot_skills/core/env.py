import abc
from typing import Any, Generic, Sequence, TypeVar, overload, TypeAlias

import gymnasium as gym
from jaxtyping import Bool, Float, Shaped
import torch
import numpy as np

from robot_skills.core.spaces import Action, ActionSpec, ArrayLike, BatchedAction, BatchedObservation, ObservationSpec, Observation, SpaceItem, State

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

class Environment(_EnvironmentBase[TObs, TAction], gym.Wrapper[TObs, TAction, TObs, TAction], Generic[TObs, TAction]):
    """An environment interface for the Robot Skills framework.
    
    Generic type parameters:
        TEnvObs: The type of the environment observation. e.g. np.ndarray[(8,), float]
        TAction: The type associated with the action spec
    """
    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

class BatchedEnvironment(_EnvironmentBase[TBObs, TBAction], gym.vector.VectorWrapper, Generic[TBObs, TBAction]):
    """A batched environment that supports batched observations and actions."""
    
    def __init__(self, env: gym.vector.VectorEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

class BasicEnvironment(Environment[TObs, TAction], Generic[TObs, TAction]):
    """A basic environment that supports raw state observations (full observability)."""

    def __init__(self, env: gym.Env, is_torch: bool = False, *args, **kwargs):
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
    def obs_spec(self) -> ObservationSpec[TObs]:
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TAction]:
        return self._action_spec

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return True # TODO: compare obs_spec.space with self.env.observation_space

    def reset(self, *args, **kwargs) -> tuple[TObs, dict]:
        obs, info = self.env.reset(*args, **kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action: TAction) -> tuple[TObs, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, term, trunc, info
    
    @overload
    def get_observation(self) -> TObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...
    
    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is not None and not self.supports_observation_spec(obs_spec):
            raise ValueError(f"Observation spec {obs_spec} not supported by environment.")
        return self.last_obs

    def get_state(self) -> TObs:
        return self.get_observation()

class BasicBatchedEnvironment(BatchedEnvironment[TBObs, TBAction], Generic[TBObs, TBAction]):
    """A batched environment that supports batched observations and actions."""
    
    def __init__(self, env: gym.vector.VectorEnv, is_torch: bool = False, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_obs = None
        self._obs_spec = ObservationSpec[TObs](
            space=env.single_observation_space,
            name="obs",
            is_torch=is_torch,
            is_batched=True,
            n_envs=-1, # Variable batch size is more flexible
        )
        self._action_spec = ActionSpec[TAction](
            space=env.single_action_space,
            name="action",
            is_torch=is_torch,
            is_batched=True,
            n_envs=-1, 
        )

    @property
    def obs_spec(self) -> ObservationSpec[TObs]:
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TAction]:
        return self._action_spec

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return True # TODO: compare obs_spec.space with self.env.observation_space

    def reset(self, *args, **kwargs) -> tuple[TBObs, dict]:
        obs, info = self.env.reset(*args, **kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action: TBAction) -> tuple[TBObs, Float[ArrayLike, "b"], Bool[ArrayLike, "b"], Bool[ArrayLike, "b"], dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, term, trunc, info
    
    @overload
    def get_observation(self) -> TObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...
    
    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is not None and not self.supports_observation_spec(obs_spec):
            raise ValueError(f"Observation spec {obs_spec} not supported by environment.")
        return self.last_obs

    def get_state(self) -> TObs:
        return self.get_observation()
    