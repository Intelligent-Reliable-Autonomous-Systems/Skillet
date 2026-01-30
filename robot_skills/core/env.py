import abc
from typing import Any, Generic, TypeVar, overload

import gymnasium as gym

from skills.core.spaces import ObservationSpec, Observation, State

type Action = Any
"""Represents an action in the environment."""
TObs = TypeVar("TObs", bound=Observation)
"""The type associated with the observation spec"""
TEnvObs = TypeVar("TEnvObs", bound=Observation)
"""The default observation type returned by the environment."""

class Environment(gym.Wrapper, abc.ABC, Generic[TEnvObs]):
    """An environment interface for the Robot Skills framework.
    
    Generic type parameters:
        TEnvObs: The type of the environment observation. e.g. np.ndarray[(8,), float]
    """

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

    @abc.abstractmethod
    def supports_observation_spec(self, obs_spec: ObservationSpec[Any]) -> bool:
        """Check if the environment supports a specific observation type."""
        raise NotImplementedError

    @overload
    def get_observation(self) -> TEnvObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TObs]) -> TObs: ...
    
    @abc.abstractmethod
    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:
        """Get the latest observation from the environment, optionally querying a specific observation type."""
        raise NotImplementedError

    def get_state(self) -> State:
        """Get the latest state from the environment."""
        raise NotImplementedError

class BasicEnvironment(Environment[TEnvObs]):
    """A basic environment that supports raw state observations (full observability)."""

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_obs = None

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return True # TODO: compare obs_spec.space with self.env.observation_space

    def reset(self, *args, **kwargs) -> None:
        obs, info = self.env.reset(*args, **kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action: Action) -> tuple[TEnvObs, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, term, trunc, info
    
    @overload
    def get_observation(self) -> TEnvObs: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TObs]) -> TObs: ...
    
    @abc.abstractmethod
    def get_observation(self, obs_spec: ObservationSpec[Any] | None = None) -> Any:
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is not None and not self.supports_observation_spec(obs_spec):
            raise ValueError(f"Observation spec {obs_spec} not supported by environment.")
        return self.last_obs

    def get_state(self) -> TEnvObs:
        return self.get_observation()