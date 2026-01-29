import abc
import gymnasium as gym

class Environment(gym.Wrapper, abc.ABC):
    """An environment interface for the Robot Skills framework."""

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

    @abc.abstractmethod
    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        """Check if the environment supports a specific observation type."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self, obs_spec: ObservationSpec = None) -> Observation:
        """Get the latest observation from the environment, optionally querying a specific observation type."""
        raise NotImplementedError

    def get_state(self) -> State:
        """Get the latest state from the environment."""
        return self.get_observation(ObservationSpec.RAW_STATE)

class BasicEnvironment(Environment):
    """A basic environment that supports raw state observations (full observability)."""

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_obs = None

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return obs_spec == ObservationSpec.RAW_STATE

    def reset(self, *args, **kwargs) -> None:
        obs, info = self.env.reset(*args, **kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, term, trunc, info

    def get_observation(self, obs_spec: ObservationSpec = None) -> Observation:
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is not None and not self.supports_observation_spec(obs_spec):
            raise ValueError(f"Observation spec {obs_spec} not supported by environment.")
        return self.last_obs
