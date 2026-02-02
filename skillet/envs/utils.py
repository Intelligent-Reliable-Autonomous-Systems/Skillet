"""Environment utilities."""

from typing import Any

import gymnasium as gym
from jaxtyping import Bool, Float

from skillet.core.spaces import ArrayLike, BatchedAction, BatchedSpaceValue


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

    def reset(  # noqa: D102
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[BatchedSpaceValue, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(  # noqa: D102
        self, actions: BatchedAction
    ) -> tuple[BatchedSpaceValue, Float[ArrayLike, "b"], Bool[ArrayLike, "b"], Bool[ArrayLike, "b"], dict[str, Any]]:  # noqa: F821
        return self.env.step(actions)

    def render(self):  # noqa: ANN201, D102
        return self.env.render()

    def close(self, **kwargs: Any) -> None:  # noqa: D102
        return self.env.close(**kwargs)
