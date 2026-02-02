"""Example of the core functionality of the Robot Skills framework with some classic gym environments."""

import gymnasium as gym
import numpy as np
from jaxtyping import Float, Int

from robot_skills.core.env import BasicEnvironment
from robot_skills.core.spaces import (
    ActionSpec,
    ObservationSpec,
)

CartPoleAction = Int[np.ndarray, "2"]
"""A Discrete(2) numpy array."""

CartPoleObservation = Float[np.ndarray, "4"]
"""A Box(4) numpy array."""

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    action_spec = ActionSpec[CartPoleAction](
        space=env.action_space,
        name="cartpole_action",
        is_torch=False,
        is_batched=False,
    )
    observation_spec = ObservationSpec[CartPoleObservation](
        space=env.observation_space,
        name="cartpole_observation",
        is_torch=False,
        is_batched=False,
    )
    env = BasicEnvironment[CartPoleObservation, CartPoleAction](env)
    obs, _ = env.reset()
    print(obs)
    action = action_spec.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(obs)
    print(reward)
