"""Example of the core functionality of the Robot Skills framework with some classic gym environments."""

import gymnasium as gym
import numpy as np
from jaxtyping import Int, Float

from robot_skills.core.env import BasicEnvironment
from robot_skills.core.spaces import ActionSpec, ObservationSpec, Observation, SpaceItem, SpaceSpecification, State, make_action_spec

CartPoleAction = Int[np.ndarray, "2"]
"""A Discrete(2) numpy array."""

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    action_spec = ActionSpec[CartPoleAction](
        obs_type=None,
        space=env.action_space,
        name="cartpole_action",
        is_torch=False,
        is_batched=False,
    )
    observation_spec = ObservationSpec[Float[np.ndarray, "4"]](
        space=env.observation_space,
        name="cartpole_observation",
        is_torch=False,
        is_batched=False,
    )
    env = BasicEnvironment[Float[np.ndarray, "4"], Int[np.ndarray, "2"]](env)
    obs = env.reset()
    print(obs)
    action = action_spec.space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(obs)
    print(reward)