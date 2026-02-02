"""Example of the core functionality of the Robot Skills framework with some classic gym environments."""

import gymnasium as gym
import numpy as np
from jaxtyping import Float, Int

from robot_skills.agents.policy_over_options import PolicyOverOptionsAgent
from robot_skills.core.env import BasicBatchedEnvironment
from robot_skills.core.spaces import (
    ActionSpec,
)
from robot_skills.policy.dummy import RandomPolicy, ZeroPolicy
from robot_skills.skill.fixed_length import FixedLengthSkill

CartPoleAction = Int[np.ndarray, "b 2"]
"""A batched Discrete(2) numpy array."""

CartPoleObservation = Float[np.ndarray, "b 4"]
"""A batched Box(4) numpy array."""

B_Int_HighLevel = Int[np.ndarray, "b"]
"""Selected skills action: numpy.ndarray[(b,), int]"""

if __name__ == "__main__":
    env_id = "MountainCar-v0"
    num_envs = 4
    # env = gym.make_vec("CartPole-v1", num_envs=num_envs)
    # env = gym.make_vec("MountainCarContinuous-v0", num_envs=num_envs, render_mode="rgb_array")
    env = gym.make_vec(env_id, num_envs=num_envs, render_mode="rgb_array")
    env = gym.wrappers.vector.HumanRendering(env)

    env = BasicBatchedEnvironment[CartPoleObservation, CartPoleAction](env)
    print(f"Created environment {env_id} (x{num_envs})")
    print(env.obs_spec)
    print(env.action_spec)

    # Low-level policies
    zero_policy = ZeroPolicy[CartPoleObservation, CartPoleAction](env.obs_spec, env.action_spec)
    random_policy = RandomPolicy[CartPoleObservation, CartPoleAction](env.obs_spec, env.action_spec)
    # Skills
    skill_length = 100
    zero_skill = FixedLengthSkill[CartPoleObservation, CartPoleAction, None](
        name="zero_skill", policy=zero_policy, length=skill_length
    )
    random_skill = FixedLengthSkill[CartPoleObservation, CartPoleAction, None](
        name="random_skill", policy=random_policy, length=skill_length
    )
    skills = [zero_skill, random_skill]

    # High-level policy
    options_spec = ActionSpec[B_Int_HighLevel](
        space=gym.spaces.MultiDiscrete([len(skills)] * num_envs),
        name="options",
        is_torch=False,
        is_batched=True,
    )
    policy_over_options = RandomPolicy[CartPoleObservation, B_Int_HighLevel](env.obs_spec, options_spec)

    policy_over_options_agent = PolicyOverOptionsAgent[CartPoleObservation, CartPoleObservation, CartPoleAction, None](
        skills=[zero_skill, random_skill],
        high_level_policy=policy_over_options,
        params_policy=None,
    )

    for episode in range(10):
        env.reset()
        policy_over_options_agent.execute(env)
        print(f"Episode {episode} finished")

    env.close()
