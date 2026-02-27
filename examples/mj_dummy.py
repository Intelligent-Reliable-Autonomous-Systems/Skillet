"""main_isaac.py.

Test file for executor integration with IsaacSim and ROS2

Written by Will Solow and Jeff Jewett, 2026

"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

import gymnasium as gym
import torch
from jaxtyping import Float, Int

from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.mj_env_wrapper import MJEnvWrapper
from skillet.envs.util import parse_mj_env_cfg
from skillet.policy.dummy import RandomPolicy, ZeroPolicy
from skillet.skill.low_level import FixedLengthSkill

# add argparse arguments
parser = argparse.ArgumentParser(description="Main Mujoco Executor file through Mujoco Warp.")

parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="MJ-Lift-Cube-Kinova-v0", help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="GPU device")

# parse the arguments
args_cli = parser.parse_args()


import kinova_tasks.mj_tasks  # noqa: F401

BxN_Obs = Float[torch.Tensor, "b n"]
"""Environment observation: torch.Tensor[(b, n), float]"""
BxM_Action = Float[torch.Tensor, "b m"]
"""Environment action: torch.Tensor[(b, m), float]"""
B_Int_HighLevel = Int[torch.Tensor, "b"]
"""Selected skills action: torch.Tensor[(b,), int]"""


def main() -> None:
    """Test the executor within the IsaacLab/IsaacSim framework."""
    # create environment configuration

    # For example, the Reach task with the Franka arm has the config
    env_cfg = parse_mj_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Set up Skill executor and environment in framework
    env = MJEnvWrapper[BxN_Obs, BxM_Action](env)

    action_spec: ActionSpec[BxM_Action] = env.action_spec
    observation_spec: ObservationSpec[BxN_Obs] = env.obs_spec

    # Low-level policies
    zero_policy = ZeroPolicy[BxN_Obs, BxM_Action](observation_spec, action_spec)
    random_policy = RandomPolicy[BxN_Obs, BxM_Action](observation_spec, action_spec)
    # Skills
    skill_length = 40
    zero_skill = FixedLengthSkill[BxN_Obs, BxM_Action, None](name="zero_skill", policy=zero_policy, length=skill_length)
    random_skill = FixedLengthSkill[BxN_Obs, BxM_Action, None](
        name="random_skill", policy=random_policy, length=skill_length
    )
    skills = [zero_skill, random_skill]

    # High-level policy
    options_spec = ActionSpec[B_Int_HighLevel](
        space=gym.spaces.MultiDiscrete([len(skills)] * args_cli.num_envs),
        name="options",
        is_torch=True,
        is_batched=True,
        # n_envs=args_cli.num_envs,
    )
    policy_over_options = RandomPolicy[BxN_Obs, B_Int_HighLevel](observation_spec, options_spec)

    policy_over_options_agent = PolicyOverOptionsAgent[BxN_Obs, BxM_Action, B_Int_HighLevel, None](
        skills=[zero_skill, random_skill],
        high_level_policy=policy_over_options,
        params_policy=None,
    )

    # env.step()
    # skill_executor = SkillExecutor(DummyCfg(), env)

    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            # skill_executor.execute()
            print("[INFO][Main] finished run of skill executor, resetting")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
