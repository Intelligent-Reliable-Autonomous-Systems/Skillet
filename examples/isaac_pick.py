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
from isaaclab.app import AppLauncher
from jaxtyping import Float, Int

from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.isaac_env_wrapper import IsaacEnvWrapper
from skillet.policy.dummy import RandomFixedPolicy, RandomPolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy
from skillet.skill.reach_xyz_rpy import ReachXYZRPYSkill

# add argparse arguments
parser = argparse.ArgumentParser(description="Main IsaacSim Executor file through IsaacLab.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Franka-v0", help="Name of the task.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# import isaaclab_tasks after app launcher
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaac_kinova.tasks as tasks  # noqa: F401

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
    # isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg:FrankaReachEnvCfg
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Set up Skill executor and environment in framework
    env = IsaacEnvWrapper[BxN_Obs, BxM_Action](env)

    action_spec: ActionSpec[BxM_Action] = env.action_spec
    observation_spec: ObservationSpec[BxN_Obs] = env.obs_spec

    # Low-level policies
    _obs_spec_ik = ObservationSpec[Float[torch.Tensor, "b ..."]](
        space=gym.spaces.Dict(),
        name="ik_ee",
        is_torch=True,
        is_batched=True,
        n_envs=-1,
        device=env.device,
    )

    ik_ee_pose_policy = PoseAbsIKEEPolicy[BxN_Obs, BxM_Action](_obs_spec_ik, action_spec)
    # Skills
    skill_length = 100
    reach_xyz_skill = ReachXYZRPYSkill[BxN_Obs, BxM_Action, None](
        name="reach_xyz_skill", policy=ik_ee_pose_policy, length=skill_length
    )

    skills = [reach_xyz_skill]

    # Parameters policy
    fixed_param_policy = RandomFixedPolicy[BxN_Obs, BxM_Action](
        observation_spec,
        action_spec,
        # torch.as_tensor([[0.6, 0.1, 0.6, 0.0, 1.57, 0.0]], device=env.device),
        torch.as_tensor([[0.6, 0.1, 0.6, 2.7638, 1.5049, -0.2819]], device=env.device),
    )

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
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    # env.step()
    # skill_executor = SkillExecutor(DummyCfg(), env)

    # simulate environment
    while simulation_app.is_running():
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
