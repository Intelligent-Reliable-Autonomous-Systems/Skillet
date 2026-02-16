# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for ROS2 environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument("--skill", action="store_true", help="Name of the task.")
parser.add_argument(
    "--ros2_ws", type=str, default=None, required=False, help="Absolute path to ROS2 workspace containing bringup files"
)

# parse the arguments
args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


"""Rest everything follows."""

import gymnasium as gym
import torch

import kinova_tasks.ros2_tasks  # noqa: F401
from skillet.envs.ros2_env_wrapper import ROS2EnvWrapper
from skillet.envs.skill_ros2_env_wrapper import SkillROS2EnvWrapper
from skillet.envs.util import parse_ros2_env_cfg, setup_ros


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_ros2_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, ros2_workspace=args_cli.ros2_ws
    )
    env_cfg.launch_ros = False
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())
    env = SkillROS2EnvWrapper(env) if args_cli.skill else ROS2EnvWrapper(env)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # print(f"Actions:\n{actions}")
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
