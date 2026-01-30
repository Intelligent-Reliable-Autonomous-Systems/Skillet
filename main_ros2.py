"""main_ros2.py.

Test file for executor integration with IsaacSim and ROS2

Written by Will Solow and Jeff Jewett, 2026

"""

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Main ROS2 executor file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")

# parse the arguments
args_cli = parser.parse_args()


"""Rest everything follows."""
import time

import gymnasium as gym
import numpy as np
from roslibpy import Ros

from env import ROS2EnvWrapper
from env.utils import parse_env_cfg
from executor import SkillExecutor
from policy_cfgs import DummyCfg


def setup_ros() -> Ros:
    """Open the ROS2 interface."""
    print("[INFO][Setup ROS] Waiting to connect to ROSBridge")
    print(
        "[INFO][Setup ROS] Ensure that rosbridge node is running: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`"
    )
    # Wait until it starts
    ros = Ros(host="localhost", port=9090)
    start = time.time()
    while True:
        try:
            ros.run(timeout=1)
            if ros.is_connected:
                print("[INFO][Setup ROS] Connected to rosbridge")
                break
        except RuntimeError:
            if time.time() - start > 30:
                raise TimeoutError(
                    "RosBridge failed to start. Is the rosbridge node running? ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
                )
            time.sleep(0.2)

    return ros


def main() -> None:
    """Test the executor within the IsaacLab/IsaacSim framework."""
    np.set_printoptions(precision=3)
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Set up Skill executor and environment in framework
    env = ROS2EnvWrapper(env)
    skill_executor = SkillExecutor(DummyCfg(), env)

    # simulate environment
    skill_executor.execute()

    # close the environment. Note that the spun up ROS2 nodes will not close automatically
    env.close()


if __name__ == "__main__":
    main()
