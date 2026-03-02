"""ros2_teleop.py.

Script to run a keyboard teleoperation with ROS2 manipulation environments.

Written by Will Solow, 2026.
"""

import argparse
import os

import gymnasium as gym
import torch

import kinova_tasks.ros2_tasks  # noqa: F401
from skillet.controllers.devices import Se3Keyboard, Se3KeyboardCfg
from skillet.envs.ros2_skillet_env import ROS2SkilletEnv
from skillet.envs.util import parse_ros2_env_cfg, setup_ros

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to run on.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--task", type=str, default="ROS2-Kinova-Twist-Rel-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--robot_ip", default="192.168.8.10", type=str, help="IP of the robot.")
parser.add_argument("--launch_ros", action="store_true", help="If to launch robot bringup files.")
parser.add_argument("--use_fake_hardware", default="false", type=str, help="If to use fake hardware (RViz) or not.")

# parse the arguments
args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


def main() -> None:
    """Run keyboard teleoperation with ROS2.

    Defaults to ROS2-Kinova-Twist-Rel-v0 environment which assumes that actions are velocities of the end effector in 6 DoF.
    """
    env_cfg = parse_ros2_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        ros2_workspace=args_cli.ros2_ws,
    )
    env_cfg.robot_ip = args_cli.robot_ip
    env_cfg.use_fake_hardware = args_cli.use_fake_hardware
    env_cfg.launch_ros = args_cli.launch_ros

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    # Wrap environment in Skillet
    env = ROS2SkilletEnv(env)

    # Always active for other devices
    sensitivity = args_cli.sensitivity
    # Create teleop device from config if present, otherwise create manually
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=10 * sensitivity)
        )
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    print(f"[INFO] Using teleop device: {teleop_interface}")

    env.reset()
    teleop_interface.reset()

    print("[INFO] Teleoperation started")

    while True:
        with torch.inference_mode():
            action = teleop_interface.advance()
            actions = action.repeat(env.num_envs, 1)
            env.step(actions)


if __name__ == "__main__":
    main()
