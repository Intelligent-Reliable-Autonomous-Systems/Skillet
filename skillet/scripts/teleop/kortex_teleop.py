"""ros2_teleop_vr.py.

Script to run a keyboard or joystick teleoperation with ROS2 manipulation environments.

Written by Will Solow, 2026.
"""

import argparse

import torch

import skillet_tasks.kortex_tasks  # noqa: F401
from skillet.controllers.devices import Se3Keyboard, Se3KeyboardCfg, VRHeadset, VRHeadsetCfg, VRJoystick, VRJoystickCfg
from skillet.envs.skillet_env import SkilletEnv
from skillet.agents import SkilletModerator
from skillet_tasks.kortex_tasks.factory import create_kortex_env

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Kortex environments.")
parser.add_argument("--task", type=str, default="Kortex-Gen3-v0", help="Environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to run on.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices={"keyboard", "vr_joystick", "vr_headset"},
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")


# parse the arguments
args_cli = parser.parse_args()


def main() -> None:
    """Run keyboard teleoperation with Kortex API.

    Defaults to Kortex-Gen3-v0 environment and passes the twist_tcp Action Specification.
    """
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "device": "cuda",
        "num_envs": args_cli.num_envs,
        "use_tabletop_camera": False,
        "gripper_close_time": 0.0,
    }

    env = create_kortex_env(args_cli.task, env_cfg)

    # Wrap environment in Skillet
    env = SkilletEnv(env)

    # Always active for other devices
    sensitivity = args_cli.sensitivity
    # Create teleop device from config if present, otherwise create manually
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.15 * sensitivity, rot_sensitivity=10 * sensitivity)
        )
    elif args_cli.teleop_device.lower() == "vr_joystick":
        teleop_interface = VRJoystick(
            VRJoystickCfg(pos_sensitivity=0.15 * sensitivity, rot_sensitivity=10 * sensitivity)
        )
    elif args_cli.teleop_device.lower() == "vr_headset":
        teleop_interface = VRHeadset(VRHeadsetCfg(pos_sensitivity=0.15 * sensitivity, rot_sensitivity=10 * sensitivity))
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    env.reset()
    teleop_interface.reset()

    moderator = SkilletModerator()
    moderator.run_teleop_loop(env, teleop_interface)


if __name__ == "__main__":
    main()
