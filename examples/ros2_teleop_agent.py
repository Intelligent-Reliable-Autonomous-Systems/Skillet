"""Script to run a keyboard teleoperation with ROS2 manipulation environments."""

import argparse
import os
from collections.abc import Callable

import gymnasium as gym
import torch
from jaxtyping import Float

import kinova_tasks.ros2_tasks  # noqa: F401
from skillet.controllers.devices import Se3Keyboard, Se3KeyboardCfg
from skillet.envs.ros2_env_wrapper import ROS2EnvWrapper
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
parser.add_argument("--task", type=str, default="ROS2-Kinova-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)

# parse the arguments
args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")

BxN_Obs = Float[torch.Tensor, "b n"]
"""Environment observation: torch.Tensor[(b, n), float]"""
BxM_Action = Float[torch.Tensor, "b m"]
"""Environment action: torch.Tensor[(b, m), float]"""


def main() -> None:
    """Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None

    """
    env_cfg = parse_ros2_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        ros2_workspace=args_cli.ros2_ws,
    )
    env_cfg.robot_ip = "192.168.1.10"
    env_cfg.use_fake_hardware = "true"
    env_cfg.launch_ros = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    # Set up Skill executor and environment in framework
    env = ROS2EnvWrapper[BxN_Obs, BxM_Action](env)

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None

        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None

        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None

        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # Always active for other devices
    teleoperation_active = True  # TODO maybe switch to False
    sensitivity = args_cli.sensitivity
    # Create teleop device from config if present, otherwise create manually
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    # Add callbacks to fallback device
    for key, callback in teleoperation_callbacks.items():
        try:
            teleop_interface.add_callback(key, callback)
        except (ValueError, TypeError) as e:
            raise Exception(f"Failed to add callback for key {key}: {e}")

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            # get device command
            action = teleop_interface.advance()

            # Only apply teleop commands when active
            if teleoperation_active:
                # process actions
                actions = action.repeat(env.num_envs, 1)
                # apply actions
                env.step(actions)
            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                print("Environment reset complete")


if __name__ == "__main__":
    # run the main function
    main()
