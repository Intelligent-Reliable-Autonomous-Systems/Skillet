"""ros2_rl_env_cfg.py.

Main configuration for ROS2 RL Env CFG file

Written by Will Solow, 2026

"""

from dataclasses import MISSING

from cfg import configclass


@configclass
class ROS2RLEnvCfg:
    """The configuration class for ROS2 RL Envs."""

    """Robot configuration"""
    # Workspace which bringup configuration is defined
    ros2_workspace: str = MISSING

    # Whether to spin up real robot or not
    use_fake_hardware: str = "true"

    # IP of the robot
    robot_ip: str = "www.xxx.yyy.zzz"

    # Visualize
    vision: str = "false"

    # Default joint position of robot
    default_joint_positions: list[float] = MISSING

    """RL environment configuration"""
    num_envs: int = 1

    device: str = "cuda"

    dt: float = 1 / 60

    decimation: float = 1.0

    episode_length_s: float = 5.0
