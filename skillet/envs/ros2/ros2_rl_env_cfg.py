"""ros2_rl_env_cfg.py.

Main configuration for ROS2 RL Env CFG file

Written by Will Solow, 2026

"""

from dataclasses import MISSING

from skillet.envs.util import configclass


@configclass
class ROS2RLEnvCfg:
    """The configuration class for ROS2 RL Envs."""

    """Robot configuration"""
    # Workspace which bringup configuration is defined
    ros2_workspace: str = MISSING

    # Whether to spin up real robot or not
    use_fake_hardware: str = "true"

    launch_ros: bool = True

    # IP of the robot
    robot_ip: str = "www.xxx.yyy.zzz"

    # Visualize
    vision: str = "false"

    # Default joint position of robot
    default_joint_positions: list[float] = MISSING

    """RL environment configuration"""
    num_envs: int = 1
    """Number of parallel environments, will always be 1 for ROS"""

    device: str = "cuda"
    """GPU device"""

    dt: float = 2
    """Delta time per step"""

    decimation: float = 1.0
    """Decimation (steps through physics)"""

    episode_length_s: float = 5.0
    """Episode length in seconds"""

    skills: list[str] | None = None
    """List of behavior primitives available"""

    seed: int = MISSING
    """Seed for the environment"""

    is_finite_horizon: bool = False
    """Whether learning is treated as a finite or infinite horizon problem"""
