"""ros2_rl_env.py.

Main ROS2 RL Env Runner

Written by Will Solow, 2026

"""

from dataclasses import MISSING
from cfg import configclass


@configclass
class ROS2RLEnvCfg:
    """The configuratino class for ROS2 RL Envs."""

    use_fake_hardware: bool = MISSING
