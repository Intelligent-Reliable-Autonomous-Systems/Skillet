"""Init file for env."""

from .env_cfg import EnvCfg as EnvCfg

# Conditional import
ISAAC_ENABLED = False
ROS2_ENABLED = True

if ISAAC_ENABLED:
    from .isaac_env_wrapper import IsaacEnvWrapper as IsaacEnvWrapper

if ROS2_ENABLED:
    from .ros2_env_wrapper import ROS2EnvWrapper as ROS2EnvWrapper
