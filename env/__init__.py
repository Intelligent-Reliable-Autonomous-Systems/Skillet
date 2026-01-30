"""Init file for env."""

from collections.abc import Callable

from .env_cfg import EnvCfg as EnvCfg


def import_isaac_wrapper() -> Callable:
    """Lazy import of the isaac wrapper"""
    try:
        from .isaac_env_wrapper import IsaacEnvWrapper

        return IsaacEnvWrapper
    except ImportError:
        raise RuntimeError("ISAAC not available")


def import_ros2_wrapper() -> Callable:
    """Lazy import of the ROS2 wrapper"""
    try:
        from .ros2_env_wrapper import ROS2EnvWrapper

        return ROS2EnvWrapper
    except ImportError:
        raise RuntimeError("ISAAC not available")
