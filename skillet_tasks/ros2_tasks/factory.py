from typing import Any

import rclpy

from skillet.envs.ros2 import Ros2Env
from skillet_tasks.ros2_tasks.gen3.gen3_ros2 import Gen3Ros2Env, Gen3Ros2EnvCfg
from skillet_tasks.ros2_tasks.gen3_lite.gen3lite_ros2 import Gen3LiteRos2Env, Gen3LiteRos2EnvCfg


def create_ros2_env(task_name: str, cfg: dict[str, Any]) -> Ros2Env:
    """Create a ROS2 environment for the given task name and configuration."""
    rclpy.init()
    if task_name == "Ros2-Gen3-v0":
        env_cfg = Gen3Ros2EnvCfg(**cfg)
        return Gen3Ros2Env(cfg=env_cfg)
    if task_name == "Ros2-Gen3Lite-v0":
        env_cfg = Gen3LiteRos2EnvCfg(**cfg)
        return Gen3LiteRos2Env(cfg=env_cfg)

    raise ValueError(f"Invalid task name: {task_name}")
