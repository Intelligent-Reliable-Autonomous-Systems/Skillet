from typing import Any

from skillet.envs.ros2 import ROS2Env
from skillet.envs.util import setup_ros
from skillet_tasks.ros2_tasks.gen3.gen3_ros2 import Gen3ROS2Env, Gen3ROS2EnvCfg
from skillet_tasks.ros2_tasks.gen3_lite.gen3lite_ros2 import Gen3LiteROS2Env, Gen3LiteROS2EnvCfg

def create_ros2_env(task_name: str, cfg: dict[str, Any]) -> ROS2Env:
    """Create a ROS2 environment for the given task name and configuration."""
    ros = setup_ros()
    if task_name == "ROS2-Gen3-v0":
        env_cfg = Gen3ROS2EnvCfg(**cfg)
        return Gen3ROS2Env(cfg=env_cfg, ros=ros)
    if task_name == "ROS2-Gen3Lite-v0":
        env_cfg = Gen3LiteROS2EnvCfg(**cfg)
        return Gen3LiteROS2Env(cfg=env_cfg, ros=ros)

    raise ValueError(f"Invalid task name: {task_name}")
