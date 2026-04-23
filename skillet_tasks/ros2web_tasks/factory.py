from typing import Any

from skillet.envs.ros2 import Ros2Env
from skillet.envs.ros2.websocket.ros_bridge import setup_ros
from skillet_tasks.ros2_tasks.gen3_lite.gen3lite_ros2 import Gen3LiteRos2WebEnv, Gen3LiteRos2WebEnvCfg
from skillet_tasks.ros2web_tasks.gen3.gen3_ros2 import Gen3Ros2WebEnv, Gen3Ros2WebEnvCfg


def create_ros2web_env(task_name: str, cfg: dict[str, Any]) -> Ros2Env:
    """Create a ROS2 environment for the given task name and configuration."""
    ros = setup_ros()
    if task_name == "Ros2-Gen3-v0":
        env_cfg = Gen3Ros2WebEnvCfg(**cfg)
        return Gen3Ros2WebEnv(cfg=env_cfg, ros=ros)
    if task_name == "Ros2-Gen3Lite-v0":
        env_cfg = Gen3LiteRos2WebEnvCfg(**cfg)
        return Gen3LiteRos2WebEnv(cfg=env_cfg, ros=ros)

    raise ValueError(f"Invalid task name: {task_name}")
