"""gen3lite_ros2.py.

Gen3 Arm class for ROS2 RL

Written by Will Solow, 2026

"""

import pathlib
from typing import Any

from skillet.envs.ros2 import Ros2EnvCfg
from skillet.envs.util import configclass
from skillet_tasks.ros2_tasks.gen3.gen3_ros2 import Gen3Ros2Env


@configclass
class Gen3LiteRos2EnvCfg(Ros2EnvCfg):
    """The configuration class for Kinova Gen3 Lite Arm."""

    """Robot configuration"""
    urdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3lite/gen3_lite.urdf"
    srdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3lite/gen3_lite.srdf"
    assets_dir = [f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3lite/"]

    # Default joint position of robot
    default_joint_positions = [0.2, -0.18, 2.16, -1.57, -0.6, -1.34, 0.0]  # Double format for ROS 2

    """RL environment configuration"""
    num_envs = 1

    device = "cuda"

    dt = 1 / 60

    decimation = 1

    episode_length_s = 1e9

    skills = ["reach_xyz"]

    joint_ids = [0, 1, 2, 3, 4, 5, 6]
    tcp_offset = [0.0, 0.0, 0.130, 0.70710678, 0, 0, 0.70710678]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["left_finger_bottom_joint"]
    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    gripper_cmd_topic = "/gen3_lite_2f_gripper_controller/gripper_cmd"

    move_group_name = "arm"

    tool_frame_name = "tool_frame"


class Gen3LiteRos2Env(Gen3Ros2Env):
    """Kinova Gen3Lite 6DoF ROS2 implementation."""

    def __init__(self, cfg: Gen3LiteRos2EnvCfg, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize Gen3Lite Arm ROS2."""
        super().__init__(cfg, render_mode, **kwargs)
