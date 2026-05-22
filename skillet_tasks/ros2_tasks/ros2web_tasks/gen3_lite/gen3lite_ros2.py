"""gen3_ros2.py.

Gen3 Arm class for ROS2 RL

Written by Will Solow, 2026

"""

from typing import Any

from roslibpy import Ros

from skillet.envs.ros2.websocket import Ros2WebEnvCfg
from skillet.envs.util import configclass
from skillet_tasks.ros2_tasks.ros2web_tasks.gen3.gen3_ros2 import Gen3Ros2WebEnv


@configclass
class Gen3LiteRos2WebEnvCfg(Ros2WebEnvCfg):
    """The configuration class for Kinova Gen3 Lite Arm."""

    """Robot configuration"""

    # IP of the robot
    robot_ip = "192.168.1.10"

    # Visualize
    vision = False

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
    gripper_joint_names = ["right_finger_bottom_joint"]
    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    gripper_cmd_topic = "/gen3_lite_2f_gripper_controller/gripper_cmd"

    move_group_name = "arm"

    tool_frame_name = "tool_frame"


class Gen3LiteRos2WebEnv(Gen3Ros2WebEnv):
    """Kinova Gen3Lite 6DoF ROS2 implementation."""

    def __init__(
        self, cfg: Gen3LiteRos2WebEnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """Initialize Gen3Lite Arm ROS2."""
        super().__init__(cfg, ros, render_mode, **kwargs)
