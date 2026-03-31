"""gen3lite_kortex.py.

Gen3 Arm class for Kortex API

Written by Will Solow, 2026

"""

import pathlib
from typing import Any

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from skillet.envs.kortex.kortex_bridge import DeviceConnection

from skillet.envs.kortex import (
    KortexEnvCfg,
)
from skillet.envs.util import configclass
from skillet_tasks.kortex_tasks.gen3.gen3_kortex import Gen3KortexEnv


@configclass
class Gen3LiteKortexEnvCfg(KortexEnvCfg):
    """The configuration class for Kinova Gen3 Lite Arm."""

    """Robot configuration"""

    # IP of the robot
    robot_ip = "192.168.1.10"

    urdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3lite/gen3_lite.urdf"

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
    gripper_joint_names = ["left_finger_bottom_joint"]
    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    base_apriltag_id = 3


class Gen3LiteKortexEnv(Gen3KortexEnv):
    """Kinova Gen3Lite 6DoF ROS2 implementation."""

    def __init__(
        self,
        cfg: Gen3LiteKortexEnvCfg,
        kortex_connection: DeviceConnection,
        render_mode: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize Gen3Lite Arm ROS2."""
        super().__init__(cfg, kortex_connection=kortex_connection, render_mode=render_mode, **kwargs)
