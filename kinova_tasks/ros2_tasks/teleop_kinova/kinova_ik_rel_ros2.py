"""kinova_ik_rel_ros.py.

Kinova Arm class Relative Inverse Kinematics Control.

Assumes that the action space is size (7,) for XYZ RPY + Gripper

Written by Will Solow, 2026

"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from roslibpy import Ros

from skillet.controllers import DifferentialIKController
from skillet.core.math import (
    convert_quat,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)
from skillet.envs.ros2 import (
    ROS2RLEnvCfg,
)
from skillet.envs.util import configclass

from ..kinova.kinova_ros2 import KinovaROS2Env


@configclass
class TeleOpKinovaROS2EnvCfg(ROS2RLEnvCfg):
    """The configuration class for Kinova Gen3 Arm."""

    """Robot configuration"""

    # Whether to spin up real robot or not
    use_fake_hardware = "true"

    # IP of the robot
    robot_ip = "www.xxx.yyy.zzz"

    # Visualize
    vision = False

    # Default joint position of robot
    default_joint_positions = [0.0, 0.523599, 0.0, 1.5708, 0.0, 0.785398, 0.0, 0.0]  # Double format for ROS 2

    """RL environment configuration"""
    num_envs = 1

    device = "cuda"

    dt = 1 / 60

    decimation = 6.0

    episode_length_s = 5.0

    skills = ["reach_xyz"]

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["robotiq_85_left_knuckle_joint"]


class KinovaROS2IKRelEnv(KinovaROS2Env):
    """Relative inverse kinematics control assuming a 7 DoF action space (delta xyz, delta rpy + gripper)."""

    def __init__(self, cfg: ROS2RLEnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
        cfg.episode_length_s = 10e12  # Basically make it so no resets
        cfg.decimation = 6.0
        cfg.dt = 1 / 60

        super().__init__(cfg, ros, render_mode=render_mode, **kwargs)
        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(7,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.ik_controller = DifferentialIKController(self.device, command_type="pose", use_relative_mode=True)

    def _reset_idx(self) -> None:
        """Reset environment based on specified indices to default position."""
        super()._reset_idx()
        #self._publish_action_to_robot(self.default_joint_positions, duration=8)
        #time.sleep(8)
        self.ik_controller.reset(n_envs=self.num_envs)

    def _pre_process_action(self, actions: torch.Tensor) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).

        """
        ee_pose_b = self._get_ee_pose_b()
        self.ik_controller.set_command(actions[:, :6], ee_pos=ee_pose_b[:, 0:3], ee_quat=ee_pose_b[:, 3:7])
        joint_pos = self.ik_controller.compute(
            ee_pos=ee_pose_b[:, 0:3],
            ee_quat=ee_pose_b[:, 3:7],
            jacobian=self._get_jacobians(),
            joint_pos=torch.as_tensor(
                self._current_joint_positions[self.cfg.joint_ids[:-1]], device=self.device
            ).unsqueeze(0),
        )
        self.actions = torch.cat((joint_pos, actions[:, -1:]), dim=-1).squeeze()

        return self.actions.cpu().numpy()

    def _get_jacobians(
        self,
    ) -> torch.Tensor:
        """Return the jacobians.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        arm_joint_ids = self.cfg.joint_ids[:-1]

        ee_link_idx = self._find_link_idx(self.cfg.ee_link_name)
        base_link_idx = self._find_link_idx(self.cfg.base_link_name)

        robot_base_pose_w = torch.as_tensor(self._robot_body_pose_w, device=self.device, dtype=torch.float32).unsqueeze(
            0
        )[:, base_link_idx]

        # Have to convert quaternion from ROS format (x,y,z,w) to IsaacLab format (w,x,y,z)
        robot_base_pose_w[:, 3:7] = convert_quat(robot_base_pose_w[:, 3:7], to="wxyz")
        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))

        jacobian = torch.as_tensor(self._jacobians, device=self.device, dtype=torch.float32)
        jacobian = jacobian.unsqueeze(0)[:, ee_link_idx][:, :, arm_joint_ids]

        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        return jacobian.to(torch.float32)

    def _get_ee_pose_b(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the end effector pose of the robot in the robot's base frame.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm

        Returns:
            The robot EE position in shape (N, 3) relative to the base of the robot
            The robot EE orientation in shape (N, 4) relative to the base of the robot

        """
        ee_link_idx = self._find_link_idx(self.cfg.ee_link_name)
        base_link_idx = self._find_link_idx(self.cfg.base_link_name)

        # Get the pose of the end effector and base in the world frame
        # (B, 7) with (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
        robot_ee_pose_w = torch.as_tensor(
            self._robot_body_pose_w[ee_link_idx], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        robot_base_pose_w = torch.as_tensor(
            self._robot_body_pose_w[base_link_idx], device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        # Have to convert quaternion from ROS format (x,y,z,w) to IsaacLab format (w,x,y,z)
        robot_ee_pose_w[:, 3:7] = convert_quat(robot_ee_pose_w[:, 3:7], to="wxyz")
        robot_base_pose_w[:, 3:7] = convert_quat(robot_base_pose_w[:, 3:7], to="wxyz")

        # Compute the end effector pose in the robot base frame
        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
            robot_base_pose_w[:, :3],
            robot_base_pose_w[:, 3:7],
            robot_ee_pose_w[:, :3],
            robot_ee_pose_w[:, 3:7],
        )

        return torch.cat((robot_ee_pos_b, robot_ee_quat_b), dim=1).to(torch.float32)
