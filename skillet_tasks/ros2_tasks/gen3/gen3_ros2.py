"""gen3_ros2.py.

Gen3 Arm class for ROS2 RL

Written by Will Solow, 2026

"""

import math
import pathlib
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from skillet.core.spaces import ActionSpec
from skillet.envs.ros2 import (
    Ros2Env,
    Ros2EnvCfg,
)
from skillet.envs.util import configclass
from skillet.perception.localization import RealsenseCameraLocalizer


@configclass
class Gen3Ros2EnvCfg(Ros2EnvCfg):
    """The configuration class for Kinova Gen3 Arm."""

    """Robot configuration"""
    urdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3/gen3_2f85.urdf"
    urdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3/gen3_2f85.srdf"
    assets_dir = [f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3/"]

    # Default joint position of robot
    default_joint_positions = [0.0, 0.523599, 0.0, 1.5708, 0.0, 0.785398, 0.0, 0.0]  # Double format for ROS 2

    """RL environment configuration"""
    num_envs = 1

    device = "cuda"

    dt = 1 / 60

    decimation = 1.0

    episode_length_s = 1e9

    skills = ["reach_xyz"]

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

    gripper_cmd_topic = "/robotiq_gripper_controller/gripper_cmd"

    move_group_name = "manipulator"

    tool_frame_name = "tcp"

    base_apriltag_id = 3


class Gen3Ros2Env(Ros2Env):
    """Kinova Gen3 7DoF ROS2 implementation."""

    def __init__(self, cfg: Ros2EnvCfg, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize Gen3 Arm ROS2."""
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_names = np.asarray(self.cfg.arm_joint_names + self.cfg.gripper_joint_names)
        self.arm_joint_names = np.asarray(self.cfg.arm_joint_names)
        self.gripper_joint_names = np.asarray(self.cfg.gripper_joint_names)

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            float("-inf"), float("inf"), shape=(2 * len(self.joint_names),)
        )
        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(len(self.cfg.joint_ids),))

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self._current_joint_positions = np.zeros(shape=len(self.joint_names))
        self._current_joint_velocities = np.zeros(shape=len(self.joint_names))
        self._current_joint_efforts = np.zeros(shape=len(self.joint_names))
        self.prev_action = np.zeros(shape=(len(self.joint_names),))

        self.active_controller = "joint_trajectory_controller"

        self._rs_cam_localizer = RealsenseCameraLocalizer(apriltag_size_m=0.1, apriltag_id=self.cfg.base_apriltag_id)

        self._curr_gripper_goal = None
        self._new_gripper_goal = False
        self._gripper_goal_start = None
        self._blocking_gripper_cmd = True

    def _pre_process_action(self, actions: torch.Tensor, action_spec: ActionSpec[Any] | None = None) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).
            action_spec: The action specification of the batch of actions. Note: Currently assumes all actions are
                of same spec.

        """
        self.actions = actions.cpu().numpy().squeeze()  # Actions can only be 1 dimensional

        return self.actions

    def _publish_action_to_ros(
        self, action: np.ndarray, action_spec: ActionSpec[Any] | None = None, duration: float = 3, timeout: float = 30
    ) -> None:
        """Publish the robot action.

        Args:
            action: NDArray of joint positions
            action_spec: Action specficiation of each of the actions
            duration: Duration of trajectory
            timeout: duration of time out

        """
        # Publish BLOCKING gripper command. To keep the gripper stationary
        # Assumes we can either move joints or close gripper, not both
        gripper_moving = self._publish_gripper(action, action_spec, close_time=2.0)

        if not gripper_moving or not self._blocking_gripper_cmd:
            if action_spec is None or action_spec.name == "joints_vel":
                self._publish_joint_vel_spec(action, duration)
            elif action_spec.name == "twist_tcp":
                self._publish_twist_tcp_spec(action)
            else:
                raise ValueError(f"Unknown Action Specification `{action_spec.name}`")

    def _reset_idx(self) -> None:
        """Reset environment based on specified indices to default position."""
        super()._reset_idx()

    def _get_dones(self) -> tuple[bool, bool]:
        """Return dones if longer than max episode length."""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return False, truncated

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Return the observations from the robot."""
        return {
            "policy": np.concatenate(
                (self._current_joint_positions, self._current_joint_velocities), axis=0, dtype=np.float32
            )
        }

    def _get_rewards(self) -> np.ndarray:
        """Compute the rewards."""
        return np.array([0.0])

    def _supports_action_spec(self, action_spec: ActionSpec[Any] | None = None) -> bool:
        if action_spec is None:
            return True
        return action_spec.name in [s.name for s in self._action_specs]

    def _get_latest_rgbd(self) -> dict[str, Any]:
        """Grab the latest RGB-D snapshot in the raw Realsense format.

        Returns:
            A dictionary containing:
              - ``rgb``: (H, W, 3) uint8 RGB image
              - ``depth``: (H, W) uint16 depth image
              - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
              - ``camera_pose``: 7D float64 array (x, y, z, qx, qy, qz, qw) in ROS xyzw
              - ``timestamp``: float timestamp in seconds

        """
        latest = self._rs_cam_localizer._get_latest_rgbd_raw()
        # Realsense xyzw format -> IsaacLab wxyz format
        q = latest["camera_pose"][3:7]
        latest["camera_pose"][3:7] = q[[3, 0, 1, 2]]
        # RGB is (H, W, 3) -> (3, H, W)
        latest["rgb"] = latest["rgb"].transpose((2, 0, 1))
        # Depth is (H, W) -> (1, H, W), always float32 meters.
        depth = np.expand_dims(latest["depth"], axis=0)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        latest["depth"] = depth
        return latest

    def _publish_gripper(
        self,
        joint_pos: np.ndarray,
        action_spec: ActionSpec,
        timeout: int = 30,
        duration: int = 3,
        close_time: float = 0.5,
    ) -> None:
        """Publish a joint position to the joint trajectory controller.

        Args:
            joint_pos: Joint positions of the robot.
            action_spec: what stationary action to specify
            timeout: Duration before timeout
            duration: what duration to specify
            close_time: time the gripper takes to close

        """
        new_gripper_goal = False

        gripper_val = float(joint_pos[-1])
        gripper_val = max(0, min(gripper_val, 1)) * 0.8

        if gripper_val != self._curr_gripper_goal:
            if action_spec is None or action_spec.name == "joints_vel":
                self._publish_joint_vel_spec(np.zeros_like(joint_pos), duration)
            elif action_spec.name == "twist_tcp":
                self._publish_twist_tcp_spec(np.zeros_like(joint_pos))
            new_gripper_goal = True
            self._ros2_listener.send_gripper_goal(gripper_val)
            self._curr_gripper_goal = gripper_val
            self._new_gripper_goal = True
            self._gripper_goal_start = time.perf_counter()
        elif (time.perf_counter() - self._gripper_goal_start) < close_time:
            self._new_gripper_goal = True
        else:
            self._new_gripper_goal = False

        return new_gripper_goal

    def _publish_joint_vel_spec(self, joint_pos: np.ndarray, duration: float = 3) -> None:
        """Publish a joint position to the joint trajectory controller.

        Args:
            joint_pos: Joint positions of the robot.
            duration: Duration of trajectory

        """
        if self.active_controller != "joint_trajectory_controller":
            if not self.switch_controllers(activate=["joint_trajectory_controller"], deactivate=["twist_controller"]):
                print("[INFO] Unable to switch controller to `joint_trajectory_controller`. Aborting trajectory.")
                return
            self.active_controller = "joint_trajectory_controller"
            print("[INFO] Successfully switched controller to `joint_trajectory_controller`")

        joint_traj = JointTrajectory()
        joint_traj.joint_names = self.cfg.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_pos[: len(self.cfg.arm_joint_names)]
        point.time_from_start.secs = math.floor(duration)
        point.time_from_start.nsecs = int((duration - math.floor(duration)) * 10e9)
        joint_traj.points = [point]

        self.joint_traj_pub.publish(joint_traj)

    def _publish_twist_tcp_spec(self, twist: np.ndarray) -> None:
        """Publish a twist in the tcp frame to the robot twist controller.

        Args:
            twist: Cartesian twist command (velocity) in XYZ (linear) and RPY (angular).

        """
        if self.active_controller != "twist_controller":
            print("[INFO] Switching controller to `twist_controller`")
            if not self._ros2_listener.switch_controllers(
                activate=["twist_controller"], deactivate=["joint_trajectory_controller"]
            ):
                print("[INFO] Unable to switch controller to `twist_controller`. Aborting trajectory.")
                return
            self.active_controller = "twist_controller"

            print("[INFO] Successfully switched controller to `twist_controller`")

        self._ros2_listener.twist_cmd = twist
