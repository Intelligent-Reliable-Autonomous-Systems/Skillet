"""gen3_kortex.py.

Gen3 Arm class using Kortex base

Written by Will Solow, 2026

"""

import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from kortex_api.autogen.messages import Base_pb2
from kortex_api.Exceptions.KServerException import KServerException

from skillet.core.spaces import ActionSpec
from skillet.envs.kortex import KortexEnv, KortexEnvCfg
from skillet.envs.kortex.kortex_bridge import DeviceConnection
from skillet.envs.util import configclass
from skillet.perception.localization import RealsenseCameraLocalizer


@configclass
class Gen3KortexEnvCfg(KortexEnvCfg):
    """The configuration class for Kinova Gen3 Arm."""

    """Robot configuration"""

    urdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3/gen3_2f85.urdf"
    srdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3/gen3_2f85.srdf"
    assets_dir = [f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3/"]
    # IP of the robot
    robot_ip = "www.xxx.yyy.zzz"

    # Visualize
    vision = False

    # Default joint position of robot
    default_joint_positions = [0.0, 0.523599, 0.0, 1.5708, 0.0, 0.785398, 0.0, 0.0]  # Double format for Kortex

    """RL environment configuration"""
    num_envs = 1

    device = "cuda"

    dt = 3 / 60

    decimation = 1.0

    episode_length_s = 1e9

    skills = ["reach_xyz"]

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    tcp_offset = [0.0, 0.0, 0.12, 0.0, 0.7071, -0.7071, 0.0]
    # tcp_offset = [0.0, 0.0, 0.12, 0.70710678, 0, 0, 0.70710678]
    # tcp_offset = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]

    ee_link_name = "end_effector_link"

    base_link_name = "base_link"

    gripper_joint_names = ["robotiq_85_left_inner_knuckle_joint"]

    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]


class Gen3KortexEnv(KortexEnv):
    """Kinova Gen3 7DoF Kortex implementation."""

    def __init__(
        self,
        cfg: KortexEnvCfg,
        kortex_connection: DeviceConnection,
        render_mode: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize Gen3 Arm Kortex."""
        super().__init__(cfg, kortex_connection=kortex_connection, render_mode=render_mode, **kwargs)

        self.joint_names = np.asarray(self.cfg.arm_joint_names + self.cfg.gripper_joint_names)
        self.arm_joint_names = np.asarray(self.cfg.arm_joint_names)
        self.gripper_joint_names = np.asarray(self.cfg.gripper_joint_names)

        self.active_controller = Base_pb2.SINGLE_LEVEL_SERVOING

        self.default_joint_positions = np.asarray(self.cfg.default_joint_positions)
        self.prev_action = np.zeros(shape=(len(self.joint_names),))

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
        self._current_prev_actions = np.zeros(shape=(7,))  # TODO find out how to change dynamically

        self._curr_gripper_goal = None
        self._new_gripper_goal = False
        self._gripper_goal_start = None
        self._blocking_gripper_cmd = True

        self._curr_motion_goal = None
        self._new_motion_goal = False
        self._motion_goal_start = None
        self._motion_event = None
        self._motion_handle = None

        if self.cfg.use_tabletop_camera:
            self._rs_cam_localizer = RealsenseCameraLocalizer(
                apriltag_size_m=self.cfg.base_apriltag_size,
                apriltag_id=self.cfg.base_apriltag_id,
                apriltag_pose=np.asarray(self.cfg.base_apriltag_pose),
                apriltag_fam=self.cfg.base_apriltag_fam,
            )

    def _pre_process_action(self, actions: torch.Tensor, action_spec: ActionSpec[Any] | None = None) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).
            action_spec: The action specification of the batch of actions. Note: Currently assumes all actions are
                of same spec.

        """
        self.actions = actions.cpu().numpy().squeeze()  # Actions can only be 1 dimensional
        self._current_prev_actions = self.actions.copy()
        return self.actions

    def _publish_action_to_kortex(
        self, action: np.ndarray, action_spec: ActionSpec[Any] | None = None, duration: float = 3, timeout: float = 30
    ) -> None:
        """Publish the robot action through the Kortex API.

        Args:
            action: NDArray of joint positions
            action_spec: Action specficiation of each of the actions
            duration: Duration of trajectory
            timeout: duration of time out

        """
        # Publish BLOCKING gripper command. To keep the gripper stationary
        # Assumes we can either move joints or close gripper, not both
        gripper_moving = self._publish_gripper(action, action_spec, close_time=self.cfg.gripper_close_time)
        if not gripper_moving or not self._blocking_gripper_cmd:
            if action_spec is None or action_spec.name == "joints_vel":
                self._publish_joint_vel_spec(action, duration)
            elif action_spec.name == "twist_tcp":
                self._publish_twist_tcp_spec(action)
            elif action_spec.name == "tcp_cart":
                self._publish_tcp_cart_spec(action)
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

    def _get_latest_rgbd(self) -> dict[str, Any]:
        """Grab the latest RGB-D snapshot in the raw Realsense format.

        Returns:
            A dictionary containing:
              - ``rgb``: (H, W, 3) uint8 RGB image
              - ``depth``: (H, W) uint16 depth image
              - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
              - (0.8, 0.1, 0.1, 1.0)
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
        duration: int = 3,
        close_time: float = 0.5,
        mode: Literal["position", "velocity"] = "position",
    ) -> None:
        """Publish a joint position to the joint trajectory controller.

        Args:
            joint_pos: Joint positions of the robot.
            action_spec: what stationary action to specify
            duration: what duration to specify
            close_time: time the gripper takes to close
            mode: The mode to operate the gripper in

        """
        gripper_val = float(joint_pos[-1])
        gripper_goal = max(0, min(gripper_val, 1))
        if mode == "position":
            if gripper_goal != self._curr_gripper_goal:
                gripper_command = Base_pb2.GripperCommand()
                finger = gripper_command.gripper.finger.add()
                gripper_command.mode = Base_pb2.GRIPPER_POSITION
                finger.finger_identifier = 1
                finger.value = gripper_goal

                if action_spec is None or action_spec.name == "joints_vel":
                    self._publish_joint_vel_spec(np.zeros_like(joint_pos), duration)
                elif action_spec.name == "twist_tcp":
                    self._publish_twist_tcp_spec(np.zeros_like(joint_pos))
                self.kortex.SendGripperCommand(gripper_command)
                self._curr_gripper_goal = gripper_goal
                self._new_gripper_goal = True
                self._gripper_goal_start = time.perf_counter()
            elif (time.perf_counter() - self._gripper_goal_start) < close_time:
                self._new_gripper_goal = True
            else:
                self._new_gripper_goal = False
        elif mode == "velocity":
            if gripper_goal != self._curr_gripper_goal:
                self._count = 0
                gripper_command = Base_pb2.GripperCommand()
                finger = gripper_command.gripper.finger.add()
                gripper_command.mode = Base_pb2.GRIPPER_SPEED
                finger.finger_identifier = 1
                finger.value = -0.2 if gripper_goal > self._current_joint_positions[-1] else 0.2
                self.kortex.SendGripperCommand(gripper_command)
                self._curr_gripper_goal = gripper_goal
                self._new_gripper_goal = True
                self._gripper_goal_start = time.perf_counter()
            elif (
                (time.perf_counter() - self._gripper_goal_start) > close_time
                or np.abs(gripper_goal - self._current_joint_positions[-1]) < 0.001
                or (self._count > 1 and self._current_joint_velocities[-1] < 0.05)
            ):
                gripper_command = Base_pb2.GripperCommand()
                finger = gripper_command.gripper.finger.add()
                gripper_command.mode = Base_pb2.GRIPPER_SPEED
                finger.finger_identifier = 1
                finger.value = 0.0
                self.kortex.SendGripperCommand(gripper_command)
                self._new_gripper_goal = False
                self._count = 0
            else:
                self._new_gripper_goal = True
                self._count += 1
        return self._new_gripper_goal

    def _publish_joint_vel_spec(self, joint_vel: np.ndarray, duration: float = 20) -> None:
        """Publish a joint velocity commmand to the kortex API.

        Args:
            joint_vel: Joint velocities of the robot.
            duration: Max duration of trajectory

        """
        if self.active_controller != Base_pb2.SINGLE_LEVEL_SERVOING:
            print("[INFO] Switching controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")
            if not self.switch_controllers(active_controller=Base_pb2.SINGLE_LEVEL_SERVOING):
                print("[INFO] Unable to switch controller to `Base_pb2.SINGLE_LEVEL_SERVOING`. Aborting trajectory.")
                return
            self.active_controller = Base_pb2.SINGLE_LEVEL_SERVOING
            print("[INFO] Successfully switched controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")

        command = Base_pb2.JointSpeeds()
        for i, j in enumerate(np.rad2deg(joint_vel[:-1])):  # Publish all except gripper
            joint_speed = command.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = j
            joint_speed.duration = 0

        self.kortex.SendJointSpeedsCommand(command)

    def _publish_twist_tcp_spec(self, twist: np.ndarray, max_retries=2) -> None:
        """Publish a twist in the tcp frame to the robot twist controller.

        Args:
            twist: Cartesian twist command (velocity) in XYZ (linear) and RPY (angular).
            max_retries: maximum number of times trying to reconnect to server.

        """
        if self.active_controller != Base_pb2.SINGLE_LEVEL_SERVOING:
            print("[INFO] Switching controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")
            if not self.switch_controllers(active_controller=Base_pb2.SINGLE_LEVEL_SERVOING):
                print("[INFO] Unable to switch controller to `Base_pb2.SINGLE_LEVEL_SERVOING`. Aborting trajectory.")
                return
            self.active_controller = Base_pb2.SINGLE_LEVEL_SERVOING
            print("[INFO] Successfully switched controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")

        command = Base_pb2.TwistCommand()

        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
        command.duration = 1  # must be int type
        twist_cmd = command.twist
        twist_cmd.linear_x = twist[0]
        twist_cmd.linear_y = twist[1]
        twist_cmd.linear_z = twist[2]
        twist_cmd.angular_x = twist[3]
        twist_cmd.angular_y = twist[4]
        twist_cmd.angular_z = twist[5]

        # for attempt in range(max_retries + 1):
        #     try:
        #     except KServerException as ex:
        #         error_code = ex.get_error_code()
        #         sub_error_code = ex.get_error_sub_code()
        #         if sub_error_code == 130:
        #             print(
        #                 f"[WARN][KORTEX] Lost control of session (attempt {attempt + 1}/{max_retries + 1}), "
        #                 "attempting to retake control..."
        #             )
        #             if not self._retake_control():
        #                 raise
        #             continue
        #         else:
        #             pass

    def _publish_tcp_cart_spec(self, tcp_cart: np.ndarray, duration: float = 5) -> None:
        """Publish a TCP cartesian trajectory to the kortex API.

        Args:
            tcp_cart: End effector position in XYZ RPY
            duration: Max duration of trajectory

        """
        if self.active_controller != Base_pb2.SINGLE_LEVEL_SERVOING:
            print("[INFO] Switching controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")
            if not self.switch_controllers(active_controller=Base_pb2.SINGLE_LEVEL_SERVOING):
                print("[INFO] Unable to switch controller to `Base_pb2.SINGLE_LEVEL_SERVOING`. Aborting trajectory.")
                return
            self.active_controller = Base_pb2.SINGLE_LEVEL_SERVOING
            print("[INFO] Successfully switched controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")

        action = Base_pb2.Action()
        action.name = "TCP Cartesian Action"
        action.application_data = ""
        if (tcp_cart[:6] != self._curr_motion_goal).any():
            self._curr_motion_goal = tcp_cart[:6]

            cartesian_pose = action.reach_pose.target_pose
            cartesian_pose.x = tcp_cart[0]
            cartesian_pose.y = tcp_cart[1]
            cartesian_pose.z = tcp_cart[2]
            cartesian_pose.theta_x = np.rad2deg(tcp_cart[3])
            cartesian_pose.theta_y = np.rad2deg(tcp_cart[4])
            cartesian_pose.theta_z = np.rad2deg(tcp_cart[5])
            speed = action.reach_pose.constraint.speed
            speed.translation = 0.1  # NOTE might be fast
            speed.orientation = 30

            self._motion_event = threading.Event()
            self._motion_handle = self.kortex.OnNotificationActionTopic(
                self._check_for_end_or_abort(self._motion_event), Base_pb2.NotificationOptions()
            )

            self.kortex.ExecuteAction(action)  # TODO can catch KServerException

            self._new_motion_goal = True
            self._gripper_goal_start = time.perf_counter()
        elif (time.perf_counter() - self._gripper_goal_start) < duration and not self._motion_event.is_set():
            self._new_motion_goal = True
        else:
            self._new_motion_goal = False

    def _check_for_end_or_abort(self, e: threading.Event) -> Callable:
        """Return a closure checking for END or ABORT notifications.

        Args:
            e: event to signal when the action is completed. (will be set when an END or ABORT occurs)

        """

        def check(notification: Base_pb2.ActionNotification, e: threading.Event = e) -> bool:
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()

        return check
