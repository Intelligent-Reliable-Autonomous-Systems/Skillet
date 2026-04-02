"""gen3_kortex.py.

Gen3 Arm class using Kortex base

Written by Will Solow, 2026

"""

import pathlib
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from kortex_api.autogen.messages import Base_pb2

from skillet.core.math import np_euler_xyz_degrees_from_quat
from skillet.core.spaces import ActionSpec
from skillet.envs.kortex import KortexEnv, KortexEnvCfg
from skillet.envs.kortex.kortex_bridge import DeviceConnection, check_for_end_or_abort
from skillet.envs.util import configclass
from skillet.perception.realsense import RealsenseCameraLocalizer
from skillet.policy.specs import JOINTS_SPEC


@configclass
class Gen3KortexEnvCfg(KortexEnvCfg):
    """The configuration class for Kinova Gen3 Arm."""

    """Robot configuration"""

    urdf_path = f"{pathlib.Path.cwd()}/skillet_tasks/assets/kortex/kinova_gen3/gen3_2f85.urdf"

    # IP of the robot
    robot_ip = "www.xxx.yyy.zzz"

    # Visualize
    vision = False

    # Default joint position of robot
    default_joint_positions = [0.0, 0.523599, 0.0, 1.5708, 0.0, 0.785398, 0.0, 0.0]  # Double format for Kortex

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

    base_apriltag_id = 3


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

        self._action_specs = [
            JOINTS_SPEC.bind(
                n_joints=len(self.cfg.joint_ids),
            ).replace(device=self.device),
            ActionSpec(
                name="twist_tcp",
                space=gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(6 + len(self.cfg.gripper_joint_names),)
                ),
                is_torch=True,
                is_batched=True,
                n_envs=-1,
                device=self.device,
            ),
        ]

        self._current_joint_positions = np.zeros(shape=len(self.joint_names))
        self._current_joint_velocities = np.zeros(shape=len(self.joint_names))
        self._current_joint_efforts = np.zeros(shape=len(self.joint_names))

        self.curr_gripper_goal = None

        self._rs_cam_localizer = RealsenseCameraLocalizer(apriltag_size_m=0.1, apriltag_id=self.cfg.base_apriltag_id)

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
        if self._publish_gripper(action, action_spec, close_time=2.0):
            return

        if action_spec is None or action_spec.name == "joints":
            self._publish_joint_spec(action, duration)
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
        return self._rs_cam_localizer._get_latest_rgbd_raw()

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
        gripper_goal = max(0, min(gripper_val, 1))

        if gripper_goal != self.curr_gripper_goal:
            gripper_command = Base_pb2.GripperCommand()
            finger = gripper_command.gripper.finger.add()
            gripper_command.mode = Base_pb2.GRIPPER_POSITION
            finger.finger_identifier = 1
            finger.value = gripper_goal

            if action_spec is None or action_spec.name == "joints":
                self._publish_joint_spec(joint_pos, duration)
            elif action_spec.name == "twist_tcp":
                self._publish_twist_tcp_spec(np.zeros_like(joint_pos))

            self.kortex.SendGripperCommand(gripper_command)

            self.curr_gripper_goal = gripper_goal
            new_gripper_goal = True
            time.sleep(close_time)

        return new_gripper_goal

    def _publish_joint_spec(self, joint_pos: np.ndarray, duration: float = 20) -> None:
        """Publish a joint position trajectory to the kortex API.

        Args:
            joint_pos: Joint positions of the robot.
            duration: Max duration of trajectory

        """
        if self.active_controller != Base_pb2.SINGLE_LEVEL_SERVOING:
            print("[INFO] Switching controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")
            if not self.switch_controllers(active_controller=Base_pb2.SINGLE_LEVEL_SERVOING):
                print("[INFO] Unable to switch controller to `Base_pb2.SINGLE_LEVEL_SERVOING`. Aborting trajectory.")
                return
            self.active_controller = Base_pb2.SINGLE_LEVEL_SERVOING
            print("[INFO] Successfully switched controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")

        print("Starting angular action movement ...")
        action = Base_pb2.Action()
        action.name = "Joint Trajectory"
        action.application_data = ""

        # Place arm straight up
        for joint_id in range(len(self.arm_joint_names)):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = joint_pos[joint_id]

        e = threading.Event()
        notification_handle = self.kortex.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        self.kortex.ExecuteAction(action)
        finished = e.wait(duration)
        self.kortex.Unsubscribe(notification_handle)

    def _publish_tcp_spec(self, tcp_pose: np.ndarray, duration: float = 20) -> None:
        """Publish a TCP position to the Kortex API to be solved with its own motion planning.

        Args:
            tcp_pose: Goal TCP position of the robot
            duration: Max duration of the trajectory

        """
        if self.active_controller != Base_pb2.SINGLE_LEVEL_SERVOING:
            print("[INFO] Switching controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")
            if not self.switch_controllers(active_controller=Base_pb2.SINGLE_LEVEL_SERVOING):
                print("[INFO] Unable to switch controller to `Base_pb2.SINGLE_LEVEL_SERVOING`. Aborting trajectory.")
                return
            self.active_controller = Base_pb2.SINGLE_LEVEL_SERVOING
            print("[INFO] Successfully switched controller to `Base_pb2.SINGLE_LEVEL_SERVOING`")

        action = Base_pb2.Action()
        action.name = "TCP Pose"
        action.application_data = ""

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = tcp_pose[0]
        cartesian_pose.y = tcp_pose[1]
        cartesian_pose.z = tcp_pose[2]
        roll, pitch, yaw = np_euler_xyz_degrees_from_quat(tcp_pose[3:7])
        cartesian_pose.theta_x = roll
        cartesian_pose.theta_y = pitch
        cartesian_pose.theta_z = yaw

        e = threading.Event()
        notification_handle = self.kortex.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        self.kortex.ExecuteAction(action)

        finished = e.wait(duration)
        self.kortex.Unsubscribe(notification_handle)

    def _publish_twist_tcp_spec(self, twist: np.ndarray) -> None:
        """Publish a twist in the tcp frame to the robot twist controller.

        Args:
            twist: Cartesian twist command (velocity) in XYZ (linear) and RPY (angular).

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
        command.duration = 1
        twist_cmd = command.twist
        twist_cmd.linear_x = twist[0]
        twist_cmd.linear_y = twist[1]
        twist_cmd.linear_z = twist[2]
        twist_cmd.angular_x = twist[3]
        twist_cmd.angular_y = twist[4]
        twist_cmd.angular_z = twist[5]

        self.kortex.SendTwistCommand(command)
