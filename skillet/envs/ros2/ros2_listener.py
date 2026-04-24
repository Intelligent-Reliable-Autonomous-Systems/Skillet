"""Class for listening to Ros2."""

import copy

import numpy as np
import rclpy
from control_msgs.action import ParallelGripperCommand
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from trajectory_msgs.msg import JointTrajectory

from .ros2_env_cfg import Ros2EnvCfg


class Ros2EnvListener(Node):
    _joint_positions: np.ndarray
    _joint_velocities: np.ndarray
    _joint_efforts: np.ndarray

    _twist_cmd: np.ndarray
    _joint_vel_cmd: np.ndarray

    def __init__(self, dt: float, cfg: Ros2EnvCfg):
        super().__init__("Ros2Env")
        self.cfg = copy.deepcopy(cfg)

        # Handle parsing joint names
        joint_names = self.cfg.arm_joint_names + self.cfg.gripper_joint_names
        if "left_finger_bottom_joint" in joint_names:
            joint_names.remove("left_finger_bottom_joint")
            joint_names.append("right_finger_bottom_joint")
        self.joint_names = np.asarray(joint_names)

        self._joint_sub = self.create_subscription(JointState, "/joint_states", self._robot_state_sub, 10)
        self._timer = self.create_timer(dt, self._joint_pub_callback)

        self.gripper_action_client = ActionClient(self, ParallelGripperCommand, self.cfg.gripper_cmd_topic)
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory, "/safety/joint_trajectory_controller/joint_trajectory", 10
        )
        self.joint_vel_pub = self.create_publisher(Float32MultiArray, "/forward_velocity_controller/commands", 10)
        self.twist_pub = self.create_publisher(Twist, "/safety/twist_controller/commands", 10)
        self.switch_ctrl_client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self._joint_positions = np.zeros(len(self.cfg.arm_joint_names + self.cfg.gripper_joint_names))
        self._joint_velocities = np.zeros(len(self.cfg.arm_joint_names + self.cfg.gripper_joint_names))
        self._joint_efforts = np.zeros(len(self.cfg.arm_joint_names + self.cfg.gripper_joint_names))

        self._twist_cmd = None
        self._joint_vel_cmd = None

    @property
    def twist_cmd(self) -> np.ndarray:
        return self._twist_cmd

    @property
    def joint_vel_cmd(self) -> np.ndarray:
        return self._joint_vel_cmd

    @twist_cmd.setter
    def twist_cmd(self, twist: np.ndarray) -> None:
        self._twist_cmd = twist

    @joint_vel_cmd.setter
    def joint_vel_cmd(self, joint_vel: np.ndarray) -> None:
        self._joint_vel_cmd = joint_vel

    @property
    def joint_positions(self) -> np.ndarray:
        return self._joint_positions

    @property
    def joint_velocities(self) -> np.ndarray:
        return self._joint_positions

    @property
    def joint_efforts(self) -> np.ndarray:
        return self._joint_positions

    def send_gripper_goal(self, gripper_pos: float) -> bool:
        """Send a gripper goal."""
        gripper_goal = ParallelGripperCommand.Goal()
        gripper_goal.command.name = self.cfg.gripper_joint_names
        gripper_goal.command.position = [gripper_pos]
        gripper_goal.command.effort = [100.0]
        self.gripper_action_client.send_goal(gripper_goal)

    def _joint_pub_callback(self) -> None:
        """Publish the robot commands on callback."""
        if self._twist_cmd is not None:
            twist_cmd = Twist()
            twist_cmd.linear.x = float(self._twist_cmd[0])
            twist_cmd.linear.y = float(self._twist_cmd[1])
            twist_cmd.linear.z = float(self._twist_cmd[2])
            twist_cmd.angular.x = float(self._twist_cmd[3])
            twist_cmd.angular.y = float(self._twist_cmd[4])
            twist_cmd.angular.z = float(self._twist_cmd[5])
            self.twist_pub.publish(twist_cmd)
        if self._joint_vel_cmd is not None:
            pass

    def _robot_state_sub(self, msg: JointState) -> None:

        self._joint_positions = np.asarray([msg.position[msg.name.index(j)] for j in self.joint_names]).astype(
            np.float32
        )
        self._joint_velocities = np.asarray([msg.velocity[msg.name.index(j)] for j in self.joint_names]).astype(
            np.float32
        )
        self._joint_efforts = np.asarray([msg.effort[msg.name.index(j)] for j in self.joint_names]).astype(np.float32)

    def switch_controllers(self, activate: list[str], deactivate: list[str], timeout: float = 5.0) -> bool:
        """Switch active controllers.

        Args:
            activate:   list of controller names to start
            deactivate: list of controller names to stop
            timeout: seconds until timeout

        """
        # Wait for service to be available
        if not self.switch_ctrl_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("controller_manager service not available!")
            return False

        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = rclpy.duration.Duration(seconds=timeout).to_msg()

        future = self.switch_ctrl_client.call_async(request)
        return True  # TODO address the fact that switches could fail

        if future.result() is not None:
            if future.result().ok:
                self.get_logger().info("Switched controllers successfully")
                return True
            self.get_logger().error("Switch controller request failed")
            return False
        self.get_logger().error("Service call returned no result (timeout?)")
        return False

    @classmethod
    def initialize_ros2(cls, dt: float, cfg: Ros2EnvCfg) -> "Ros2EnvListener":
        """Initialize Ros2 and return the listener object."""
        rclpy.init()
        return cls(dt, cfg)

    @staticmethod
    def spin_node(node: Node) -> None:
        """Spin the Ros2 Node on its own thread."""
        while rclpy.ok():
            rclpy.spin_once(node)
        node.destroy_node()
