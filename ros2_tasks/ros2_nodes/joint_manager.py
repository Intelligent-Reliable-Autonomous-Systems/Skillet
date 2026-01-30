"""policy_controller.py

Minimal joint position controller to publish joint positions to ROS2

Written by Will Solow, 2026
"""

import io
import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from rcl_interfaces.msg import SetParametersResult


class JointMananger(Node):
    """A joint manager passes joint positions to the robot"""

    current_joint_positions = None
    current_joint_velocities = None

    def __init__(self, name) -> None:
        super().__init__(name)

        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("cmd_topic", "/joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("min_traj_dur", 1.0)
        self.state_topic = self.get_parameter("state_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.min_traj_dur = self.get_parameter("min_traj_dur").value
        self.step_size = self.get_parameter("step_size").value

        self.traj_sub = self.create_subscription(JointState, self.state_topic, self.robot_state_callback, 10)
        self.add_on_set_parameters_callback(self.param_callback)

        self.traj_pub = self.create_publisher(JointTrajectory, self.cmd_topic, 10)
        self.gripper_action_client = ActionClient(self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")

        self.get_logger().info(f"Initialized {name} policy controller")
        self.has_joint_data = False
        self.has_default_pos = False

    def robot_state_callback(self, msg: JointState):
        """
        Callback for receiving controller state messages.
        Updates the current joint positions and passes the state to the robot model.
        """
        self.update_joint_state(msg.position, msg.velocity)

    def update_joint_state(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Update the current joint state.

        Args:
            position: A list or array of joint positions.
            velocity: A list or array of joint velocities.
        """

        self.current_joint_positions = np.array(position[: self.num_joints], dtype=np.float32)

        self.current_joint_velocities = np.array(velocity[: self.num_joints], dtype=np.float32)
        self.has_joint_data = True

    def send_robot_cmd(self, joint_pos: np.ndarray) -> None:
        """
        Timer callback to compute and publish the next joint trajectory command
        and send action to gripper
        """

        if joint_pos is not None:
            if len(joint_pos) != self.num_actions:
                self.get_logger().error(f"Expected {self.num_actions} joint positions, got {len(joint_pos)}!")
            else:
                traj = JointTrajectory()
                traj.joint_names = self.arm_joints

                point = JointTrajectoryPoint()
                point.positions = joint_pos.tolist()[: len(self.arm_actions)]
                point.time_from_start = Duration(sec=1, nanosec=0)

                traj.points.append(point)
                self.send_gripper_goal(position=joint_pos[-1])
                self.traj_pub.publish(traj)
        else:
            pass
            # self.get_logger().info("Joint positions are `None`")

    def send_gripper_goal(self, position: float = 0.0, max_effort: float = 100.0) -> None:
        """Send position goal to the gripper."""
        self.gripper_action_client.wait_for_server()

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        send_goal_future = self.gripper_action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future) -> None:
        result = future.result().result

    def param_callback(self, params):
        for param in params:
            if param.name == "state_topic":
                if not isinstance(param.value, str):
                    self.get_logger().warn("`state_topic` param must be of type `str`")
                    return SetParametersResult(successful=False)
                self.get_logger().info(f"Updated `state_topic` to: {param.value}")
                self.state_topic = param.value
                self.destroy_subscription(self.traj_sub)
                self.traj_sub = self.create_subscription(JointState, self.state_topic, self.robot_state_callback, 10)
            if param.name == "cmd_topic":
                if not isinstance(param.value, str):
                    self.get_logger().warn("`cmd_topic` param must be of type `str`")
                    return SetParametersResult(successful=False)
                self.get_logger().info(f"Updated `cmd_topic` to: {param.value}")
                self.cmd_topic = param.value
                self.destroy_publisher(self.traj_pub)
                self.traj_pub = self.create_publisher(self.TRAJ_TOPIC_TYPE, self.cmd_topic, 10)

        return SetParametersResult(successful=True)
