"""kinova_launch.py.

Main launch file for Kinova Gen3 Arm with Robotiq 2F 85 gripper

Written by Will Solow, 2026.
"""

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(use_fake_hardware: bool, robot_ip: str) -> list[Node]:
    """Return ROS2 nodes to launch."""
    # Packages to load
    pkg_kortex_bringup = get_package_share_directory("kortex_bringup")
    pkg_kortex_vision = get_package_share_directory("kinova_vision")

    # Kinova Arm Launch Description
    kinova_arm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_kortex_bringup, "launch", "gen3.launch.py"])),
        launch_arguments={
            "use_fake_hardware": use_fake_hardware,
            "robot_ip": robot_ip,
            "gripper": "robotiq_2f_85",
        }.items(),
    )

    # Kinova Vision launch
    kinova_vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_kortex_vision, "launch", "kinova_vision.launch.py"])),
        launch_arguments={
            "device": robot_ip,
        }.items(),
        condition=IfCondition(LaunchConfiguration("vision")),
    )

    ee_publisher = Node(
        package="gen3_py",
        executable="ee_pub",
    )

    moveit_config = (
        MoveItConfigsBuilder("gen3", package_name="kinova_gen3_7dof_robotiq_2f_85_moveit_config")
        .robot_description(
            mappings={
                "use_fake_hardware": use_fake_hardware,
                "robot_ip": robot_ip,
                "gripper": "robotiq_2f_85",
                "gripper_joint_name": "robotiq_85_left_knuckle_joint",
                "dof": "7",
                "gripper_max_velocity": "100",
                "gripper_max_force": "100",
                "use_internal_bus_gripper_comm": "true",
            }
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
        ],
    )

    return [kinova_arm_launch, kinova_vision_launch, move_group_node, ee_publisher]
