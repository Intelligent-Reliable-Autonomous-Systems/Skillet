"""ros2_interface.py.

Main interface between ROS2 nodes and Python

Written by Will Solow, 2026

"""

import subprocess
import time
from typing import Any

from roslibpy import Ros, Service, Topic


def launch_robot_hardware(cfg, workspace_path: str, pkg: str, launch_file: str, default_joint_positions: list) -> None:
    """Launch the robot hardware in separate terminal using system ROS2 installation.

    Args:
        cfg: Configuration class for Robot
        workspace_path: String of absolute path to workspace
        pkg: ROS2 package to run
        launch_file: ros2 launch file

    """
    cmd = f"""
        bash -c '
        cd {workspace_path}
        source /opt/ros/jazzy/setup.bash
        source install/setup.bash
        ros2 launch {pkg} {launch_file} use_fake_hardware:={cfg.use_fake_hardware} robot_ip:={cfg.robot_ip}
        gripper:=robotiq2f85 default_joint_pos:={default_joint_positions}
        '
        """
    subprocess.Popen(["gnome-terminal", "--", "bash", "-c", cmd])

    print("[INFO][ROS2] Spinning up ROS2 in new terminal.")


def wait_for_topic_subscribe(ros: Ros, topic_name: str, topic_type: str, timeout: int = 30) -> None:
    """Wait for a ROS2 topic to appear before continuing."""
    print(f"[INFO][ROS2] Waiting for topic {topic_name} to be exposed...")

    start = time.time()
    seen = False

    def cb(msg: dict[str, Any]) -> None:
        """Fake callback for waiting."""
        nonlocal seen
        seen = True

    topic = Topic(ros, topic_name, topic_type)
    topic.subscribe(cb)

    while not seen:
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {topic_name} to be exposed")
        time.sleep(0.1)

    topic.unsubscribe()
    print(f"[INFO][ROS2] Found topic {topic_name}")


def wait_for_topic_publish(ros: Ros, topic_name: str, topic_type: str, timeout: int = 30) -> None:
    """Wait until a topic is available to publish to (does not require messages to be published yet).

    Args:
        ros: Roslibpy Ros object
        topic_name: the topic namespace
        topic_type: the topic type
        timeout: max seconds to wait

    """
    print(f"[INFO][ROS2] Waiting for topic {topic_name} to be publishable...")
    start = time.time()

    topic = Topic(ros, topic_name, topic_type)

    while True:
        try:
            topic.publish({})  # attempt a dummy publish
            print(f"[INFO][ROS2] Topic {topic_name} is now publishable")
            # TODO check that moving the return to else works
        except Exception:
            pass
        else:
            return

        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for topic {topic_name} to be publishable")
        time.sleep(0.1)


def wait_for_action_server(ros: Ros, action_name: str, action_type: str, timeout: int = 30) -> None:
    """Wait until a ROS2 action server is available.

    Args:
        ros: Roslibpy Ros object
        action_name: the action namespace
        action_type: the action type
        timeout: max seconds to wait

    """
    print(f"[INFO][ROS2] Waiting for action server '{action_name}' to become available...")

    start_time = time.time()

    action_server_srv = Service(ros, "/rosapi/action_servers", "rosapi/Empty")

    while True:
        try:
            result = action_server_srv.call({})
            if action_name in result["action_servers"]:
                print(f"[INFO][ROS2] Action server '{action_name}' is now available")
                return
        except Exception:
            pass  # rosbridge may not be ready yet

        if timeout is not None and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Timeout waiting for action server '{action_name}'")

        time.sleep(0.1)


def wait_for_rviz(ros: Ros, timeout: int = 30) -> None:
    """Wait until rviz has loaded.

    Args:
        ros: Roslibpy Ros object
        timeout: max seconds to wait

    """
    print("[INFO][ROS2] Waiting for RViz to become render...")

    srv = Service(ros, "/rosapi/nodes", "rosapi/Empty")
    start = time.time()

    while True:
        nodes = srv.call({})["nodes"]
        if "/rviz2" in nodes:
            print("[INFO][ROS2] RViz has rendered")
            return
        if time.time() - start > timeout:
            raise TimeoutError("RViz did not start")
        time.sleep(0.1)


def wait_until_ready(ready: dict, timeout: int = 30) -> None:
    """Wait until all topics have recieved a value."""
    start = time.time()
    print(f"[INFO][ROS2] Waiting for all topics to be ready {ready}...")

    while not all(ready.values()):
        if time.time() - start > timeout:
            raise TimeoutError(f"Not all topics ready: {ready}")
        time.sleep(0.1)
    print(f"[INFO][ROS2] All topics are ready {ready}...")
