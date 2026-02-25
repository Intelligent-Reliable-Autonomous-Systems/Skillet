"""Handles starting ROS2 Websocket Bridge."""

import time

from roslibpy import Ros


def setup_ros() -> Ros:
    """Open the ROS2 interface."""
    print("[INFO][Setup ROS] Waiting to connect to ROSBridge")
    print(
        "[INFO][Setup ROS] Ensure that rosbridge node is running: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`"
    )
    # Wait until it starts
    ros = Ros(host="localhost", port=9090)
    start = time.time()
    while True:
        try:
            ros.run(timeout=1)
            if ros.is_connected:
                print("[INFO][Setup ROS] Connected to rosbridge")
                break
        except RuntimeError:
            if time.time() - start > 30:
                raise TimeoutError(
                    "RosBridge failed to start. Is the rosbridge node running? ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
                )
            time.sleep(0.1)

    return ros
