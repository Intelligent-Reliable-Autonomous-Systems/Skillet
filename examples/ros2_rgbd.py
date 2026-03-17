"""Run the perception pipeline on RGB-D observations from ROS2."""

import argparse
import os

from skillet.envs.skillet_env import SkilletEnv
from skillet.perception.perception import Perception
from skillet.perception.realsense import RealsenseEnv
from skillet.perception.sam3.sam3 import SAMConcept
from skillet.scene.visualize import Open3DVisualizer
from skillet_tasks.ros2_tasks.factory import create_ros2_env

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--segmentation", action=argparse.BooleanOptionalAction, default=True, help="Use segmentation.")
parser.add_argument("--realsense_env", action="store_true", help="Use RealSense camera environment.")
parser.add_argument(
    "--viz", type=str, default="rgb,depth,pointcloud", help="Visualization modes to display, as comma-separated string."
)
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--launch_ros", action="store_true", help="Launch ROS from env startup.")
parser.add_argument("--period_s", type=float, default=1.0, help="Seconds between service requests.")
parser.add_argument("--max_depth_m", type=float, default=None, help="Optional far-plane clipping depth in meters.")
parser.add_argument("--task", type=str, default="ROS2-Gen3Lite-v0", help="ROS2 Environment")

args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")

# Table bounds (same as perception.py __main__)
TABLE_X0 = -0.0889
TABLE_Y0 = -0.577
TABLE_DX = 0.762
TABLE_DY = 1.2446
WORLD_BOUNDS = (
    TABLE_X0,
    TABLE_Y0,
    0,
    TABLE_X0 + TABLE_DX,
    TABLE_Y0 + TABLE_DY,
    1,
)  # min_x, min_y, min_z, max_x, max_y, max_z

# PROMPTS = {
#     "wooden_block": "a light brown wooden block",
#     "purple_block": "a solid purple block without any writing or markings",
#     "yellow_block": "a solid yellow block without any writing or markings",
#     "green_block": "a solid green block without any writing or markings",
# }
PROMPTS = [
    SAMConcept(name="block_8", prompt="wooden block with number 8 on it"),
    SAMConcept(name="block_7", prompt="wooden block with number 7 on it"),
    SAMConcept(name="mouse", prompt="a computer mouse"),
]


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    if args_cli.realsense_env:
        env = RealsenseEnv()
    else:
        env_cfg = {
            "robot_ip": args_cli.robot_ip,
            "launch_ros": args_cli.launch_ros,
            "device": args_cli.device,
            "num_envs": args_cli.num_envs,
            "ros2_workspace": args_cli.ros2_ws,
        }

        env = create_ros2_env(args_cli.task, env_cfg)
        env = SkilletEnv(env)
    env.reset()
    rgbd_spec = env.obs_spec_rgbd.unbatched()

    poll_rate_hz = 1.0 / max(args_cli.period_s, 1e-6)
    perception = Perception(
        env=env,
        obs_spec=rgbd_spec,
        segmentation=args_cli.segmentation,
        poll_rate=poll_rate_hz,
        device=args_cli.device,
        max_depth_m=args_cli.max_depth_m,
        prompts=PROMPTS,
        world_bounds=WORLD_BOUNDS,
    )

    vis = Open3DVisualizer(world_bounds=WORLD_BOUNDS)
    if "pointcloud" in args_cli.viz:
        perception.set_visualizer(vis, segment_point_cloud=True)
    perception.start_cv2_visualization(
        display_rgb="rgb" in args_cli.viz,
        display_depth="depth" in args_cli.viz,
        segment_rgb="rgb" in args_cli.viz,
        segment_depth="depth" in args_cli.viz,
    )
    perception.run_thread()

    print("[INFO] Running perception loop. Close the Open3D window or press Ctrl+C to quit.")
    try:
        vis.run()
    except KeyboardInterrupt:
        print("[INFO] Stopping perception loop.")
    finally:
        perception.stop()
        env.close()


if __name__ == "__main__":
    main()
