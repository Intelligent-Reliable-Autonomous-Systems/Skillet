"""Run the perception pipeline on RGB-D observations from ROS2."""

import argparse
import os

from kinova_tasks.ros2_tasks.kinova.kinova_ros2 import KinovaROS2Env, KinovaROS2EnvCfg
from skillet.envs.ros2_skillet_env import ROS2SkilletEnv
from skillet.envs.util import setup_ros
from skillet.perception.perception import Perception
from skillet.perception.visualize import PointCloudVisualizer

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--use_fake_hardware", type=str, default="true", help="'true' or 'false'.")
parser.add_argument("--launch_ros", action="store_true", help="Launch ROS from env startup.")
parser.add_argument("--period_s", type=float, default=1.0, help="Seconds between service requests.")
parser.add_argument("--max_depth_m", type=float, default=None, help="Optional far-plane clipping depth in meters.")

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
WORLD_BOUNDS = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)

PROMPTS = {
    "wooden_block": "a light brown wooden block",
    "purple_block": "a solid purple block without any writing or markings",
    "yellow_block": "a solid yellow block without any writing or markings",
    "green_block": "a solid green block without any writing or markings",
}


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    env_cfg = KinovaROS2EnvCfg(
        robot_ip=args_cli.robot_ip,
        use_fake_hardware=args_cli.use_fake_hardware,
        launch_ros=args_cli.launch_ros,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        ros2_workspace=args_cli.ros2_ws,
    )

    env = KinovaROS2Env(cfg=env_cfg, ros=setup_ros())
    env = ROS2SkilletEnv(env)
    env.reset()
    rgbd_spec = env.obs_spec_rgbd.unbatched()

    poll_rate_hz = 1.0 / max(args_cli.period_s, 1e-6)
    perception = Perception(
        env=env,
        obs_spec=rgbd_spec,
        poll_rate=poll_rate_hz,
        device=args_cli.device,
        max_depth_m=args_cli.max_depth_m,
        prompts=PROMPTS,
        world_bounds=WORLD_BOUNDS,
    )

    vis = PointCloudVisualizer(world_bounds=WORLD_BOUNDS)
    perception.set_visualizer(vis, segment_point_cloud=True)
    perception.start_cv2_visualization(
        display_rgb=True, display_depth=True,
        segment_rgb=True, segment_depth=True,
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
