"""Run the perception pipeline on RGB-D observations from ROS2."""

import argparse
import os
import time

import cv2
import numpy as np
from jaxtyping import Int

from kinova_tasks.ros2_tasks.kinova.kinova_ros2 import KinovaROS2Env, KinovaROS2EnvCfg
from skillet.envs.ros2_env_wrapper import ROS2EnvWrapper
from skillet.envs.util import parse_ros2_env_cfg, setup_ros
from skillet.perception.perception import Perception

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument(
    "--prompts", nargs="+", type=str, default=["block", "eraser"], help="Prompts to use for segmentation."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ROS2-Reach-Kinova-v0", help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--use_fake_hardware", type=str, default="true", help="'true' or 'false'.")
parser.add_argument("--launch_ros", action="store_true", help="Launch ROS from env startup.")
parser.add_argument("--period_s", type=float, default=1.0, help="Seconds between service requests.")
parser.add_argument("--max_depth_m", type=float, default=None, help="Optional far-plane clipping depth in meters.")
parser.add_argument(
    "--prompts",
    nargs="*",
    default=[],
    help="Optional SAM3 prompts (e.g. --prompts figurine mug). If empty, no SAM segmentation is used.",
)

args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


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

    sam3 = SAM3(device=args_cli.device)
    env = KinovaROS2Env(cfg=env_cfg, ros=setup_ros())
    env = ROS2EnvWrapper(env)
    env.reset()
    rgbd_spec = env.obs_spec_rgbd.unbatched()

    poll_rate_hz = 1.0 / max(args_cli.period_s, 1e-6)
    perception = Perception(
        env=env,
        obs_spec=rgbd_spec,
        poll_rate=poll_rate_hz,
        device=args_cli.device,
        max_depth_m=args_cli.max_depth_m,
        prompts=args_cli.prompts,
    )
    perception.start_visualization()
    print("[INFO] Running perception loop. Press Ctrl+C to quit.")
    try:
        perception.run()
    except KeyboardInterrupt:
        print("[INFO] Stopping perception loop.")
    finally:
        perception.stop()
        env.close()


def _depth_to_colormap(depth_mm: Int[np.ndarray, "h w"]) -> Int[np.ndarray, "h w 3"]:
    valid = depth_mm > 0
    if not valid.any():
        return cv2.applyColorMap(depth_mm.astype("uint8"), cv2.COLORMAP_TURBO)

    depth_valid = depth_mm[valid].astype("float32")
    lo = float(depth_valid.min())
    hi = float(depth_valid.max())
    if hi <= lo:
        hi = lo + 1.0
    depth_norm = ((depth_mm.astype("float32") - lo) / (hi - lo) * 255.0).clip(0, 255).astype("uint8")
    depth_norm[~valid] = 0
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


if __name__ == "__main__":
    main()
