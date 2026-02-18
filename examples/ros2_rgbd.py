"""Visualize RGB-D snapshots from the Kinova ROS2 environment."""

import argparse
import os
import time

import cv2
import gymnasium as gym
import torch

import kinova_tasks.ros2_tasks  # noqa: F401
from kinova_tasks.ros2_tasks.kinova.kinova_reach_ros2 import KinovaROS2ReachEnv
from skillet.envs.ros2_env_wrapper import ROS2EnvWrapper
from skillet.envs.util import parse_ros2_env_cfg, setup_ros
from skillet.perception.object_localization import segmented_rgbd_to_point_cloud
from skillet.perception.visualize import visualize_point_cloud
from skillet.perception.sam3.sam3 import SAM3


def _depth_to_colormap(depth_mm):
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


parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
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

args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    env_cfg = parse_ros2_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, ros2_workspace=args_cli.ros2_ws
    )
    env_cfg.robot_ip = args_cli.robot_ip
    env_cfg.use_fake_hardware = args_cli.use_fake_hardware
    env_cfg.launch_ros = args_cli.launch_ros

    sam3 = SAM3(device=args_cli.device)
    # rgb = torch.zeros((3, 480, 640), dtype=torch.uint8, device=args_cli.device)
    # seg, seg_indices = sam3.predict(rgb, ["figurine"])
    # print(f"Found {len(seg)} objects: {seg_indices} with areas {seg.sum(dim=(1,2))}")

    # env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())
    env = KinovaROS2ReachEnv(cfg=env_cfg, ros=setup_ros())
    env = ROS2EnvWrapper(env)
    env.reset()
    rgbd_spec = env._obs_spec_rgbd.unbatched()

    # cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    print("[INFO] Press 'q' in an image window to quit.")
    prev_frame_t = None
    fps_ema = None
    fps_window_count = 0
    fps_window_start = time.perf_counter()

    try:
        while True:

            frame_t = time.perf_counter()
            print('getting observation')
            obs = env.get_observation(rgbd_spec)
            # mask = torch.zeros((4, 480, 640), dtype=torch.bool, device=obs["depth"].device)
            # mask[0, :240, :320] = True
            # mask[1, 240:, :320] = True
            # mask[2, :240, 320:] = True
            # mask[3, 240:, 320:] = True
            mask, prompt_indices = sam3.predict(obs["rgb"], ["figurine"])
            print(f"Found {len(mask)} objects: {prompt_indices} with areas {mask.sum(dim=(1,2))}")
            # for i in range(480):
            #     obs["depth"][0, i, :] = 500 + i
            # obs["depth"].fill_(500)
            print('converting to point cloud')
            point_cloud, segment_indices = segmented_rgbd_to_point_cloud(
                obs["depth"], mask, obs["intrinsic_k"], obs["camera_pose"],
                rgb=obs["rgb"], use_perspective=False)
            print('visualizing point cloud')
            visualize_point_cloud(point_cloud, segment_indices=segment_indices)
            # pc = point_cloud.reshape((1, 480, 640, 3))
            rgb = obs["rgb"].cpu().numpy().transpose((1, 2, 0))
            depth = obs["depth"].cpu().numpy()[0]

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_color = _depth_to_colormap(depth)

            # cv2.imshow("RGB", rgb_bgr)
            # cv2.imshow("Depth", depth_color)

            # time.sleep(0.3)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            fps_window_count += 1
            if prev_frame_t is not None:
                inst_fps = 1.0 / max(frame_t - prev_frame_t, 1e-6)
                fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)
            prev_frame_t = frame_t

            elapsed = frame_t - fps_window_start
            if elapsed >= 1.0:
                fps_window = fps_window_count / elapsed
                fps_ema_str = f"{fps_ema:.2f}" if fps_ema is not None else "n/a"
                print(
                    f"[INFO] ts={obs['timestamp']:.3f} fps={fps_window:.2f} fps_ema={fps_ema_str} ")
                fps_window_count = 0
                fps_window_start = frame_t

            time.sleep(args_cli.period_s)

    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
