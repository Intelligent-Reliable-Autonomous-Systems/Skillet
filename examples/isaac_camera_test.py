"""
isaac_camera_test.py

Minimal smoke-test for Camera Sensor integration in a Kinova Gen3 environment:
  - Instantiates KinovaGenCameraEnvCfg with a custom camera pose
  - Spins up KinovaGenCameraEnv (Kinova Gen3 + RGBD camera)
  - Resets the env and steps a few times with zero actions
  - Prints camera output shapes to confirm RGBD data is flowing

Run (headless, no display):
    python examples/isaac_camera_test.py --headless --enable_cameras --num_steps 5

With display:
    python examples/isaac_camera_test.py --enable_cameras --num_steps 5

With ROS2 bridge (publishes /camera/rgb and /camera/depth to ROS2 (Need to create Action Graph programmatically)):
    python examples/isaac_camera_test.py --enable_cameras --ros2_bridge --num_steps 5

Note: --ros2_bridge requires ROS2 to be sourced in the shell before running.
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI args must be parsed before AppLauncher ---
parser = argparse.ArgumentParser(description="Smoke-test for Kinova Gen3 + RGBD camera env.")
parser.add_argument("--num_steps", type=int, default=5, help="Number of sim steps to run after reset.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--ros2_bridge", action="store_true", default=False,
                    help="Enable the isaacsim.ros2.bridge extension. ROS2 must be sourced first.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.ros2_bridge:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")

import torch

from kinova_tasks.isaac_tasks.direct.kinova_camera import (
    KinovaGenCameraEnv,
    KinovaGenCameraEnvCfg,
)


def main():
    # --- Build config with a custom camera pose ---
    cfg = KinovaGenCameraEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.replicate_physics = args_cli.num_envs > 1

    cfg.camera_cfg.prim_path = "/World/envs/env_.*/Robot/Arm/bracelet_link/wrist_mounted_camera"
    cfg.camera_cfg.pos = (0.0, 0.05, 0.05)
    cfg.camera_cfg.rot = (0.0, 1.0, 0.0, 0.0)
    cfg.camera_cfg.width = 640
    cfg.camera_cfg.height = 480

    print("[TEST] Config:")
    print(f"       camera pos = {cfg.camera_cfg.pos}")
    print(f"       camera rot = {cfg.camera_cfg.rot}")
    print(f"       resolution = {cfg.camera_cfg.width} x {cfg.camera_cfg.height}")

    # --- Instantiate env ---
    env = KinovaGenCameraEnv(cfg, render_mode="rgb_array")

    # --- Reset ---
    obs, _ = env.reset()
    print(f"\n[TEST] Reset done. Observation keys: {list(obs.keys())}")
    print(f"       policy obs shape: {obs['policy'].shape}")

    # --- Check camera sensor is registered ---
    assert "camera" in env.scene.sensors, "[FAIL] Camera not found in scene.sensors"
    print("[TEST] Camera sensor found in scene.")

    # --- Step and inspect camera data ---
    zero_action = torch.zeros((cfg.scene.num_envs, cfg.action_space), device=env.device)

    for step in range(args_cli.num_steps):
        obs, _, _, _, _ = env.step(zero_action)

        cam_data = env.scene.sensors["camera"].data.output

        rgb   = cam_data.get("rgb")
        depth = cam_data.get("distance_to_image_plane")

        rgb_shape   = tuple(rgb.shape)   if rgb   is not None else None
        depth_shape = tuple(depth.shape) if depth is not None else None

        print(
            f"[TEST] step {step + 1:02d} | "
            f"rgb: {rgb_shape}  depth: {depth_shape}"
        )

    print("\n[TEST] Smoke-test passed — RGBD camera is live.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
