"""Open3D visualization for point clouds from object localization - HDF5 Playback Mode."""

from __future__ import annotations

import argparse
import threading
from contextlib import suppress
from typing import Literal

import cv2
import h5py
import numpy as np
import torch

from skillet.scene.base import Scene
from skillet.scene.cube import Cube


class SkilletPlaybackEnv:
    """Environment for playing back observations."""

    def __init__(self, log_file: str, device: str = "cuda", render_mode: Literal["cv", "human", None] = None):

        self._load_log_file(log_file)
        self._curr_obs_idx = 0
        self.device = device
        self.render_mode = render_mode

        self._display_rgb = True
        self._display_depth = False
        self._width = 1200
        self._height = 900
        self._perception_window_name = "Playback Perception Scene"
        self._perception_active = False
        self._perception_stop_event = threading.Event()

    def step(self, action=None) -> tuple[dict, float, bool, bool, dict]:
        """Step through the observations."""
        obs = self._get_obs_at(self._curr_obs_idx)

        if self.render_mode == "human":
            self._ensure_perception_window()
            self._update_perception_window(obs["perception_frame"])
        self._curr_obs = obs
        self._curr_obs_idx += 1

        term = self._curr_obs_idx >= self._n_idx
        return (obs, 0, term, term, {})

    def reset(self) -> tuple[dict, dict]:
        self._curr_obs = self._get_obs_at(0)
        if self.render_mode == "human":
            self._ensure_perception_window()
            self._update_perception_window(self._curr_obs["perception_frame"])
            cv2.waitKey(100)
        return self._curr_obs, {}

    def get_observation(self, obs_spec=None):
        return self._curr_obs

    def _load_log_file(self, log_file: str) -> None:
        with h5py.File(log_file, "r") as f:

            def print_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")

            f.visititems(print_datasets)
            self._rgb_obs = f["episode/rgb"][:]
            self._depth_obs = f["episode/depth"][:]
            self._camera_pose = f["episode/camera_pose"][:]
            self._tcp_pose = f["episode/tcp_pose"][:]
            self._perception_frame = f["episode/perception_frame"][:]
            self._time_stamps = f["episode/time_stamps"][:]
            # self._abs_state = f["episode/abs_state"][:]
            self._intrinsic_k = f["episode/intrinsic_k"][:]  # (3, 3)
            self._obj_poses = f["episode/obj_poses"][:]
            self._obj_ids = f["episode/obj_ids"][:]
            self._obj_names = f["episode/obj_names"][:]
            self._world_bounds = f["episode/world_bounds"][:]

        self._n_idx = self._time_stamps.shape[0]
        print(f"[Playback] Loaded {self._n_idx} frames from {log_file}")

    def _get_obs_at(self, idx: int) -> dict[str, torch.Tensor]:
        """Build an obs dict (matching env observation format) for frame idx."""
        # rgb: stored as (H, W, 3) uint8 -> (3, H, W) float32 in [0,1]
        rgb = torch.from_numpy(self._rgb_obs[idx]).to(self.device) / 255.0
        depth = torch.from_numpy(self._depth_obs[idx]).to(self.device).unsqueeze(0)
        camera_pose = torch.from_numpy(self._camera_pose[idx]).to(self.device)
        tcp_pose = torch.from_numpy(self._tcp_pose[idx]).to(self.device)
        perception_frame = self._perception_frame[idx]
        intrinsic_k = torch.from_numpy(self._intrinsic_k[idx]).to(self.device)
        cube_poses = torch.from_numpy(self._obj_poses[idx]).to(self.device)
        cube_names = self._obj_names[idx].astype(str)
        cube_ids = torch.from_numpy(self._obj_ids[idx]).to(self.device)
        world_bounds = self._world_bounds[idx]
        scene_objs = [Cube(size=0.036, init_pose=cube_poses[i], name=cube_names[i]) for i in range(cube_ids.shape[0])]
        self.scene = Scene(scene_objs, bounds=world_bounds, closed_set=True)

        return {
            "rgb": rgb,
            "depth": depth,
            "camera_pose": camera_pose,
            "tcp_pose_b": tcp_pose,
            "intrinsic_k": intrinsic_k,
            "scene": self.scene,
            "perception_frame": perception_frame,
        }

    def run_perception(self) -> None:
        """CV2 loop: render current frame, wait for spacebar to advance."""
        self._ensure_perception_window()

        while not self._perception_stop_event.is_set() and self._curr_obs_idx < self._n_idx:
            obs = self._get_obs_at(self._curr_obs_idx)

            self._update_perception_window(obs["perception_frame"])

            ts = self._time_stamps[self._curr_obs_idx]
            print(f"[Playback] Frame {self._curr_obs_idx}/{self._n_idx - 1}  t={ts:.3f}s  (SPACE=next)")

            # Block until spacebar
            while not self._perception_stop_event.is_set():
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    self._curr_obs_idx += 1
                    break

        if self._curr_obs_idx >= self._n_idx:
            print("[Playback] End of recording.")

        self.stop()

    def _update_perception_window(self, frame: np.ndarray) -> None:
        """Show the most recent perception frame."""
        cv2.imshow(self._perception_window_name, frame)
        cv2.waitKey(1)

    def _ensure_perception_window(self) -> None:
        if self._perception_active:
            return
        if not (self._display_rgb or self._display_depth):
            return
        cv2.namedWindow(self._perception_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._perception_window_name, self._width, self._height)
        self._perception_active = True

    def _stop_perception(self) -> None:
        if self._perception_active:
            with suppress(cv2.error):
                cv2.destroyWindow(self._perception_window_name)
            self._perception_active = False

    # Public API
    def run(self) -> None:
        """Launch both windows. Call on the main thread."""
        self.run_perception()

    def stop(self) -> None:
        self._stop_perception()

    def close(self) -> None:
        self.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="data/test/20260403_140731/exp_0/data.h5")
    parser.add_argument("--env", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.env:
        env = SkilletPlaybackEnv(log_file=args.log_file, render_mode="human")
        env.reset()
        term = False
        i = 0
        while not term:
            obs, _, term, _, _ = env.step()
            print(f"Scene at step: {i}")
            print(f"{obs['scene']}\n")
            i += 1
            while True:
                user_input = input("Press Enter to load next frame: ")
                if user_input == "":
                    break
                if user_input.lower() == "q":
                    break
        env.close()
    else:
        viz = SkilletPlaybackEnv(log_file=args.log_file, render_mode="cv")
        viz.run()


if __name__ == "__main__":
    main()
