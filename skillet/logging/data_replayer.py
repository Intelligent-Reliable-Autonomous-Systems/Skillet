"""Open3D visualization for point clouds from object localization - HDF5 Playback Mode."""

from __future__ import annotations

import argparse
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import cv2
import h5py
import numpy as np
import torch

from skillet.scene.base import Scene
from skillet.scene.cube import Cube
from skillet.scene.utils import (
    _BBOX_THICKNESS,
    _FONT,
    _FONT_SCALE,
    _FONT_THICKNESS,
    _OVERLAY_ALPHA,
    _PALETTE_BGR,
    create_aabb_lineset,
    create_camera_model,
    depth_to_colormap_np,
    get_object_geometry,
    point_cloud_to_open3d,
    quat_to_roll_pitch_yaw,
    segmented_rgbd_to_point_cloud,
    tilt_from_quat_wxyz,
)

try:
    import open3d as o3d
    import open3d.visualization.gui as _gui
    import open3d.visualization.rendering as _rendering
except ImportError:
    o3d = None  # type: ignore[assignment]
    _gui = None  # type: ignore[assignment]
    _rendering = None  # type: ignore[assignment]


class SkilletPlaybackVisualizer:
    """Playback visualizer that reads from an HDF5 log file.

    Advances one frame per spacebar press in the CV2 window.
    Mirrors the structure of SkilletVisualizer but replaces environment
    polling with indexed reads from pre-loaded numpy arrays.
    """

    _CAM_GEOMETRY_NAMES = ("cam_face", "cam_body", "cam_marker")

    def __init__(
        self,
        log_file: str,
        device: str = "cuda",
        width: int = 640,
        height: int = 480,
        display_rgb: bool = True,
        display_depth: bool = True,
        segment_rgb: bool = False,
        segment_depth: bool = False,
    ) -> None:
        self._width = width
        self._height = height

        self.device = device
        self.scene = None
        # Playback state
        self._frame_idx = 0
        self._n_frames = 0
        self._advance_event = threading.Event()

        # Data arrays
        self._rgb_obs: np.ndarray | None = None
        self._depth_obs: np.ndarray | None = None
        self._camera_pose: np.ndarray | None = None
        self._twist_obs: np.ndarray | None = None
        self._time_stamps: np.ndarray | None = None
        self._abs_state: np.ndarray | None = None
        self.intrinsic_k: torch.Tensor | None = None

        self._load_log_file(log_file)

        # Open3D state
        self._app: Any | None = None
        self._window: Any | None = None
        self._scene_widget: Any | None = None
        self._hud_label: Any | None = None
        self._mat_unlit: Any | None = None
        self._mat_lit: Any | None = None
        self._mat_line: Any | None = None
        self._added_geometries: set[str] = set()
        self._needs_camera_setup = True
        self._closed = False

        self._scene_thread: threading.Thread | None = None

        # CV2 state
        self._display_rgb = display_rgb
        self._display_depth = display_depth
        self._segment_rgb = segment_rgb
        self._segment_depth = segment_depth
        self._rgbd_window_name = "Playback RGB-D Scene"
        self._rgbd_active = False
        self._rgbd_thread: threading.Thread | None = None
        self._rgbd_stop_event = threading.Event()

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
            self._time_stamps = f["episode/time_stamps"][:]
            # self._abs_state = f["episode/abs_state"][:]
            self._intrinsic_k = f["episode/intrinsic_k"][:]  # (3, 3)
            self._obj_poses = f["episode/obj_poses"][:]
            self._obj_ids = f["episode/obj_ids"][:]
            self._world_bounds = f["episode/world_bounds"][:]

        self._n_frames = self._rgb_obs.shape[0]
        print(f"[Playback] Loaded {self._n_frames} frames from {log_file}")

    def _get_obs_at(self, idx: int) -> dict[str, torch.Tensor]:
        """Build an obs dict (matching env observation format) for frame idx."""
        # rgb: stored as (H, W, 3) uint8 -> (3, H, W) float32 in [0,1]
        rgb = torch.from_numpy(self._rgb_obs[idx]).to(self.device) / 255.0
        depth = torch.from_numpy(self._depth_obs[idx]).to(self.device).unsqueeze(0)
        camera_pose = torch.from_numpy(self._camera_pose[idx]).to(self.device)
        tcp_pose = torch.from_numpy(self._tcp_pose[idx]).to(self.device)
        intrinsic_k = torch.from_numpy(self._intrinsic_k[idx]).to(self.device)
        cube_poses = torch.from_numpy(self._obj_poses[idx]).to(self.device)
        cube_ids = torch.from_numpy(self._obj_ids[idx]).to(self.device)
        world_bounds = self._world_bounds[idx]
        scene_objs = [Cube(size=0.036, init_pose=cube_poses[i]) for i in range(cube_ids.shape[0])]
        self.scene = Scene(scene_objs, bounds=world_bounds, closed_set=True)

        return {
            "rgb": rgb,
            "depth": depth,
            "camera_pose": camera_pose,
            "tcp_pose_b": tcp_pose,
            "intrinsic_k": intrinsic_k,
        }

    def _default_masks(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single full-image mask — mirrors _default_segmentation."""
        masks = torch.ones((1, depth.shape[-2], depth.shape[-1]), dtype=torch.bool, device=depth.device)
        segment_ids = torch.zeros((1,), dtype=torch.int64, device=depth.device)
        return masks, segment_ids

    # Open3D Scene info
    def _setup(self) -> None:
        self._app = _gui.Application.instance
        self._app.initialize()

        self._window = self._app.create_window("Playback Table Scene", self._width, self._height)
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)

        self._scene_widget = _gui.SceneWidget()
        self._scene_widget.scene = _rendering.Open3DScene(self._window.renderer)
        self._window.add_child(self._scene_widget)

        self._hud_label = _gui.Label("Frame: waiting...")
        self._window.add_child(self._hud_label)

        self._mat_unlit = _rendering.MaterialRecord()
        self._mat_unlit.shader = "defaultUnlit"
        self._mat_unlit.point_size = 3 * self._window.scaling

        self._mat_lit = _rendering.MaterialRecord()
        self._mat_lit.shader = "defaultLit"

        self._mat_line = _rendering.MaterialRecord()
        self._mat_line.shader = "unlitLine"
        self._mat_line.line_width = 2 * self._window.scaling

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self._add_geometry("coord_frame", coord, self._mat_lit)

        if self.scene is not None and self.scene.bounds is not None:
            bounds_ls = create_aabb_lineset(self.scene.bounds)
            if bounds_ls is not None:
                self._add_geometry("scene_bounds", bounds_ls, self._mat_line)

    def _add_geometry(self, name: str, geom: Any, mat: Any) -> None:
        scene = self._scene_widget.scene
        if name in self._added_geometries:
            scene.remove_geometry(name)
        scene.add_geometry(name, geom, mat)
        self._added_geometries.add(name)

    def _remove_geometry(self, name: str) -> None:
        if name in self._added_geometries:
            self._scene_widget.scene.remove_geometry(name)
            self._added_geometries.discard(name)

    def _on_layout(self, layout_context: Any) -> None:
        r = self._window.content_rect
        self._scene_widget.frame = r
        pref = self._hud_label.calc_preferred_size(layout_context, _gui.Widget.Constraints())
        self._hud_label.frame = _gui.Rect(r.x + 10, r.y + 10, pref.width, pref.height)

    def _on_close(self) -> bool:
        self._closed = True
        return True

    def update(
        self,
        point_cloud: torch.Tensor,
        segment_indices: torch.Tensor | None = None,
        camera_pose: torch.Tensor | None = None,
        frame_idx: int = 0,
        timestamp: float | None = None,
    ) -> None:
        """Push new frame data to the Open3D scene (thread-safe)."""
        if self._closed or self._app is None or self._window is None:
            return

        pcd = point_cloud_to_open3d(
            point_cloud,
            segment_indices=segment_indices,
            filter_zero=True,
            world_bounds=self.scene.bounds,
        )

        cam_meshes: list[Any] | None = None
        hud_text = ""
        if camera_pose is not None:
            cam_np = camera_pose.detach().cpu().numpy().astype(np.float64)
            cam_meshes = create_camera_model(cam_np)
            pos = cam_np[:3]
            q = cam_np[3:7]
            roll, pitch, yaw = quat_to_roll_pitch_yaw(q)
            ts_str = f"  t={timestamp:.3f}s" if timestamp is not None else ""
            hud_text = (
                f"Frame {frame_idx}/{self._n_frames - 1}{ts_str}\n"
                f"Cam: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
                f"q=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})\n"
                f"rpy=({roll:.3f}, {pitch:.3f}, {yaw:.3f})  "
                f"Tilt: {tilt_from_quat_wxyz(q):.3f}°"
            )

        do_camera_setup = self._needs_camera_setup

        def _do_update() -> None:
            if self._closed:
                return

            if pcd is not None:
                self._add_geometry("pcd", pcd, self._mat_unlit)
            else:
                self._remove_geometry("pcd")

            if cam_meshes is not None:
                for name, mesh in zip(self._CAM_GEOMETRY_NAMES, cam_meshes, strict=True):
                    self._add_geometry(name, mesh, self._mat_lit)
            else:
                for name in self._CAM_GEOMETRY_NAMES:
                    self._remove_geometry(name)

            for obj in self.scene.objects:
                geometry = get_object_geometry(obj)
                if geometry is not None and len(geometry) > 0:
                    self._add_geometry(obj.identifier, geometry[0], self._mat_line)
                else:
                    self._remove_geometry(obj.identifier)
            if hud_text:
                self._hud_label.text = hud_text
                self._window.set_needs_layout()

            if do_camera_setup and pcd is not None:
                bounds = self._scene_widget.scene.bounding_box
                self._scene_widget.setup_camera(60, bounds, bounds.get_center())
                self._needs_camera_setup = False

        self._app.post_to_main_thread(self._window, _do_update)

    def run_scene(self) -> None:
        if o3d is None:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        self._setup()
        self._app.run()

    def run_rgbd(self) -> None:
        """CV2 loop: render current frame, wait for spacebar to advance."""
        self._ensure_rgbd_window()

        while not self._rgbd_stop_event.is_set() and self._frame_idx < self._n_frames:
            obs = self._get_obs_at(self._frame_idx)
            masks, segment_ids = self._default_masks(obs["depth"])

            # Update Open3D scene
            point_cloud, segment_indices = self._observation_to_point_cloud(obs, masks, segment_ids)
            self.update(
                point_cloud,
                segment_indices=segment_indices,
                camera_pose=obs["camera_pose"],
                frame_idx=self._frame_idx,
                timestamp=float(self._time_stamps[self._frame_idx]),
            )

            self._update_rgbd_window(obs, masks, segment_ids)

            ts = self._time_stamps[self._frame_idx]
            print(f"[Playback] Frame {self._frame_idx}/{self._n_frames - 1}  t={ts:.3f}s  (SPACE=next  Q=quit)")

            # Block until spacebar or quit
            while not self._rgbd_stop_event.is_set():
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    self._frame_idx += 1
                    break
                if key in (ord("q"), ord("Q"), 27):
                    self._rgbd_stop_event.set()
                    self.stop()
                    break

        if self._frame_idx >= self._n_frames:
            print("[Playback] End of recording.")

        self.stop()

    def _update_rgbd_window(self, obs: dict[str, torch.Tensor], masks: torch.Tensor, segment_ids: torch.Tensor) -> None:
        if not (self._display_rgb or self._display_depth):
            return

        panels: list[np.ndarray] = []

        if self._display_rgb:
            rgb_np = (obs["rgb"].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            if self._segment_rgb:
                rgb_bgr = self._draw_instance_annotations(rgb_bgr, masks, segment_ids)
            panels.append(rgb_bgr)

        if self._display_depth:
            depth_np = obs["depth"].detach().cpu().numpy()[0]  # (H, W)
            if depth_np.dtype != np.uint16:
                depth_np = (depth_np * 1000.0).astype(np.uint16)
            depth_bgr = depth_to_colormap_np(depth_np)
            if self._segment_depth:
                depth_bgr = self._draw_instance_annotations(depth_bgr, masks, segment_ids)
            panels.append(depth_bgr)

        if not panels:
            return

        frame = panels[0] if len(panels) == 1 else np.concatenate(panels, axis=1)
        cv2.imshow(self._rgbd_window_name, frame)
        cv2.waitKey(1)

    def _draw_instance_annotations(
        self, image: np.ndarray, masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> np.ndarray:
        out = image.copy()
        overlay = image.copy()
        masks_np = masks.detach().cpu().numpy()
        ids_np = segment_ids.detach().cpu().numpy()
        n = masks_np.shape[0]

        for i in range(n):
            color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
            overlay[masks_np[i] > 0] = color

        cv2.addWeighted(overlay, _OVERLAY_ALPHA, out, 1.0 - _OVERLAY_ALPHA, 0, out)

        for i in range(n):
            seg_mask = masks_np[i] > 0
            prompt_idx = int(ids_np[i])
            color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
            ys, xs = np.where(seg_mask)
            if len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cv2.rectangle(out, (x1, y1), (x2, y2), color, _BBOX_THICKNESS)
            label = f"#{i} obj_{prompt_idx}"
            (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
            tx, ty = x1, y1 - 6
            if ty - th < 0:
                ty = y1 + th + 6
            cv2.rectangle(out, (tx - 1, ty - th - 4), (tx + tw + 5, ty + 4), color, cv2.FILLED)
            cv2.putText(out, label, (tx + 2, ty), _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICKNESS, cv2.LINE_AA)

        return out

    def _observation_to_point_cloud(
        self, obs: dict[str, torch.Tensor], masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_cloud, segment_indices = segmented_rgbd_to_point_cloud(
            obs["depth"],
            masks,
            obs["intrinsic_k"],
            obs["camera_pose"],
            rgb=obs["rgb"],
            use_perspective=False,
        )
        if segment_ids.numel() > 0 and segment_indices.numel() > 0:
            segment_indices = segment_ids[segment_indices]
        return point_cloud, segment_indices

    def _ensure_rgbd_window(self) -> None:
        if self._rgbd_active:
            return
        if not (self._display_rgb or self._display_depth):
            return
        cv2.namedWindow(self._rgbd_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._rgbd_window_name, self._width, self._height)
        self._rgbd_active = True

    def _stop_rgbd(self) -> None:
        if self._rgbd_active:
            with suppress(cv2.error):
                cv2.destroyWindow(self._rgbd_window_name)
            self._rgbd_active = False

    def request_close(self) -> None:
        self._closed = True
        if self._app is not None:
            self._app.quit()

    # Public API
    def run(self) -> None:
        """Launch both windows. Call on the main thread."""
        self._rgbd_thread = threading.Thread(target=self.run_rgbd, name="PlaybackRGBDThread", daemon=True)
        self._rgbd_thread.start()
        self.run_scene()

    def stop(self) -> None:
        self._rgbd_stop_event.set()
        if self._rgbd_thread is not None and threading.current_thread() != self._rgbd_thread:
            self._rgbd_thread.join(timeout=2.0)
        self._stop_rgbd()
        self.request_close()


class SkilletPlaybackEnv:
    """Environment for playing back observations."""

    def __init__(self, log_file: str, device: str = "cuda"):

        self._load_log_file(log_file)
        self._curr_obs_id = 0
        self.device = device

    def step(self, action=None) -> tuple[dict, float, bool, bool, dict]:
        """Step through the observations."""
        obs = self._get_obs_at(self._curr_obs_id)
        self._curr_obs = obs
        self._curr_obs_id += 1

        term = self._curr_obs_id >= len(self._time_stamps)
        return (obs, 0, term, term, {})

    def reset(self) -> tuple[dict, dict]:

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
            self._time_stamps = f["episode/time_stamps"][:]
            # self._abs_state = f["episode/abs_state"][:]
            self._intrinsic_k = f["episode/intrinsic_k"][:]  # (3, 3)
            self._obj_poses = f["episode/obj_poses"][:]
            self._obj_ids = f["episode/obj_ids"][:]
            self._obj_names = f["episode/obj_names"][:]
            self._world_bounds = f["episode/world_bounds"][:]

        self._n_frames = self._rgb_obs.shape[0]
        print(f"[Playback] Loaded {self._n_frames} frames from {log_file}")

    def _get_obs_at(self, idx: int) -> dict[str, torch.Tensor]:
        """Build an obs dict (matching env observation format) for frame idx."""
        # rgb: stored as (H, W, 3) uint8 -> (3, H, W) float32 in [0,1]
        rgb = torch.from_numpy(self._rgb_obs[idx]).to(self.device) / 255.0
        depth = torch.from_numpy(self._depth_obs[idx]).to(self.device).unsqueeze(0)
        camera_pose = torch.from_numpy(self._camera_pose[idx]).to(self.device)
        tcp_pose = torch.from_numpy(self._tcp_pose[idx]).to(self.device)
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
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="data/test/20260403_140731/exp_0/data.h5")
    parser.add_argument("--env", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.env:
        env = SkilletPlaybackEnv(log_file=args.log_file)
        term = False
        i = 0
        while not term:
            obs, _, term, _, _ = env.step()
            print(f"Scene at step: {i}")
            print(f"{obs['scene']}\n")
            i += 1

    else:
        viz = SkilletPlaybackVisualizer(log_file=args.log_file)
        viz.run()


if __name__ == "__main__":
    main()
