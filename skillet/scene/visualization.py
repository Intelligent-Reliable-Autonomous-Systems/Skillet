"""Open3D visualization for point clouds from object localization."""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

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
    make_point_marker,
    point_cloud_to_open3d,
    quat_to_roll_pitch_yaw,
    segmented_rgbd_to_point_cloud,
    tilt_from_quat_wxyz,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from skillet.core import BatchedEnvironment
    from skillet.core.env import Environment
    from skillet.core.spaces import ObservationSpec
    from skillet.scene.base import Scene

try:
    import open3d as o3d
    import open3d.visualization.gui as _gui
    import open3d.visualization.rendering as _rendering
except ImportError:
    o3d = None  # type: ignore[assignment]
    _gui = None  # type: ignore[assignment]
    _rendering = None  # type: ignore[assignment]


class SkilletVisualizer:
    """Stateful visualizer for Open3D using the ``gui.Application`` API and cv2 window for raw RGB-D images.

    Renders a point cloud with optional world-bounds wireframe, camera model,
    and a HUD label showing the current camera pose.  Designed to ``run()`` on
    the main thread while a perception pipeline pushes updates from a
    background thread via ``update()``.
    """

    _CAM_GEOMETRY_NAMES = ("cam_face", "cam_body", "cam_marker")

    def __init__(
        self,
        env: Environment | BatchedEnvironment,
        obs_spec: ObservationSpec,
        scene: Scene,
        poll_rate: float = 8,
        device: str | torch.device | None = None,
        width: int = 1024,
        height: int = 768,
    ) -> None:
        self.scene = scene
        self.env = env
        self._width = width
        self._height = height
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or obs_spec.device
        self.obs_spec = replace(obs_spec, device=self.device, is_torch=True)
        self.poll_rate = poll_rate

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
        self._scene_window_enabled = False

        # Marker state (written from any thread, read on GUI thread)
        self._target_pos: np.ndarray | None = None
        self._target_size: float = 0.007
        self._tcp_pos: np.ndarray | None = None
        self._get_tcp_pos = None

        self._scene_thread: threading.Thread | None = None
        self._scene_stop_event = threading.Event()

        # CV2 visualization variables
        self._display_rgb = True
        self._display_depth = True
        self._segment_rgb = True
        self._segment_depth = True
        self._rgbd_window_name = "Raw RGB-D Scene"
        self._rgbd_active = False

        self._rgbd_thread: threading.Thread | None = None
        self._rgbd_stop_event = threading.Event()

    def _setup(self) -> None:
        """Initialize gui.Application, create window / scene / HUD / static geometry."""
        self._app = _gui.Application.instance
        self._app.initialize()

        self._window = self._app.create_window(
            "Processed Table Scene",
            self._width,
            self._height,
        )
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)

        self._scene_widget = _gui.SceneWidget()
        self._scene_widget.scene = _rendering.Open3DScene(self._window.renderer)
        self._window.add_child(self._scene_widget)

        self._hud_label = _gui.Label("Cam: waiting for data...")
        self._window.add_child(self._hud_label)

        # Materials
        self._mat_unlit = _rendering.MaterialRecord()
        self._mat_unlit.shader = "defaultUnlit"
        self._mat_unlit.point_size = 3 * self._window.scaling

        self._mat_lit = _rendering.MaterialRecord()
        self._mat_lit.shader = "defaultLit"

        self._mat_line = _rendering.MaterialRecord()
        self._mat_line.shader = "unlitLine"
        self._mat_line.line_width = 2 * self._window.scaling

        # Static geometries
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self._add_geometry("coord_frame", coord, self._mat_lit)

        if self.scene.bounds is not None:
            bounds_ls = create_aabb_lineset(self.scene.bounds)
            if bounds_ls is not None:
                self._add_geometry("scene_bounds", bounds_ls, self._mat_line)

        self._scene_window_enabled = True

    def _on_layout(self, layout_context: Any) -> None:
        r = self._window.content_rect
        self._scene_widget.frame = r
        pref = self._hud_label.calc_preferred_size(layout_context, _gui.Widget.Constraints())
        self._hud_label.frame = _gui.Rect(r.x + 10, r.y + 10, pref.width, pref.height)

    def _on_close(self) -> bool:
        self._closed = True
        self._scene_window_enabled = False
        return True

    # ── Geometry helpers (GUI thread only)
    def _add_geometry(self, name: str, geom: Any, mat: Any) -> None:
        """Add or replace a named geometry in the scene."""
        scene = self._scene_widget.scene
        if name in self._added_geometries:
            scene.remove_geometry(name)
        if isinstance(geom, list):
            for g in geom:
                scene.add_geometry(name, g, mat)
        else:
            scene.add_geometry(name, geom, mat)
        self._added_geometries.add(name)

    def _remove_geometry(self, name: str) -> None:
        if name in self._added_geometries:
            self._scene_widget.scene.remove_geometry(name)
            self._added_geometries.discard(name)

    def update(
        self,
        point_cloud: torch.Tensor,
        segment_indices: torch.Tensor | None = None,
        camera_pose: torch.Tensor | None = None,
    ) -> None:
        """Push new data from any thread; the scene update runs on the GUI thread."""
        if self._closed or self._app is None or self._window is None:
            return

        # Prepare heavy geometry conversion on the calling (RGBD) thread.
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
            hud_text = (
                f"Cam: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
                f"q=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}) \n"
                f"rpy=({roll:.3f}, {pitch:.3f}, {yaw:.3f})  "
                f"Tilt: {tilt_from_quat_wxyz(q):.3f}°"
            )

        do_camera_setup = self._needs_camera_setup

        def _apply_to_scene() -> None:
            if self._closed:
                return

            # Point cloud
            if pcd is not None:
                self._add_geometry("pcd", pcd, self._mat_unlit)
            else:
                self._remove_geometry("pcd")

            # Camera model (3 meshes)
            if cam_meshes is not None:
                for name, mesh in zip(
                    self._CAM_GEOMETRY_NAMES,
                    cam_meshes,
                    strict=True,
                ):
                    self._add_geometry(name, mesh, self._mat_lit)
            else:
                for name in self._CAM_GEOMETRY_NAMES:
                    self._remove_geometry(name)

            for obj in self.scene.objects:
                geometry = get_object_geometry(obj)
                if geometry is not None and len(geometry) > 0:
                    self._add_geometry(obj.identifier, geometry, self._mat_line)
                else:
                    self._remove_geometry(obj.identifier)

            # Target position sphere
            if self._target_pos is not None:
                sphere = make_point_marker(self._target_pos, radius=self._target_size, color=(1, 0.5, 0))
                self._add_geometry("target_pos", sphere, self._mat_lit)
            else:
                self._remove_geometry("target_pos")

            # TCP position sphere
            if self._get_tcp_pos is not None:
                xyz = self._get_tcp_pos()
                if isinstance(xyz, torch.Tensor):
                    xyz = xyz.detach().cpu().numpy().astype(np.float64)
                if xyz is not None:
                    self._tcp_pos = np.asarray(xyz, dtype=np.float64)
                    if self._tcp_pos.ndim > 1:
                        self._tcp_pos = self._tcp_pos[0]
            if self._tcp_pos is not None:
                sphere = make_point_marker(self._tcp_pos[:3], radius=0.007, color=(1, 0, 1))
                self._add_geometry("tcp_pos", sphere, self._mat_lit)
            else:
                self._remove_geometry("tcp_pos")

            # HUD
            if hud_text:
                self._hud_label.text = hud_text
                self._window.set_needs_layout()

            # Auto-fit view on first valid point cloud
            if do_camera_setup and pcd is not None:
                bounds = self._scene_widget.scene.bounding_box
                self._scene_widget.setup_camera(60, bounds, bounds.get_center())
                self._needs_camera_setup = False

        self._app.post_to_main_thread(self._window, _apply_to_scene)

    def run_scene(self) -> None:
        """Set up the window and block on the GUI event loop (call on main thread)."""
        if o3d is None:
            raise ImportError("Open3D is required for visualization. Install with: pip install open3d")
        self._setup()
        self._app.run()

    def run_rgbd(self) -> None:
        """Run the RGBD window visualizer of the SkilletVisualizer Pipeline."""
        poll_period_s = 1.0 / self.poll_rate
        next_poll_t = time.perf_counter()

        while not self._rgbd_stop_event.is_set():
            obs = self.env.get_observation(self.obs_spec)
            obs_unbatched = self._maybe_unbatch(obs)

            # NOTE: No meaningful masks currently, this will all happen in scene
            depth = obs_unbatched["depth"]
            masks = torch.ones((1, depth.shape[-2], depth.shape[-1]), dtype=torch.bool, device=depth.device)
            segment_ids = torch.zeros((1,), dtype=torch.int64, device=depth.device)
            point_cloud, segment_indices = self._observation_to_point_cloud(obs_unbatched, masks, segment_ids)

            if self._scene_window_enabled:
                self.update(
                    point_cloud,
                    segment_indices=segment_indices,
                    camera_pose=obs_unbatched["camera_pose"],
                )

            self._update_rgbd_window(obs_unbatched, masks, segment_ids)

            next_poll_t += poll_period_s
            sleep_s = max(0.0, next_poll_t - time.perf_counter())
            if sleep_s > 0:
                time.sleep(sleep_s)
        self._stop_rgbd()

    def run_thread(self) -> None:
        """Run the visualizer in a thread."""
        if self._scene_thread is not None and self._scene_thread.is_alive():
            return
        self._scene_stop_event.clear()
        self._rgbd_stop_event.clear()
        self._scene_thread = threading.Thread(target=self.run_scene, name="SceneVisualizerThread", daemon=True)
        self._rgbd_thread = threading.Thread(target=self.run_rgbd, name="RGBDVisualizerThread", daemon=True)
        self._rgbd_thread.start()
        self._scene_thread.start()

    def stop_thread(self) -> None:
        """Stop the visualizer thread."""
        if self._scene_thread is not None and self._scene_thread.is_alive():
            self._scene_stop_event.set()
            self._scene_thread.join(timeout=2.0)
            self._scene_thread = None
        if self._rgbd_thread is not None and self._rgbd_thread.is_alive():
            self._rgbd_stop_event.set()
            self._rgbd_thread.join(timeout=2.0)
            self._rgbd_thread = None
        self._stop_rgbd()

    def request_close(self) -> None:
        """Request the GUI to shut down (safe to call from any thread)."""
        self._closed = True
        if self._app is not None:
            self._app.quit()

    def set_target_pos(self, xyz: Sequence[float] | None, size: float = 0.007) -> None:
        """Set the target position sphere marker. Pass None to clear."""
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy().astype(np.float64)
        self._target_pos = np.array(xyz, dtype=np.float64) if xyz is not None else None
        if self._target_pos is not None and self._target_pos.ndim > 1:
            self._target_pos = self._target_pos[0]
        self._target_size = size

    def start_rgbd_visualization(
        self,
        display_rgb: bool = False,
        display_depth: bool = False,
        segment_rgb: bool = False,
        segment_depth: bool = False,
        rgbd_window_name: str = "Raw RGB-D Scene",
    ) -> None:
        """Enable the CV2 RGB-D preview window."""
        self._display_rgb = display_rgb
        self._display_depth = display_depth
        self._segment_rgb = segment_rgb
        self._segment_depth = segment_depth
        self._rgbd_window_name = rgbd_window_name

    def _draw_instance_annotations(
        self, image: np.ndarray, masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> np.ndarray:
        """Draw semi-transparent overlays, bounding boxes, and prompt labels."""
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
            color = _PALETTE_BGR[i % len(_PALETTE_BGR)]

            ys, xs = np.where(seg_mask)
            if len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cv2.rectangle(out, (x1, y1), (x2, y2), color, _BBOX_THICKNESS)

            label = f"#{i} obj_{int(ids_np[i])}"
            (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
            tx, ty = x1, y1 - 6 if y1 - 6 - th >= 0 else y1 + th + 6
            cv2.rectangle(out, (tx - 1, ty - th - 4), (tx + tw + 5, ty + 4), color, cv2.FILLED)
            cv2.putText(out, label, (tx + 2, ty), _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICKNESS, cv2.LINE_AA)

        return out

    def _colorize_segmented_rgb(
        self, rgb_bgr: np.ndarray, masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> np.ndarray:
        """Annotate RGB image with per-type overlays, bounding boxes, and labels."""
        return self._draw_instance_annotations(rgb_bgr, masks, segment_ids)

    def _colorize_segmented_depth(
        self, depth_bgr: np.ndarray, masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> np.ndarray:
        """Annotate depth colormap with per-type overlays, bounding boxes, and labels."""
        return self._draw_instance_annotations(depth_bgr, masks, segment_ids)

    def _update_rgbd_window(
        self, obs_unbatched: Mapping[str, Any], masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> None:
        """Update side-by-side RGB/Depth CV2 preview."""
        if not (self._display_rgb or self._display_depth):
            return

        self._ensure_rgbd_window()

        panels: list[np.ndarray] = []
        if self._display_rgb:
            rgb = obs_unbatched["rgb"].detach().to("cpu").numpy().transpose((1, 2, 0))
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if self._segment_rgb:
                rgb_bgr = self._colorize_segmented_rgb(rgb_bgr, masks, segment_ids)
            panels.append(rgb_bgr)
        if self._display_depth:
            depth = obs_unbatched["depth"].detach().to("cpu").numpy()[0]
            depth_vis = depth
            if depth.dtype != np.uint16:
                depth_vis = (depth * 1000.0).astype(np.uint16)
            depth_bgr = depth_to_colormap_np(depth_vis)
            if self._segment_depth:
                depth_bgr = self._colorize_segmented_depth(depth_bgr, masks, segment_ids)
            panels.append(depth_bgr)

        if not panels:
            return
        frame = panels[0] if len(panels) == 1 else np.concatenate(panels, axis=1)
        cv2.imshow(self._rgbd_window_name, frame)
        cv2.waitKey(1)

    def _stop_rgbd(self) -> None:
        """Tear down the CV2 window."""
        if self._rgbd_active:
            with suppress(cv2.error):
                cv2.destroyWindow(self._rgbd_window_name)
            self._rgbd_active = False
        self._display_rgb = False
        self._display_depth = False

    def _ensure_rgbd_window(self) -> None:
        """Create the CV2 RGBD window lazily on the run-loop thread."""
        if self._rgbd_active:
            return
        if not (self._display_rgb or self._display_depth):
            return
        cv2.namedWindow(self._rgbd_window_name, cv2.WINDOW_NORMAL)
        self._rgbd_active = True

    def _maybe_unbatch(self, obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Convert observations to unbatched torch tensors."""
        rgb = obs["rgb"]
        depth = obs["depth"]
        intrinsic_k = obs["intrinsic_k"]
        camera_pose = obs["camera_pose"]

        if rgb.dim() == 4:
            rgb = rgb[0]
        if depth.dim() == 4:
            depth = depth[0]
        if intrinsic_k.dim() == 3:
            intrinsic_k = intrinsic_k[0]
        if camera_pose.dim() == 2:
            camera_pose = camera_pose[0]

        return {"rgb": rgb, "depth": depth, "intrinsic_k": intrinsic_k, "camera_pose": camera_pose}

    def _observation_to_point_cloud(
        self, obs_unbatched: Mapping[str, Any], masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a point cloud from an unbatched RGB-D observation."""
        point_cloud, segment_indices = segmented_rgbd_to_point_cloud(
            obs_unbatched["depth"],
            masks,
            obs_unbatched["intrinsic_k"],
            obs_unbatched["camera_pose"],
            rgb=obs_unbatched["rgb"],
            use_perspective=False,
        )
        if segment_ids.numel() > 0 and segment_indices.numel() > 0:
            segment_indices = segment_ids[segment_indices]
        return point_cloud, segment_indices

    def get_tcp_pos(self) -> Sequence[float]:
        return (
            self.env.get_observation(self.env.unwrapped.obs_spec_ikee.unbatched())["tcp_pose_b"][:3]
            .detach()
            .cpu()
            .numpy()
        )
