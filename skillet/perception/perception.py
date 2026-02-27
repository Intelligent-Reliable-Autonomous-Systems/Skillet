"""Perception class for the Robot Skills framework.

This class polls the environment for RGB-D observations and localizes objects in the scene.
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
import threading
import time
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np
import torch

from skillet.perception.object_localization import segmented_rgbd_to_point_cloud
from skillet.perception.realsense import RealsenseEnv
from skillet.perception.sam3.sam3 import SAM3
from skillet.perception.utils import depth_to_colormap_np
from skillet.perception.visualize import point_cloud_to_open3d

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from skillet.core import BatchedEnvironment
    from skillet.core.env import Environment
    from skillet.core.spaces import ObservationSpec

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]


class Perception:
    """Perception class for the Robot Skills framework.

    This class polls the environment for RGB-D observations and localizes objects in the scene.
    """

    def __init__(
        self,
        env: Environment | BatchedEnvironment,
        obs_spec: ObservationSpec,
        poll_rate: float,
        device: str = "cuda",
        max_depth_m: float | None = None,
        prompts: list[str] | None = None,
        sam3_model_path: str | None = None,
        segmentation_fn: Callable[[Mapping[str, Any]], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the perception poller and optional point-cloud visualizer."""
        self.env = env
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.obs_spec = replace(obs_spec, device=device, is_torch=True)
        self.poll_rate = poll_rate
        self.max_depth_m = max_depth_m
        self.prompts = prompts or []
        self.segmentation_fn = segmentation_fn
        self.sam3: SAM3 | None = None
        if self.prompts:
            self.sam3 = SAM3(model_path=sam3_model_path, device=str(self.device))

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.latest_observation: Mapping[str, Any] | None = None
        self.latest_point_cloud: torch.Tensor | None = None
        self.latest_segment_indices: torch.Tensor | None = None

        self._vis: Any | None = None
        self._vis_pcd: Any | None = None
        self._vis_active = False
        self._display_point_cloud = False
        self._display_rgb = False
        self._display_depth = False
        self._segment_point_cloud = False
        self._segment_rgb = False
        self._segment_depth = False
        self._cv2_window_name = "Perception RGB-D"
        self._vis_needs_reset = False
        self._vis_max_range_m = 0.0
        self._last_depth_debug_t = 0.0

    @staticmethod
    def _maybe_unbatch(obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
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

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": intrinsic_k,
            "camera_pose": camera_pose,
        }

    def _apply_far_plane(self, obs_unbatched: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Apply optional far-plane clipping in meters by zeroing distant depth."""
        if self.max_depth_m is None:
            return dict(obs_unbatched)

        depth = obs_unbatched["depth"].clone()
        if depth.dtype == torch.uint16:
            max_depth_native = int(self.max_depth_m * 1000.0)
            depth[depth > max_depth_native] = 0
        else:
            depth = depth.float()
            depth[depth > self.max_depth_m] = 0.0

        return {
            "rgb": obs_unbatched["rgb"],
            "depth": depth,
            "intrinsic_k": obs_unbatched["intrinsic_k"],
            "camera_pose": obs_unbatched["camera_pose"],
        }

    def _default_segmentation(self, obs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback segmentation: a single full-image mask."""
        depth = obs["depth"]
        masks = torch.ones((1, depth.shape[-2], depth.shape[-1]), dtype=torch.bool, device=depth.device)
        segment_ids = torch.zeros((1,), dtype=torch.int64, device=depth.device)
        return masks, segment_ids

    def _get_masks(self, obs_unbatched: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get segmentation masks and persistent segment IDs."""
        if self.segmentation_fn is not None:
            masks = self.segmentation_fn(obs_unbatched).to(torch.bool)
            segment_ids = torch.arange(masks.shape[0], dtype=torch.int64, device=masks.device)
            return masks, segment_ids
        if self.sam3 is not None and self.prompts:
            masks, instance_ids = self.sam3.predict(obs_unbatched["rgb"], self.prompts)
            return masks > 0, instance_ids.to(torch.int64)
        return self._default_segmentation(obs_unbatched)

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

    def _debug_depth_range(self, obs: Mapping[str, Any]) -> None:
        """Log depth range occasionally to help diagnose sensor cutoffs."""
        now = time.perf_counter()
        if now - self._last_depth_debug_t < 1.0:
            return

        depth = self._maybe_unbatch(obs)["depth"]
        valid = depth > 0
        if valid.any():
            depth_valid = depth[valid]
            depth_min = float(depth_valid.min().item())
            depth_max = float(depth_valid.max().item())
            if depth.dtype == torch.uint16:
                print(f"[Perception] depth range: {depth_min / 1000.0:.3f}m..{depth_max / 1000.0:.3f}m")
            else:
                print(f"[Perception] depth range: {depth_min:.3f}m..{depth_max:.3f}m ({depth.dtype})")
        else:
            print("[Perception] depth range: no valid depth values (>0)")
        self._last_depth_debug_t = now

    def run_thread(self) -> None:
        """Run the perception pipeline in a thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="PerceptionThread", daemon=True)
        self._thread.start()

    def start_visualization(
        self,
        display_point_cloud: bool = True,
        display_rgb: bool = False,
        display_depth: bool = False,
        segment_point_cloud: bool = False,
        segment_rgb: bool = False,
        segment_depth: bool = False,
        cv2_window_name: str = "Perception RGB-D",
    ) -> None:
        """Start Open3D and optional CV2 visualizations."""
        if display_point_cloud and o3d is None:
            raise ImportError("Open3D is required for visualization. Install with: pip install open3d")
        if self._vis_active or self._display_rgb or self._display_depth:
            return

        if display_point_cloud:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Perception Point Cloud", width=1024, height=768)
            pcd = o3d.geometry.PointCloud()
            vis.add_geometry(pcd)

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            vis.add_geometry(coord)

            self._vis = vis
            self._vis_pcd = pcd
            self._vis_active = True

        self._display_point_cloud = display_point_cloud
        self._display_rgb = display_rgb
        self._display_depth = display_depth
        self._segment_point_cloud = segment_point_cloud
        self._segment_rgb = segment_rgb
        self._segment_depth = segment_depth
        self._cv2_window_name = cv2_window_name
        self._vis_needs_reset = True
        self._vis_max_range_m = 0.0

        if self._display_rgb or self._display_depth:
            cv2.namedWindow(self._cv2_window_name, cv2.WINDOW_NORMAL)

    @staticmethod
    def _segment_palette(mask_idx: int) -> tuple[int, int, int]:
        """Return deterministic BGR color for a segment id."""
        r = int((mask_idx * 37) % 255)
        g = int((mask_idx * 57) % 255)
        b = int((mask_idx * 97) % 255)
        return b, g, r

    def _colorize_segmented_rgb(
        self, rgb_bgr: np.ndarray, masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> np.ndarray:
        """Color RGB image by segment ids."""
        out = rgb_bgr.copy()
        masks_np = masks.detach().to("cpu").numpy()
        segment_ids_np = segment_ids.detach().to("cpu").numpy()
        for mask_idx in range(masks_np.shape[0]):
            seg_mask = masks_np[mask_idx] > 0
            out[seg_mask] = self._segment_palette(int(segment_ids_np[mask_idx]))
        return out

    def _colorize_segmented_depth(
        self, depth_bgr: np.ndarray, masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> np.ndarray:
        """Color depth image by segment ids."""
        out = np.zeros_like(depth_bgr)
        masks_np = masks.detach().to("cpu").numpy()
        segment_ids_np = segment_ids.detach().to("cpu").numpy()
        for mask_idx in range(masks_np.shape[0]):
            seg_mask = masks_np[mask_idx] > 0
            out[seg_mask] = self._segment_palette(int(segment_ids_np[mask_idx]))
        return out

    def _update_rgbd_window(
        self, obs_unbatched: Mapping[str, Any], masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> None:
        """Update side-by-side RGB/Depth CV2 preview."""
        if not (self._display_rgb or self._display_depth):
            return

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
        cv2.imshow(self._cv2_window_name, frame)
        cv2.waitKey(1)

    def stop_visualization(self) -> None:
        """Stop the Open3D visualization."""
        if self._vis is not None:
            self._vis.destroy_window()
        if self._display_rgb or self._display_depth:
            with suppress(cv2.error):
                cv2.destroyWindow(self._cv2_window_name)
        self._vis = None
        self._vis_pcd = None
        self._vis_active = False
        self._display_point_cloud = False
        self._display_rgb = False
        self._display_depth = False
        self._segment_point_cloud = False
        self._segment_rgb = False
        self._segment_depth = False

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for the worker thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.stop_visualization()

    def run(self) -> None:
        """Run the perception pipeline."""
        poll_period_s = 1.0 / self.poll_rate
        next_poll_t = time.perf_counter()

        while not self._stop_event.is_set():
            obs = self.env.get_observation(self.obs_spec)
            obs_unbatched = self._apply_far_plane(self._maybe_unbatch(obs))
            masks, segment_ids = self._get_masks(obs_unbatched)
            point_cloud, segment_indices = self._observation_to_point_cloud(obs_unbatched, masks, segment_ids)

            with self._lock:
                self.latest_observation = obs
                self.latest_point_cloud = point_cloud
                self.latest_segment_indices = segment_indices

            if self._display_point_cloud and self._vis_active and self._vis is not None and self._vis_pcd is not None:
                # Color by instance ids if requested; otherwise use embedded RGB colors.
                vis_segment_indices = segment_indices if self._segment_point_cloud else None
                pcd = point_cloud_to_open3d(point_cloud, segment_indices=vis_segment_indices, filter_zero=True)
                if pcd is not None:
                    self._vis_pcd.points = pcd.points
                    self._vis_pcd.colors = pcd.colors
                    self._vis.update_geometry(self._vis_pcd)
                    if point_cloud.shape[0] > 0:
                        max_range_m = float(torch.linalg.vector_norm(point_cloud[:, :3], dim=1).max().item())
                        if self._vis_needs_reset or max_range_m > (1.25 * self._vis_max_range_m + 0.05):
                            # Refit the camera/frustum as range increases to avoid far-plane clipping.
                            self._vis.reset_view_point(True)
                            self._vis_needs_reset = False
                            self._vis_max_range_m = max(self._vis_max_range_m, max_range_m)
                    if not self._vis.poll_events():
                        self.stop_visualization()
                    else:
                        self._vis.update_renderer()

            self._update_rgbd_window(obs_unbatched, masks, segment_ids)
            self._debug_depth_range(obs)

            next_poll_t += poll_period_s
            sleep_s = max(0.0, next_poll_t - time.perf_counter())
            if sleep_s > 0:
                time.sleep(sleep_s)

        self.stop_visualization()

if __name__ == "__main__":
    env = RealsenseEnv()
    perception = Perception(env, env.obs_spec, 8, prompts=["wooden block", "plastic block"])
    perception.start_visualization(
        display_rgb=True, display_depth=True,
        segment_rgb=True, segment_depth=True,
        segment_point_cloud=True
    )
    perception.run()
