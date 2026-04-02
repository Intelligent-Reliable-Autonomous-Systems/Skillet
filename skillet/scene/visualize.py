"""Visualization class for the Robot Skills framework.

This class polls the environment for RGB-D observations and localizes objects in the scene.
"""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

from skillet.perception.utils import depth_to_colormap_np
from skillet.scene.utils import segmented_rgbd_to_point_cloud

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from skillet.core import BatchedEnvironment
    from skillet.core.env import Environment
    from skillet.core.spaces import ObservationSpec
    from skillet.perception.localization.reconstructor_base import ReconstructorBase
    from skillet.scene.scene_visualization import Open3DVisualizer

_PALETTE_BGR: list[tuple[int, int, int]] = [
    (44, 44, 220),
    (44, 190, 44),
    (220, 110, 44),
    (0, 190, 240),
    (200, 44, 200),
    (210, 210, 44),
    (0, 130, 255),
    (170, 44, 240),
    (44, 240, 160),
    (240, 160, 44),
]
_OVERLAY_ALPHA = 0.35
_BBOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1


class SkilletVisualizer:
    """Skillet Visualizer class for the Skillet framework.

    This class polls the environment for RGB-D observations and localizes objects in the scene.
    """

    def __init__(
        self,
        env: Environment | BatchedEnvironment,
        obs_spec: ObservationSpec,
        reconstructor: ReconstructorBase,
        poll_rate: float = 8,
        device: str | torch.device | None = None,
        max_depth_m: float | None = None,
    ) -> None:
        """Initialize the visualizer."""
        self.env = env
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or obs_spec.device
        self.obs_spec = replace(obs_spec, device=self.device, is_torch=True)
        self.poll_rate = poll_rate
        self.max_depth_m = max_depth_m

        # Scene reconstructor
        self._reconstructor = reconstructor

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.latest_observation: Mapping[str, Any] | None = None
        self.latest_point_cloud: torch.Tensor | None = None
        self.latest_segment_indices: torch.Tensor | None = None

        self._pc_vis: Open3DVisualizer | None = None
        self._segment_point_cloud = False

        self._display_rgb = False
        self._display_depth = False
        self._segment_rgb = False
        self._segment_depth = False
        self._cv2_window_name = "Perception RGB-D"
        self._cv2_active = False
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

    def run_thread(self) -> None:
        """Run the SkilletVisualizer pipeline in a thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="SkilletVisualizerThread", daemon=True)
        self._thread.start()

        if self._pc_vis is not None:
            self._pc_vis.run_thread()

    def set_open3d_visualizer(
        self,
        vis: Open3DVisualizer | None = None,
        segment_point_cloud: bool = False,
    ) -> None:
        """Attach an external :class:`PointCloudVisualizer` for 3-D rendering.

        The visualizer's ``run()`` should be called separately on the main
        thread; this method just stores the reference so the main visualization loop
        can push updates via ``vis.update()``.
        """

        def get_tcp_pos() -> Sequence[float]:
            return self.env.get_observation(self.env.ikee_spec.unbatched())["tcp_pose_b"][:3].detach().cpu().numpy()

        self._pc_vis = vis or Open3DVisualizer(self._reconstructor.scene, get_tcp_pos=get_tcp_pos)
        self._segment_point_cloud = segment_point_cloud

    def start_cv2_visualization(
        self,
        display_rgb: bool = False,
        display_depth: bool = False,
        segment_rgb: bool = False,
        segment_depth: bool = False,
        cv2_window_name: str = "Perception RGB-D",
    ) -> None:
        """Enable the CV2 RGB-D preview window."""
        self._display_rgb = display_rgb
        self._display_depth = display_depth
        self._segment_rgb = segment_rgb
        self._segment_depth = segment_depth
        self._cv2_window_name = cv2_window_name

    def _ensure_cv2_window(self) -> None:
        """Create the CV2 window lazily on the run-loop thread."""
        if self._cv2_active:
            return
        if not (self._display_rgb or self._display_depth):
            return
        cv2.namedWindow(self._cv2_window_name, cv2.WINDOW_NORMAL)
        self._cv2_active = True

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
            prompt_idx = int(ids_np[i])
            color = _PALETTE_BGR[i % len(_PALETTE_BGR)]

            ys, xs = np.where(seg_mask)
            if len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cv2.rectangle(out, (x1, y1), (x2, y2), color, _BBOX_THICKNESS)

            name = self._prompt_names[prompt_idx] if prompt_idx < len(self._prompt_names) else f"obj_{prompt_idx}"
            label = f"#{i} {name}"
            (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
            tx, ty = x1, y1 - 6
            if ty - th < 0:
                ty = y1 + th + 6
            cv2.rectangle(out, (tx - 1, ty - th - 4), (tx + tw + 5, ty + 4), color, cv2.FILLED)
            cv2.putText(
                out,
                label,
                (tx + 2, ty),
                _FONT,
                _FONT_SCALE,
                (255, 255, 255),
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )

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

    def _stop_cv2(self) -> None:
        """Tear down the CV2 window."""
        if self._cv2_active:
            with suppress(cv2.error):
                cv2.destroyWindow(self._cv2_window_name)
            self._cv2_active = False
        self._display_rgb = False
        self._display_depth = False

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for the worker thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._stop_cv2()
        if self._pc_vis is not None:
            self._pc_vis.request_close()

    def run(self) -> None:
        """Run the SkilletVisualizer pipeline."""
        poll_period_s = 1.0 / self.poll_rate
        next_poll_t = time.perf_counter()

        while not self._stop_event.is_set():
            obs = self.env.get_observation(self.obs_spec)
            obs_unbatched = self._apply_far_plane(self._maybe_unbatch(obs))

            # Update the state based on reconstruction
            self._reconstructor.update_state(obs_unbatched, update=False)

            masks, segment_ids = self._get_masks(obs_unbatched)
            point_cloud, segment_indices = self._observation_to_point_cloud(obs_unbatched, masks, segment_ids)

            with self._lock:
                self.latest_observation = obs
                self.latest_point_cloud = point_cloud
                self.latest_segment_indices = segment_indices

            if self._pc_vis is not None:
                vis_seg = segment_indices if self._segment_point_cloud else None
                self._pc_vis.update(
                    point_cloud,
                    segment_indices=vis_seg,
                    camera_pose=obs_unbatched["camera_pose"],
                )

            self._ensure_cv2_window()
            self._update_rgbd_window(obs_unbatched, masks, segment_ids)

            next_poll_t += poll_period_s
            sleep_s = max(0.0, next_poll_t - time.perf_counter())
            if sleep_s > 0:
                time.sleep(sleep_s)

        self.stop()
