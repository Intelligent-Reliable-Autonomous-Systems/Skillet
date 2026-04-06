"""Perception class for the Skillet framework.

This class polls the environment for RGB-D observations and localizes objects in the scene.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from skillet.perception.reconstruction.sam_reconstructor import SAMReconstructor
from skillet.scene.utils import segmented_rgbd_to_point_cloud

if TYPE_CHECKING:
    from collections.abc import Mapping

    from skillet.core import BatchedEnvironment
    from skillet.core.env import Environment
    from skillet.core.spaces import ObservationSpec
    from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
    from skillet.scene.base import Scene


class SkilletPerception:
    """Skillet Perception class for the Skillet framework.

    This class polls the environment for RGB-D observations and localizes objects in the scene.
    """

    def __init__(
        self,
        env: Environment | BatchedEnvironment,
        scene: Scene,
        obs_spec: ObservationSpec,
        reconstructor: ReconstructorBase,
        poll_rate: float = 8,
        device: str | torch.device | None = None,
        max_depth_m: float | None = None,
    ) -> None:
        """Initialize the perception pipeline."""
        self.env = env
        self._scene = scene
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
        """Run the SkilletPerception pipeline in a thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="SkilletPerceptionThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for the worker thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def run(self) -> None:
        """Run the SkilletPerception pipeline."""
        if self._reconstructor is None:
            print("[INFO][PERCEPTION] Loading SAM reconstructor")
            self._reconstructor = SAMReconstructor(scene=self._scene)
        poll_period_s = 1.0 / self.poll_rate
        next_poll_t = time.perf_counter()

        while not self._stop_event.is_set():
            obs = self.env.get_observation(self.obs_spec)
            obs_unbatched = self._apply_far_plane(self._maybe_unbatch(obs))

            # Update the state based on reconstruction
            self._reconstructor.update_state(obs_unbatched, update=True)  # TODO: update only at specific Hz

            masks, segment_ids = self._get_masks(obs_unbatched)
            point_cloud, segment_indices = self._observation_to_point_cloud(obs_unbatched, masks, segment_ids)

            with self._lock:
                self.latest_observation = obs
                self.latest_point_cloud = point_cloud
                self.latest_segment_indices = segment_indices

            next_poll_t += poll_period_s
            sleep_s = max(0.0, next_poll_t - time.perf_counter())
            if sleep_s > 0:
                time.sleep(sleep_s)

        self.stop()
