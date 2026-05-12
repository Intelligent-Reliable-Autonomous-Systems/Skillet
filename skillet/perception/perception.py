"""Perception class for the Skillet framework.

This class polls the environment for RGB-D observations and localizes objects in the scene.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import torch

from skillet.perception.reconstruction.apriltag_reconstructor import ApriltagStateReconstructor
from skillet.perception.reconstruction.sam_reconstructor import Sam3Reconstructor
from skillet.perception.reconstruction.sam_vlm_reconstructor import SamVlmReconstructor
from skillet.perception.reconstruction.vlm_reconstructor import VlmReconstructor
from skillet.scene.utils import arrange_panels, depth_to_colormap_np, segmented_rgbd_to_point_cloud

if TYPE_CHECKING:
    from collections.abc import Mapping

    from skillet.core import BatchedEnvironment
    from skillet.core.env import Environment
    from skillet.core.spaces import ObservationSpec
    from skillet.scene.base import Scene
    from skillet.scene.visualization import Open3DVisualizer


class SkilletPerception:
    """Skillet Perception class for the Skillet framework.

    This class polls the environment for RGB-D observations and localizes objects in the scene.
    """

    def __init__(
        self,
        env: Environment | BatchedEnvironment,
        scene: Scene,
        obs_spec: ObservationSpec,
        reconstructor: Literal["sam3", "april", "vlm", "sam+vlm"] = "april",
        poll_rate_hz: float = 10,
        device: str | torch.device | None = None,
        max_depth_m: float | None = None,
        build_scene: bool = False,
        vis_perception: bool = False,
    ) -> None:
        """Initialize the perception pipeline."""
        self.env = env
        self._scene = scene
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or obs_spec.device
        self.obs_spec = replace(obs_spec, device=self.device, is_torch=True)
        self.poll_rate_hz = poll_rate_hz
        self.max_depth_m = max_depth_m

        # Scene reconstructor
        self._reconstructor_type = reconstructor
        self._reconstructor = None
        self._viz: Open3DVisualizer = None
        self._visualize_perception = vis_perception
        self._display_rgb = True
        self._display_depth = False
        self._perception_window_name = "Skillet Perception Scene"
        self._perception_width = 1200
        self._perception_height = 900
        self._perception_window_active = False
        self._perception_frame: np.ndarray = None
        self._build_scene = build_scene
        self._task_instruction = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.latest_observation: Mapping[str, Any] | None = None
        self.latest_point_cloud: torch.Tensor | None = None
        self.latest_segment_indices: torch.Tensor | None = None

        self._last_depth_debug_t = 0.0

    @property
    def build_scene(self) -> bool:
        return self._build_scene

    @build_scene.setter
    def build_scene(self, build_scene: bool) -> None:
        self._build_scene = build_scene
        if self._reconstructor is not None:
            self._reconstructor.build_scene = build_scene

    @property
    def scene(self) -> Scene:
        return self._scene or (self._reconstructor.scene if self._reconstructor is not None else None)

    @property
    def bbox_frame(self) -> np.ndarray:
        return self._reconstructor._bbox_frame if self._reconstructor is not None else None

    @property
    def mask_frame(self) -> np.ndarray:
        return self._reconstructor._mask_frame if self._reconstructor is not None else None

    @property
    def open3d_scene(self) -> np.ndarray:
        return self._viz.open3d_scene if self._viz is not None else None

    @property
    def vlm_frame(self) -> np.ndarray:
        return self._reconstructor._vlm_frame if self._reconstructor is not None else None

    @property
    def perception_frame(self) -> np.ndarray:
        return self._perception_frame

    @property
    def task_instruction(self) -> str:
        return self._task_instruction

    @task_instruction.setter
    def task_instruction(self, task: str) -> None:
        """Set the task for the reconstructor."""
        self._task_instruction = task

    @staticmethod
    def _maybe_unbatch(obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Convert observations to unbatched torch tensors."""
        rgb = obs["rgb"]
        depth = obs["depth"]
        intrinsic_k = obs["intrinsic_k"]
        camera_pose = obs["camera_pose"]
        tcp_pose = None
        gripper_pos = None

        if rgb.dim() == 4:
            rgb = rgb[0]
        if depth.dim() == 4:
            depth = depth[0]
        if intrinsic_k.dim() == 3:
            intrinsic_k = intrinsic_k[0]
        if camera_pose.dim() == 2:
            camera_pose = camera_pose[0]
        if "tcp_pose_b" in obs and obs["tcp_pose_b"].dim() == 2:
            tcp_pose_b = obs["tcp_pose_b"][0]
        if "gripper" in obs and obs["gripper"].dim() == 2:
            gripper = obs["gripper"][0]

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": intrinsic_k,
            "camera_pose": camera_pose,
            "tcp_pose_b": tcp_pose_b,
            "gripper": gripper,
        }

    def set_visualizer(
        self,
        vis,
        segment_point_cloud: bool = False,
    ) -> None:
        """Attach an external :class:`PointCloudVisualizer` for 3-D rendering.

        The visualizer's ``run()`` should be called separately on the main
        thread; this method just stores the reference so the perception loop
        can push updates via ``vis.update()``.
        """
        self._viz = vis
        self._segment_point_cloud = segment_point_cloud

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
            "tcp_pose_b": obs_unbatched["tcp_pose_b"],
            "gripper": obs_unbatched["gripper"],
        }

    def _observation_to_point_cloud(
        self, obs_unbatched: Mapping[str, Any], masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a point cloud from an unbatched RGB-D observation."""
        if masks is None:
            depth = obs_unbatched["depth"]
            masks = torch.ones((1, depth.shape[-2], depth.shape[-1]), dtype=torch.bool, device=depth.device)
            segment_ids = torch.zeros((1,), dtype=torch.int64, device=depth.device)
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

    def update_state(self) -> None:
        """Update the perception state."""
        self._build_reconstructor()
        obs = self.env.get_observation(self.obs_spec)
        obs_unbatched = self._apply_far_plane(self._maybe_unbatch(obs))

        # Update the state based on reconstruction
        self._reconstructor.update_state(obs_unbatched, update=True)
        self.scene.tcp_pose = obs_unbatched["tcp_pose_b"]
        self.scene.gripper_pos = obs_unbatched["gripper"]

    def run(self) -> None:
        """Run the SkilletPerception pipeline."""
        self._build_reconstructor()
        poll_period_s = 1.0 / self.poll_rate_hz
        next_poll_t = time.perf_counter()

        while not self._stop_event.is_set():
            self._reconstructor.task_instruction = self._task_instruction
            obs = self.env.get_observation(self.obs_spec)
            obs_unbatched = self._apply_far_plane(self._maybe_unbatch(obs))

            # Update the state based on reconstruction
            self._reconstructor.update_state(obs_unbatched, update=True)

            self.scene.tcp_pose = obs_unbatched["tcp_pose_b"]
            self.scene.gripper_pos = obs_unbatched["gripper"]

            point_cloud, segment_indices = self._observation_to_point_cloud(
                obs_unbatched, self._reconstructor.masks, self._reconstructor.segment_indices
            )

            with self._lock:
                self.latest_observation = obs
                self.latest_point_cloud = point_cloud
                self.latest_segment_indices = segment_indices

            if self._viz is not None:
                vis_seg = segment_indices if self._segment_point_cloud else None
                self._viz.update(
                    point_cloud,
                    segment_indices=vis_seg,
                    camera_pose=obs_unbatched["camera_pose"],
                )
            if self._visualize_perception:
                self._update_perception_window(obs_unbatched)
            sleep_time = (time.perf_counter() - next_poll_t) - poll_period_s
            if sleep_time < 0:
                time.sleep(min(-sleep_time, poll_period_s))
            else:
                ...
                # print(f"[WARN][PERCEPT] full loop overran by {sleep_time * 1000:.1f}ms")
            next_poll_t = time.perf_counter()
        self.stop()

    def _update_perception_window(self, obs_unbatched: Mapping[str, Any]) -> None:
        """Update side-by-side RGB/Depth CV2 preview."""
        if not (self._display_rgb or self._display_depth):
            return
        self._ensure_perception_window()
        panels: list[np.ndarray] = []
        if self._display_rgb:
            rgb = obs_unbatched["rgb"].detach().to("cpu").numpy().transpose((1, 2, 0))
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            panels.append(rgb_bgr)
        if self._display_depth:
            depth = obs_unbatched["depth"].detach().to("cpu").numpy()[0]
            depth_vis = depth
            if depth.dtype != np.uint16:
                depth_vis = (depth * 1000.0).astype(np.uint16)
            depth_bgr = depth_to_colormap_np(depth_vis)
            panels.append(depth_bgr)
        h, w, _ = rgb.shape
        (
            panels.append(cv2.resize(self._viz.open3d_scene, (w, h)))
            if (self._viz is not None and self._viz.open3d_scene is not None)
            else None
        )
        panels.append(self.bbox_frame) if self.bbox_frame is not None else None
        panels.append(self.mask_frame) if self.mask_frame is not None else None
        if not panels:
            return
        frame = arrange_panels(panels)
        self._perception_frame = frame

        cv2.imshow(self._perception_window_name, frame)
        cv2.waitKey(1)

    def _ensure_perception_window(self) -> None:
        """Create the CV2 RGBD window lazily on the run-loop thread."""
        if self._perception_window_active:
            return
        if not (self._display_rgb or self._display_depth):
            return
        cv2.namedWindow(self._perception_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._perception_window_name, self._perception_width, self._perception_height)
        self._perception_window_active = True

    def _build_reconstructor(self) -> None:
        """Build the reconstructor."""
        if self._reconstructor is None:
            if self._reconstructor_type == "sam+vlm":
                print("[INFO][PERCEPTION] Loading SAM reconstructor")
                self._reconstructor = SamVlmReconstructor(scene=self._scene, device=self.device)
            elif self._reconstructor_type == "april":
                print("[INFO][PERCEPTION] Loading AprilTag reconstructor")
                assert (
                    self._scene is not None
                ), "[ERROR] Perception Scene cannot be None when using AprilTagStateReconstructor."
                self._reconstructor = ApriltagStateReconstructor(self._scene)
            elif self._reconstructor_type == "sam3":
                self._reconstructor = Sam3Reconstructor(self._scene, device=self.device)
            elif self._reconstructor_type == "vlm":
                self._reconstructor = VlmReconstructor(scene=self._scene)
