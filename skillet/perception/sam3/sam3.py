"""SAM3 model wrappers for concept-based prompting.

Provides:
- ``SAM3``: single-image segmentation (no temporal tracking).
- ``SAM3VideoTracker``: online/streaming video tracker that maintains
  persistent object IDs across frames using SAM-3's memory-conditioned
  detection + tracking pipeline.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
except ImportError:
    print("Failed to import Ultralytics SAM3. Please install ultralytics:", file=sys.stderr)
    print("  pip install ultralytics", file=sys.stderr)
    raise


class SAM3:
    """SAM3 model for image segmentation with concept-based prompting.

    SAM3 is used one image at a time. This wrapper is designed for heterogeneous
    observations: call `set_image()` for each new RGB frame, then `segment()` with
    one or more concept prompts.

    Example:
        >>> sam3 = SAM3(model_path="data/models/sam3.pt")
        >>> prompts = ["person", "car", "bicycle"]
        >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # (H, W, 3)
        >>> masks, prompt_indices = sam3.predict(image, prompts)
        >>> # masks: (M, H, W) torch.uint8 on GPU
        >>> # prompt_indices: (M,) torch.int64 on GPU; values index into `prompts`

    """

    def __init__(
        self,
        model_path: str | Path | None = None,  # type: ignore[type-arg]
        conf: float = 0.25,
        half: bool = True,
        device: str | None = None,
    ) -> None:
        """Initialize SAM3 predictor.

        Args:
            model_path: Path to sam3.pt weights file. Defaults to `data/sam3/sam3.pt`
                relative to the repo root.
            conf: Confidence threshold for detections.
            half: Use FP16 for faster inference when supported.
            device: Device to use (e.g., 'cuda:0', 'cpu'). None lets Ultralytics decide.

        """
        if model_path is None:
            # skillet/perception/sam3/sam3.py -> repo root at parents[3]
            repo_root = Path(__file__).resolve().parents[3]
            model_path = repo_root / "data" / "models" / "sam3.pt"

        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
            "half": half,
            "device": device or None,
            "save": False,
            "verbose": False,
        }
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.conf = conf
        self.device = device
        self._image_hw: tuple[int, int] | None = None  # (H, W) of last set image

    def set_image(self, image: np.ndarray | torch.Tensor) -> None:
        """Set the current RGB image for subsequent `segment()` calls.

        Args:
            image: RGB image of shape (3, H, W) as uint8 numpy array or torch tensor.

        """
        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            # Keep on model device by default
            device = next(self.predictor.model.parameters()).device
            image_tensor = torch.as_tensor(image, device=device)

        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError(f"Expected image shape (3, H, W), got {tuple(image_tensor.shape)}")
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0
        h, w = int(image_tensor.shape[1]), int(image_tensor.shape[2])
        self._image_hw = (h, w)
        # self.predictor.set_image(image_tensor.unsqueeze(0))
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
        image_np = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.predictor.set_image(image_np)

    def segment(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment all instances of the given concept prompts on the currently set image.

        Returns:
            - masks: (M, H, W) torch.uint8 on model device
            - prompt_indices: (M,) torch.int64 on model device; indexes into `prompts`

        """
        if self._image_hw is None:
            raise RuntimeError("No image set. Call set_image(image) before segment(prompts).")

        results = self.predictor(text=prompts)
        h, w = self._image_hw
        device = next(self.predictor.model.parameters()).device

        if not results:
            return (
                torch.zeros((0, h, w), dtype=torch.uint8, device=device),
                torch.zeros((0,), dtype=torch.int64, device=device),
            )

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return (
                torch.zeros((0, h, w), dtype=torch.uint8, device=device),
                torch.zeros((0,), dtype=torch.int64, device=device),
            )

        masks = result.masks.data  # (M, H', W') float/bool tensor on model device
        m = int(masks.shape[0])
        prompt_indices_np = self._get_prompt_indices(result, m, prompts)
        prompt_indices = torch.as_tensor(prompt_indices_np, dtype=torch.int64, device=masks.device)

        if masks.shape[1] != h or masks.shape[2] != w:
            masks = self._resize_masks_torch(masks, h, w)

        masks = (masks > 0.5).to(torch.uint8)
        return masks, prompt_indices

    def predict(
        self,
        image: np.ndarray | torch.Tensor,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Set image then segment with prompts."""
        self.set_image(image)
        return self.segment(prompts)

    def _get_prompt_indices(
        self,
        result: Any,  # noqa: ANN401
        num_masks: int,
        prompts: list[str],
    ) -> np.ndarray:
        """Map each mask to its prompt index.

        Args:
            result: SAM3 Results object.
            num_masks: Number of masks in this result.
            prompts: List of prompt strings.

        Returns:
            Array of shape (num_masks,) with prompt indices.

        """
        # result.names can be dict (class_id -> prompt) or list (index -> prompt)
        names = result.names
        boxes = result.boxes

        prompt_indices = []

        for i in range(num_masks):
            # Get class ID for this mask
            cls_id = int(boxes.cls[i].item()) if boxes.cls is not None and i < len(boxes.cls) else i

            # Map class ID to prompt
            if isinstance(names, dict):
                prompt_text = names.get(cls_id, "")
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                prompt_text = str(names[cls_id])
            else:
                prompt_text = ""

            # Find prompt index
            try:
                prompt_idx = prompts.index(prompt_text)
            except ValueError:
                # If prompt not found, try to match by index
                # SAM3 may return masks in prompt order
                prompt_idx = cls_id if cls_id < len(prompts) else 0

            prompt_indices.append(prompt_idx)

        return np.array(prompt_indices, dtype=np.int64)
    def _resize_masks_torch(
        self,
        masks: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        """Resize masks on GPU. Input (N, H, W), output (N, target_height, target_width)."""
        # F.interpolate expects (N, C, H, W)
        x = masks.unsqueeze(1).float()  # (N, 1, H, W)
        x = F.interpolate(
            x,
            size=(target_height, target_width),
            mode="nearest",
        )
        return x.squeeze(1)  # (N, target_height, target_width)

    def _resize_masks(self, masks: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Resize masks (numpy, CPU). Masks (N, H, W) -> (N, target_height, target_width)."""
        import cv2

        resized = []
        for mask in masks:
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
            resized_mask = cv2.resize(
                mask_uint8,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            resized.append(resized_mask.astype(np.float32) / 255.0)
        return np.stack(resized, axis=0)


class SAM3VideoTracker:
    """Online/streaming SAM-3 video tracker with persistent object IDs.

    Wraps ``SAM3VideoSemanticPredictor`` so that frames can be fed one at a
    time (e.g. from a ROS topic or webcam) instead of requiring a complete
    video file up-front.

    Internally the Ultralytics predictor's ``_run_single_frame_inference``
    method is invoked directly, bypassing the dataset/batch loop.

    Example:
        >>> tracker = SAM3VideoTracker(prompts=["cup", "bottle"])
        >>> for rgb_frame in camera:                       # (H, W, 3) uint8 RGB
        ...     obj_ids, masks, cls_indices = tracker.track(rgb_frame)
        ...     # obj_ids:     (M,) int64 - persistent IDs across frames
        ...     # masks:       (M, H, W) bool
        ...     # cls_indices: (M,) int64 - indexes into prompts list

    """

    _MAX_FRAMES = 10_000_000

    def __init__(
        self,
        prompts: list[str],
        model_path: str | Path | None = None,
        imgsz: int = 1008,
        conf: float = 0.25,
        half: bool = True,
        device: str | None = None,
    ) -> None:
        """Initialize online SAM-3 video tracker.

        Args:
            prompts: Text concept prompts (e.g. ``["cup", "bottle"]``).
            model_path: Path to ``sam3.pt``. Defaults to ``data/models/sam3.pt``.
            imgsz: Square input resolution for the model.
            conf: Confidence threshold for detections.
            half: Use FP16 inference.
            device: Torch device string (e.g. ``"cuda:0"``).

        """
        if model_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            model_path = repo_root / "data" / "models" / "sam3.pt"

        self.prompts = prompts
        self._frame_idx = 0

        overrides: dict[str, Any] = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "imgsz": imgsz,
            "model": str(model_path),
            "half": half,
            "device": device or None,
            "save": False,
            "verbose": False,
        }
        self._pred = SAM3VideoSemanticPredictor(overrides=overrides)
        self._pred.setup_model()
        self._setup_geometry(imgsz)
        self._initialized = False

    def _setup_geometry(self, imgsz: int) -> None:
        """Configure image sizes and backbone feature map sizes."""
        sz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        self._pred.imgsz = sz
        self._pred.tracker.imgsz = sz
        self._pred.tracker.model.set_imgsz(sz)
        stride = self._pred.stride
        self._pred.tracker._bb_feat_sizes = [
            [int(x / (stride * i)) for x in sz] for i in [1 / 4, 1 / 2, 1]
        ]
        self._pred.interpol_size = (
            self._pred.tracker.model.memory_encoder.mask_downsampler.interpol_size
        )

    def _init_state(self) -> None:
        """Manually build inference_state, bypassing the dataset requirement."""
        self._pred.inference_state = {
            "num_frames": self._MAX_FRAMES,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "text_prompt": None,
            "per_frame_geometric_prompt": defaultdict(lambda: None),
        }
        self._initialized = False
        self._frame_idx = 0

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Letterbox + normalize an HWC uint8 RGB numpy image to a BCHW tensor.

        Replicates the SAM3SemanticPredictor preprocessing chain:
        ``LetterBox(scale_fill=True)`` -> BGR flip -> normalize ``(x - 127.5) / 127.5``.
        """
        from ultralytics.data.augment import LetterBox

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        letterbox = LetterBox(self._pred.imgsz, auto=False, center=False, scale_fill=True)
        im = letterbox(image=image)

        im = im[..., ::-1].transpose((2, 0, 1))  # HWC RGB -> CHW BGR
        im = np.ascontiguousarray(im)
        im_t = torch.from_numpy(im).to(self._pred.device)
        im_t = (im_t - self._pred.mean) / self._pred.std
        im_t = im_t.half() if self._pred.model.fp16 else im_t.float()
        return im_t.unsqueeze(0)  # (1, 3, H, W)

    @torch.inference_mode()
    def track(
        self,
        image: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one frame and return tracked objects.

        Args:
            image: RGB image as ``(H, W, 3)`` uint8 numpy array **or**
                ``(3, H, W)`` torch tensor (uint8 or float [0, 1]).

        Returns:
            obj_ids: ``(M,)`` int64 tensor of persistent object IDs.
            masks:   ``(M, H, W)`` bool tensor at original image resolution.
            cls_indices: ``(M,)`` int64 tensor indexing into ``self.prompts``.

        """
        if isinstance(image, torch.Tensor):
            image = self._tensor_to_hwc_uint8(image)

        orig_h, orig_w = image.shape[:2]
        im = self._preprocess_image(image)

        if not self._initialized:
            self._init_state()

        state = self._pred.inference_state
        state["im"] = im

        # Fake self.batch so that add_prompt / _prepare_geometric_prompts can
        # read the original image shape from self.batch[1][0].shape[:2].
        self._pred.batch = (
            ["<online>"],    # paths
            [image],         # original images (list of HWC numpy)
            [""],            # string descriptors
        )

        if not self._initialized:
            # add_prompt runs _run_single_frame_inference internally
            _, out = self._pred.add_prompt(
                frame_idx=self._frame_idx,
                text=self.prompts,
            )
            self._initialized = True
        else:
            out = self._pred._run_single_frame_inference(
                self._frame_idx, reverse=False
            )

        self._frame_idx += 1

        return self._build_result(out, orig_h, orig_w)

    def _build_result(
        self, out: dict[str, Any], orig_h: int, orig_w: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw predictor output into (obj_ids, masks, cls_indices)."""
        obj_id_to_mask = out["obj_id_to_mask"]
        obj_id_to_score = out["obj_id_to_score"]
        obj_id_to_cls = out.get("obj_id_to_cls", {})
        device = self._pred.device
        conf = self._pred.args.conf

        if not obj_id_to_mask:
            return (
                torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, orig_h, orig_w, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
            )

        sorted_ids = sorted(obj_id_to_mask.keys())
        low_res = torch.cat([obj_id_to_mask[oid] for oid in sorted_ids], dim=0)
        masks_full = (
            F.interpolate(
                low_res.float().unsqueeze(0), (orig_h, orig_w), mode="bilinear"
            )[0]
            > 0.5
        )

        scores = torch.tensor(
            [obj_id_to_score[oid] for oid in sorted_ids], device=device
        )
        cls_raw = torch.tensor(
            [obj_id_to_cls.get(oid, 0) for oid in sorted_ids],
            dtype=torch.int64,
            device=device,
        )
        ids_t = torch.tensor(sorted_ids, dtype=torch.int64, device=device)

        keep = (scores > conf) & masks_full.any(dim=(1, 2))
        return ids_t[keep], masks_full[keep], cls_raw[keep]

    def reset(self) -> None:
        """Reset tracker state (call when scene changes)."""
        self._init_state()

    @staticmethod
    def _tensor_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
        t = image.detach()
        if t.ndim == 3 and t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        if t.dtype != torch.uint8:
            t = (t.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return t.cpu().numpy()

