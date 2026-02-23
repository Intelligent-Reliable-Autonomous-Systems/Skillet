"""SAM3 model wrapper for concept-based prompting (single image at a time)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
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
