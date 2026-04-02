import base64
import io
import pathlib
from abc import ABC, abstractmethod
from functools import cache

import numpy as np
import requests
from jaxtyping import Float
from PIL import Image


class SAMClient(ABC):
    """Base class for all SAM clients."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        mode: str = "local",
        remote_url: str | None = None,
    ) -> None:
        """Initialize the base SAM client.

        Args:
            model_name: Name of SAM model
            device: Device to load model on
            mode: If to run on remote server or locally
            remote_url: Remote URL to run server

        """
        self.device = device
        self.mode = mode
        self.remote_url = remote_url
        self.model_name = model_name

        self.model_path = self._download_sam_checkpoint(model_name)
        self.sam_model = self._load_sam_model(checkpoint=self.model_path)

    def segment_objects(
        self,
        rgb_pil: Image.Image,
        detection_results: list[dict],
    ) -> Float[np.ndarray, "n 1 h w"]:
        """Segment detection results from VLM with SAM3.

        Args:
            rgb_pil: PIL Image to segment.
            detection_results: List of detection dicts from VLM, each with a 'box_2d' key
                            in [ymin, xmin, ymax, xmax] format normalized to 0-1000.

        Returns:
            Segmentation masks of shape (N, 1, H, W).

        """
        boxes = self._convert_bounding_boxes(rgb_pil, detection_results)

        if self.mode == "local":
            masks, _ = self._segment_local(rgb_pil, boxes)
        else:
            masks, _ = self._segment_remote(rgb_pil, boxes, self.remote_url)

        if masks.ndim == 3:
            masks = masks[None]
        return masks

    def _segment_remote(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM segmentation via remote server."""
        assert self.remote_url is not None
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = {"image_base64": base64.b64encode(buffer.getvalue()).decode(), "boxes": boxes.tolist()}

        try:
            response = requests.post(f"{self.remote_url}/segment", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            masks = np.array(
                [[np.load(io.BytesIO(base64.b64decode(m))) for m in mask_batch] for mask_batch in result["masks"]]
            )
            return masks, np.array(result["scores"])

        except Exception as e:
            print(f"[SAM][ERROR] Remote SAM segmentation failed: {e}")
            raise e

    @abstractmethod
    def _segment_local(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM segmentation locally.

        Args:
            image: PIL image to segment
            boxes: bounding boxes

        Returns:
            Masks of segmented objects and confidence scores

        """
        raise NotImplementedError

    @abstractmethod
    def _convert_bounding_boxes(self, rgb_pil: Image.Image, detection_results: list[dict]) -> np.ndarray:
        """Convert bounding boxes into required SAM format.

        Args:
            rgb_pil: RGB image to segment.
            detection_results: dictionary list of segmentation results from VLM.

        Returns:
            np.ndarray of bounding boxes

        """
        raise NotImplementedError

    @abstractmethod
    def _download_sam_checkpoint(self, model_name: str | None = None) -> pathlib.Path:
        """Download the SAM model.

        Args:
            model_name: name of the SAM model

        Returns:
            Path to the BPE file.

        """
        raise NotImplementedError

    @abstractmethod
    @cache
    def _load_sam_model(self, checkpoint: str):  # noqa: ANN202
        """Load and cache the SAM2 image predictor."""
        raise NotImplementedError
