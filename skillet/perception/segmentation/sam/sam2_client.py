"""SAM2 segmentation — local predictor and remote HTTP client."""

import base64
import io
import os
import pathlib
from functools import cache

import numpy as np
import requests
from jaxtyping import Float
from PIL import Image
from tqdm import tqdm

from skillet.perception.utils import get_skillet_model_cache_dir

_SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


class SAM2Client:
    """Main client class for the SAM3 model."""

    def __init__(
        self,
        model_name: str = "sam2.1_hiera_large.pt",
        device: str = "cuda",
        mode: str = "local",
        remote_url: str | None = None,
    ) -> None:
        self.device = device
        self.mode = mode
        self.remote_url = remote_url
        self.model_path = self._download_sam_checkpoint(model_name)

        self.sam_model = self._load_sam_model(checkpoint=self.model_path)

    def segment_objects(
        self,
        rgb_pil: Image.Image,
        detection_results: list[dict],
    ) -> Float[np.ndarray, "n 1 h w"]:
        """Segment detection results from VLM with SAM2.

        Args:
            rgb_pil: PIL Image to segment.
            detection_results: List of detection dicts from VLM, each with a 'box_2d' key
                            in [ymin, xmin, ymax, xmax] format normalized to 0-1000.

        Returns:
            Segmentation masks of shape (N, 1, H, W).

        """
        # Convert VLM bbox format [ymin, xmin, ymax, xmax] (0-1000) to SAM2 [x0, y0, x1, y1] (pixels)
        img_height, img_width = rgb_pil.height, rgb_pil.width
        boxes = np.array(
            [
                [
                    (xmin / 1000.0) * img_width,
                    (ymin / 1000.0) * img_height,
                    (xmax / 1000.0) * img_width,
                    (ymax / 1000.0) * img_height,
                ]
                for detection in detection_results
                if len(box_2d := detection.get("box_2d", [])) == 4
                for ymin, xmin, ymax, xmax in [box_2d]
            ]
        )

        if self.mode == "local":
            masks, _ = self._segment_local(rgb_pil, boxes)
        else:
            masks, _ = self._segment_remote(rgb_pil, boxes, self.remote_url)

        if masks.ndim == 3:
            masks = masks[None]
        return masks

    def _segment_local(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM2 segmentation locally.

        Args:
            image: PIL image to segment
            boxes: bounding boxes

        Returns:
            Masks of segmented objects and confidence scores

        """
        self.sam_model.set_image(image)
        masks, scores, _ = self.sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        return masks, scores

    def _segment_remote(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM2 segmentation via remote server."""
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
            print(f"[SAM][ERROR] Remote SAM2 segmentation failed: {e}")
            raise e

    def _download_sam_checkpoint(self, model_name: str = "sam2.1_hiera_large.pt") -> str:
        """Download SAM2 checkpoint if it doesn't already exist."""
        model_url = os.path.join(_SAM2_BASE_URL, model_name)
        dest_path = get_skillet_model_cache_dir() / "sam2" / model_name

        if dest_path.exists():
            return dest_path

        (get_skillet_model_cache_dir() / "sam2").mkdir(parents=True, exist_ok=True)

        print(f"[INFO][SAM]Downloading SAM2 checkpoint from {model_url} to {dest_path}.")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB

        with (
            pathlib.Path(dest_path).open("wb") as file,
            tqdm(total=total_size, unit="iB", unit_scale=True, desc=model_name) as progress_bar,
        ):
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))

        print(f"[INFO][SAM] SAM2 checkpoint {model_name} downloaded successfully.")
        return dest_path

    @cache
    def _load_sam_model(self, checkpoint: str):  # noqa: ANN202
        """Load and cache the SAM2 image predictor."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
        print(f"[INFO][SAM] Loading SAM2 with checkpoint={checkpoint}, config={config}, device={self.device}")
        return SAM2ImagePredictor(build_sam2(config, checkpoint, device=self.device))
