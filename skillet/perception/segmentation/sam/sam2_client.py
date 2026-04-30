"""SAM2 segmentation — local predictor and remote HTTP client."""

import os
from pathlib import Path

import numpy as np
import requests
import torch
from jaxtyping import Float, Int, UInt8
from PIL import Image
from tqdm import tqdm
from typing_extensions import override

from skillet.perception.segmentation.sam.sam_base import SAMClient
from skillet.perception.utils import get_skillet_model_cache_dir

_SAM2_BASE_URL = Path("https://dl.fbaipublicfiles.com/segment_anything_2/092824/")


class SAM2Client(SAMClient):
    """Main client class for the SAM3 model."""

    def __init__(
        self,
        model_name: str = "sam2.1_hiera_large.pt",
        device: str = "cuda",
        use_server: bool = True,
        load_server: bool = False,
    ) -> None:
        """Initialize the SAM2 client.

        Args:
            model_name: Name of the SAM2 model checkpoint
            device: Device to load the model on

        """
        model_path = self._download_sam_checkpoint(model_name)
        self.model_name = "sam2"
        super().__init__(model_path, device, use_server)
        if (load_server and use_server) or not use_server:
            self.sam_model = self._load_sam_model(checkpoint=model_path)

    @override
    def segment_from_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Float[torch.Tensor | np.ndarray, "n 4"] | None = None,
    ) -> tuple[Float[torch.Tensor, "n 1 h w"], Float[torch.Tensor, " n"]]:
        # bboxes are already in SAM2 format [x0, y0, x1, y1] (pixels)
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        if isinstance(rgb, np.ndarray) and rgb.shape[0] == 3:
            rgb = rgb.transpose((1, 2, 0))
        self.sam_model.set_image(rgb)
        masks, scores, _ = self.sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=bboxes,
            multimask_output=False,
        )

        masks_t = torch.as_tensor(masks, device=self.device)
        if masks.ndim == 4:
            masks_t = masks_t.squeeze()
        scores_t = torch.as_tensor(scores, device=self.device)
        return masks_t, scores_t

    @override
    def segment_from_concepts(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        concepts: list[str],
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"], Int[torch.Tensor, "n 4"], Float[torch.Tensor, " n"], Int[torch.Tensor, " n"]
    ]:
        raise NotImplementedError("SAM2 does not support segmenting from concepts.")

    def _download_sam_checkpoint(self, model_name: str = "sam2.1_hiera_large.pt") -> str:
        """Download SAM2 checkpoint if it doesn't already exist."""
        model_url = _SAM2_BASE_URL / model_name
        dest_path = get_skillet_model_cache_dir() / model_name

        if dest_path.exists():
            return dest_path

        (get_skillet_model_cache_dir()).mkdir(parents=True, exist_ok=True)

        print(f"[INFO][SAM]Downloading SAM2 checkpoint from {model_url} to {dest_path}.")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB

        with (
            Path(dest_path).open("wb") as file,
            tqdm(total=total_size, unit="iB", unit_scale=True, desc=model_name) as progress_bar,
        ):
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))

        print(f"[INFO][SAM] SAM2 checkpoint {model_name} downloaded successfully.")
        return dest_path

    def _load_sam_model(self, checkpoint: str):  # noqa: ANN202
        """Load and cache the SAM2 image predictor."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
        print(f"[INFO][SAM] Loading SAM2 with checkpoint={checkpoint}, config={config}, device={self.device}")
        return SAM2ImagePredictor(build_sam2(config, checkpoint, device=self.device))
