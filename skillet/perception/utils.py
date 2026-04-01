from functools import cache
from pathlib import Path

import cv2
import numpy as np


def depth_to_colormap_np(depth_mm: np.ndarray) -> np.ndarray:
    """Convert depth map to a numpy colormap for plotting.

    Args:
        depth_mm: Depth map in millimeters

    Returns:
        Numpy colormap

    """
    valid = depth_mm > 0
    if not valid.any():
        return cv2.applyColorMap(depth_mm.astype("uint8"), cv2.COLORMAP_TURBO)

    depth_valid = depth_mm[valid].astype("float32")
    lo = float(depth_valid.min())
    hi = float(depth_valid.max())
    if hi <= lo:
        hi = lo + 1.0
    depth_norm = ((depth_mm.astype("float32") - lo) / (hi - lo) * 255.0).clip(0, 255).astype("uint8")
    depth_norm[~valid] = 0
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


@cache
def get_skillet_model_cache_dir() -> Path:
    """Return the cache directory within skillet."""
    import skillet

    top_level_dir = Path(skillet.__file__).parent
    cache_dir = top_level_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
