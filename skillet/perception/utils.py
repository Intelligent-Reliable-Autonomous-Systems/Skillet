"""Perception utilities."""

from functools import cache
from pathlib import Path

import numpy as np


@cache
def get_skillet_model_cache_dir() -> Path:
    """Return the cache directory within skillet."""
    import skillet

    top_level_dir = Path(skillet.__file__).parent
    cache_dir = top_level_dir / "data" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def invalidate_occlusions(depth: np.ndarray, fx: float, baseline: float, max_px: int = 40) -> np.ndarray:
    """Zero depth-aligned-to-color pixels that the depth camera could not see.

    The color camera sits beside the depth camera, so it sees a strip beside each object that the depth camera
    cannot. Alignment (e.g. RealSense `rs.align`) fills that strip with whatever the depth camera saw past the
    object's edge, painting background depth onto the object. A pixel at depth z_b lies in the parallax shadow
    of a pixel k columns toward the depth camera with depth z_f when fx * |baseline| * (1/z_f - 1/z_b) >= k.

    Args:
        depth: (H, W) depth aligned to the color image, in meters (0 = invalid)
        fx: color camera focal length in pixels
        baseline: x position of the color camera in the depth camera's frame, in meters (about +0.015 for a
            RealSense D435, whose color camera is to the right of the depth camera)
        max_px: widest shadow to search for

    Returns:
        A copy of `depth` with shadowed pixels set to 0

    """
    shift = 1 if baseline > 0 else -1
    fb = fx * abs(baseline)
    inverse = np.where(depth > 0, 1.0 / np.maximum(depth, 1e-6), 0.0)
    occluded = np.zeros(depth.shape, dtype=bool)
    for k in range(1, max_px + 1):
        nearer = np.zeros_like(inverse)
        if shift > 0:
            nearer[:, k:] = inverse[:, :-k]
        else:
            nearer[:, :-k] = inverse[:, k:]
        occluded |= (depth > 0) & (nearer > 0) & (fb * (nearer - inverse) >= k)
    out = depth.copy()
    out[occluded] = 0
    return out
