"""Open3D visualization for point clouds from object localization."""

from __future__ import annotations

import numpy as np
import torch  # noqa: TC002 - used at runtime for tensor ops

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]


def point_cloud_to_open3d(
    points: torch.Tensor,
    segment_indices: torch.Tensor | None = None,
    filter_zero: bool = True,
) -> object | None:
    """Convert output of segmented_rgbd_to_point_cloud to an Open3D PointCloud.

    Args:
        points: (N, 3), (H, W, 3), (N, 6), or (H, W, 6) from
            segmented_rgbd_to_point_cloud. Last dim is XYZ or XYZRGB.
        segment_indices: Optional (N,) segment indices. If provided, points are colored
            by segment id, overriding any embedded RGB colors.
        filter_zero: If True, remove points at (0, 0, 0) (invalid/masked). Default True.

    Returns:
        Open3D PointCloud, or None if open3d is not installed.

    """
    if o3d is None:
        return None

    x = points.detach().float()
    seg = None
    if segment_indices is not None:
        seg = segment_indices.detach().cpu().numpy().astype(np.int64)

    if x.dim() == 3:
        # (H, W, C) -> (H*W, C)
        _, _, c = x.shape
        x = x.reshape(-1, c)
    x = x.cpu().numpy()

    if seg is not None and seg.shape[0] != x.shape[0]:
        raise ValueError(f"segment_indices shape {seg.shape} does not match points shape {x.shape}")

    xyz = x[:, :3].astype(np.float64)
    valid_mask = None
    if filter_zero:
        valid_mask = np.any(xyz != 0, axis=1)
        xyz = xyz[valid_mask]
        if seg is not None:
            seg = seg[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # If segment indices are provided, color by segment id.
    if seg is not None:
        colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
        unique_seg = np.unique(seg)
        for sid in unique_seg:
            mask = seg == sid
            # Simple deterministic color mapping from segment id
            r = ((sid * 37) % 255) / 255.0
            g = ((sid * 57) % 255) / 255.0
            b = ((sid * 97) % 255) / 255.0
            colors[mask] = (r, g, b)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    # Otherwise, if embedded RGB is present, use it.
    elif x.shape[-1] >= 6:
        rgb = x[:, 3:6]
        if filter_zero and valid_mask is not None:
            rgb = rgb[valid_mask]
        # Open3D expects [0, 1]
        if rgb.size > 0 and rgb.max() > 1.0:
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    return pcd


def visualize_point_cloud(
    points: torch.Tensor,
    segment_indices: torch.Tensor | None = None,
    filter_zero: bool = True,
    window_name: str = "Point cloud",
) -> None:
    """Show the output of segmented_rgbd_to_point_cloud in an Open3D window.

    Args:
        points: (N_points, 3) from segmented_rgbd_to_point_cloud.
        segment_indices: Optional(N_points,) from segmented_rgbd_to_point_cloud.
            If given, the points are colored by the segment indices.
        filter_zero: If True, hide invalid (0,0,0) points. Default True.
        window_name: Title of the Open3D window.

    """
    if o3d is None:
        raise ImportError("Open3D is required for visualization. Install with: pip install open3d")

    pcd = point_cloud_to_open3d(points, segment_indices=segment_indices, filter_zero=filter_zero)
    if pcd is None:
        raise RuntimeError("Failed to create Open3D point cloud")

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [pcd, coord],
        window_name=window_name,
        width=1024,
        height=768,
    )
