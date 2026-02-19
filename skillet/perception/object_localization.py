"""Object localization utilities: RGB-D to point cloud conversion."""

import torch
from jaxtyping import Int

from skillet.core.math import transform_points, unproject_depth


def segmented_rgbd_to_point_cloud(
    depth: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    use_perspective: bool = True,
    rgb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Int[torch.Tensor, "..."]]:
    """Convert segmented RGB-D images to packed point clouds with segment indices.

    Args:
        depth: The depth image. Shape is (1, H, W). Either uint16 in mm or float in meters.
        masks: The segmentation masks. Shape is (M, H, W). M is the number of segments.
        intrinsics: The camera intrinsics. Shape is (3, 3).
        camera_pose: The camera pose (x, y, z, qw, qx, qy, qz) relative to the world frame.
            Shape is (7,).
        use_perspective: Whether to use perspective depth. If True, the depth is considered as perspective depth.
            If False, the depth is considered as orthogonal depth.
        rgb: The RGB image. Shape is (3, H, W). If None, only XYZ coordinates are returned.

    Returns:
        A tuple of:
        - points: Point cloud with shape (N_points, 3) or (N_points, 6) if rgb is provided.
            Contains only valid points (where mask > 0 and depth > 0).
        - segment_indices: Segment indices with shape (N_points,). Each value is in [0, M-1]
            indicating which mask the point belongs to.

    """
    if depth.dim() != 3 or depth.shape[0] != 1:
        raise ValueError(f"Expected depth shape (1, H, W); got {tuple(depth.shape)}")
    if masks.dim() != 3:
        raise ValueError(f"Expected masks shape (M, H, W); got {tuple(masks.shape)}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3, 3); got {tuple(intrinsics.shape)}")
    if camera_pose.shape != (7,):
        raise ValueError(f"Expected camera_pose shape (7,); got {tuple(camera_pose.shape)}")
    if rgb is not None and (rgb.dim() != 3 or rgb.shape[0] != 3):
        raise ValueError(f"Expected rgb shape (3, H, W); got {tuple(rgb.shape)}")

    # Convert depth to float meters (keep (1, H, W) shape)
    depth = depth.float() / 1000.0 if depth.dtype == torch.uint16 else depth.float()
    depth_hw = depth[0]
    m = int(masks.shape[0])

    # Unproject all pixels once (we'll filter by mask later)
    intrinsics_expanded = intrinsics.unsqueeze(0)  # (1, 3, 3)
    points_cam = unproject_depth(depth, intrinsics_expanded, is_ortho=not use_perspective)  # (1, H*W, 3)
    points_cam = points_cam.squeeze(0)  # (H*W, 3)

    # Transform to world frame
    pos = camera_pose[:3]
    quat = camera_pose[3:7]
    points_world = transform_points(points_cam, pos, quat)  # (H*W, 3)

    # Add RGB if provided
    if rgb is not None:
        # Match unproject_depth flattening order:
        # depth[0] (H, W) is internally flattened as depth[0].transpose(0, 1).reshape(-1) == (W*H,)
        # So RGB (3, H, W) should flatten as (W, H, 3) -> (W*H, 3).
        rgb_flat = rgb.permute(2, 1, 0).reshape(-1, 3)
        points_world = torch.cat([points_world, rgb_flat], dim=-1)  # (H*W, 6)

    # Collect valid points from each mask
    all_points = []
    all_indices = []

    for mask_idx in range(m):
        mask = masks[mask_idx]  # (H, W)
        valid = (mask > 0) & (depth_hw > 0)  # (H, W)

        if valid.any():
            # Match flattening order: transpose(0,1) then reshape
            valid_flat = valid.transpose(0, 1).reshape(-1)  # (H, W) -> (W, H) -> (W*H,)
            seg_points = points_world[valid_flat]  # (N_valid, 3) or (N_valid, 6)
            seg_indices = torch.full((seg_points.shape[0],), mask_idx, dtype=torch.int64, device=points_world.device)

            all_points.append(seg_points)
            all_indices.append(seg_indices)

    if len(all_points) == 0:
        # No valid points
        out_dim = 6 if rgb is not None else 3
        return (
            torch.empty((0, out_dim), dtype=points_world.dtype, device=points_world.device),
            torch.empty((0,), dtype=torch.int64, device=points_world.device),
        )

    # Concatenate all points and indices
    points = torch.cat(all_points, dim=0)  # (N_total, 3) or (N_total, 6)
    segment_indices = torch.cat(all_indices, dim=0)  # (N_total,)

    return points, segment_indices
