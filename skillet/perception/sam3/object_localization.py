"""Object localization utilities: RGB-D to point cloud conversion."""

import torch

from skillet.core.math import transform_points, unproject_depth


def segmented_rgbd_to_point_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    mask: torch.Tensor,
    flatten: bool = True,
    use_perspective: bool = True,
) -> torch.Tensor:
    """Convert a segmented depth image to a packed point cloud representation.

    Args:
        depth: The depth image. Shape is (B, H, W).
        intrinsics: The camera intrinsics. Shape is (B, 3, 3).
        camera_pose: The camera pose (x, y, z, qw, qx, qy, qz) relative to the world frame. Shape is (B, 7).
        mask: The segmentation mask. Shape is (B, H, W).
        flatten: Whether to flatten the point cloud. If True, the point cloud is flattened to (B, N, 3).
            If False, the point cloud is kept in the shape (B, H, W, 3).
        use_perspective: Whether to use perspective depth. If True, the depth is considered as perspective depth.
            If False, the depth is considered as orthogonal depth.

    Returns:
        The packed point cloud. Shape is (B, H*W, 3) or (B, H, W, 3).
            Each point is represented by (x, y, z) in world coordinates.

    """
    b = depth.shape[0]
    # Only unproject pixels that are in the segment and have valid depth
    valid = (mask > 0) & (depth > 0)
    depth_masked = depth * valid.to(depth.dtype)

    depth_reshaped = depth_masked.reshape(b, -1)

    # Unproject to camera frame: (B, H*W, 3)
    points_cam = unproject_depth(depth_reshaped, intrinsics, is_ortho=not use_perspective)

    # Camera pose: (x, y, z, qw, qx, qy, qz) -> pos (B, 3), quat (B, 4) in (w, x, y, z)
    pos = camera_pose[:, :3]
    quat = camera_pose[:, 3:7]

    # Transform camera frame -> world frame
    points_world = transform_points(points_cam, pos, quat)

    # Zero out invalid points (unmasked or zero depth)
    valid_flat = valid.reshape(b, -1, 1)
    return points_world * valid_flat.to(points_world.dtype)
