from typing import Any

import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def assign_objects_to_id(
    positions: np.ndarray, detections: np.ndarray, ids: np.ndarray | None = None, max_distance: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """Match detections to known objects (cubes).

    Occluded cubes (no confident match) retain their last known position and are flagged in self.occluded.

    Args:
        positions: np.ndarray of shape (N, 3) of current cube positions
        detections: np.ndarray of shape (N, 3) of new cube positions
        ids: np.ndarray of shape (N,) of the ids of the cubes in the scene
        max_distance: the maximum distance for a cube to be considered not visible

    Returns:
        tuple of: np.ndarray of positions and cube ids of those positions

    """
    # Create cost matrix
    diff = positions[:, None, :] - detections[None, :, :]  # (K, D, 3)
    cost = np.linalg.norm(diff, axis=-1)  # (K, D)

    # Assignment using Hungarian Algorithm
    cube_idx, det_idx = linear_sum_assignment(cost)
    valid_cube_idx = []
    valid_det_idx = []
    for k, d in zip(cube_idx, det_idx, strict=True):
        if cost[k, d] <= max_distance:
            valid_cube_idx.append(k)
            valid_det_idx.append(d)

    return np.asarray(valid_cube_idx), np.asarray(valid_det_idx)


def find_cube_centers(
    masks: np.ndarray,
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale: float = 1.0,
    cube_size: float = 0.05,
) -> dict[str, Any]:
    """Find cube centers from segmentation masks and depth map in the camera frame.

    Args:
        masks: Binary masks for each cube, shape (N, H, W) or list of (H, W) arrays
        depth: Depth map, shape (1, H, W) in meters or scaled units
        camera_matrix: 3x3 camera intrinsics matrix
        depth_scale: Scale factor for depth values (if depth is in mm, use 1/1000)
        cube_size: Expected cube size in meters (used for validation)

    Returns:
        Dictionary containing:
            - centers: List of 3D cube centers in camera frame (N, 3)
            - normals: List of face normals (N, 3)
            - plane_equations: List of (a, b, c, d) for plane ax + by + cz + d = 0
            - valid: List of booleans indicating which detections are valid
            - details: List of dictionaries with per-cube debug info

    """
    depth = depth.squeeze(0)

    results = {
        "centers": [],
        "normals": [],
        "plane_equations": [],
        "valid": [],
        "details": [],
    }

    for i, mask in enumerate(masks):
        mask = mask.astype(bool)

        if np.sum(mask) < 10:  # Skip if too few pixels
            results["valid"].append(False)
            results["centers"].append(None)
            results["normals"].append(None)
            results["plane_equations"].append(None)
            results["details"].append({"error": "Insufficient masked pixels"})
            continue

        try:
            # Get 3D points from mask and depth
            points_3d = mask_to_3d_points(mask, depth, camera_matrix, depth_scale)

            # Fit plane to visible cube face
            normal, plane_eq, plane_distance = fit_plane_to_points(points_3d)

            # Project points onto the fitted plane
            points_on_plane = project_points_to_plane(points_3d, normal, plane_eq)

            # Find 2D center of the points on the plane
            center_on_plane = np.mean(points_on_plane, axis=0)

            # Move back along the normal to find cube center
            # Assume cube center is at distance cube_size/2 from the visible face
            cube_center = center_on_plane + normal * (cube_size / 2)

            # Validation checks
            is_valid = validate_cube_detection(points_3d, normal, center_on_plane, cube_size)

            results["centers"].append(cube_center)
            results["normals"].append(normal)
            results["plane_equations"].append(plane_eq)
            results["valid"].append(is_valid)
            results["details"].append(
                {
                    "num_pixels": np.sum(mask),
                    "plane_distance": plane_distance,
                    "center_on_face": center_on_plane.tolist(),
                    "reprojection_error": np.mean([np.abs(np.dot(normal, p) + plane_eq[3]) for p in points_on_plane]),
                }
            )

        except Exception as e:
            results["valid"].append(False)
            results["centers"].append(None)
            results["normals"].append(None)
            results["plane_equations"].append(None)
            results["details"].append({"error": str(e)})

    return results


def mask_to_3d_points(
    mask: np.ndarray,
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Convert 2D mask and depth map to 3D points in camera frame.

    Args:
        mask: Binary mask, shape (H, W)
        depth: Depth map, shape (H, W)
        camera_matrix: 3x3 camera intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        depth_scale: Scale factor for depth

    Returns:
        3D points, shape (N, 3) in camera coordinate frame

    """
    # Get pixel coordinates where mask is True
    v, u = np.where(mask)

    # Get depth values at masked pixels
    z = depth[v, u] * depth_scale

    # Backproject to 3D using camera intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack([x, y, z])


def fit_plane_to_points(
    points: np.ndarray,
    max_iterations: int = 1000,
    inlier_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a plane to 3D points using RANSAC + least squares.

    Args:
        points: 3D points, shape (N, 3)
        max_iterations: Maximum RANSAC iterations
        inlier_threshold: Distance threshold for inliers (in meters)

    Returns:
        Tuple of:
            - normal: Unit normal vector (3,)
            - plane_eq: [a, b, c, d] where ax + by + cz + d = 0
            - mean_distance: Mean distance of points to plane

    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    best_inliers = None
    best_normal = None
    max_inlier_count = 0

    # RANSAC loop
    for _ in range(max_iterations):
        # Sample 3 random points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Compute plane from 3 points
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        norm = np.linalg.norm(normal)
        if norm < 1e-6:  # Points are collinear
            continue

        normal = normal / norm

        # Count inliers
        distances = np.abs(np.dot(points - p1, normal))
        inliers = distances < inlier_threshold
        inlier_count = np.sum(inliers)

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_inliers = inliers

    # Refine with all inliers using least squares
    if best_inliers is None or max_inlier_count < 3:
        mean = np.mean(points, axis=0)
        centered = points - mean
        _, _, vh = svd(centered)
        inlier_points = points
    else:
        inlier_points = points[best_inliers]

    # Least squares refinement
    mean = np.mean(inlier_points, axis=0)
    centered = inlier_points - mean
    _, _, vh = svd(centered)
    normal = vh[-1]

    # Ensure normal points towards positive z (away from camera)
    if normal[2] < 0:
        normal = -normal

    # Plane equation: ax + by + cz + d = 0
    a, b, c = normal
    d = -np.dot(normal, mean)

    # Mean distance to plane
    distances = np.abs(np.dot(points - mean, normal))
    mean_distance = np.mean(distances)

    return normal, np.array([a, b, c, d]), mean_distance


def project_points_to_plane(
    points: np.ndarray,
    normal: np.ndarray,
    plane_eq: np.ndarray,
) -> np.ndarray:
    """Project 3D points onto a plane.

    Args:
        points: 3D points, shape (N, 3)
        normal: Unit normal vector (3,)
        plane_eq: [a, b, c, d] for plane ax + by + cz + d = 0

    Returns:
        Projected points on the plane, shape (N, 3)

    """
    # Distance from each point to plane
    a, b, c, d = plane_eq
    distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

    # Project along normal
    distances = distances.reshape(-1, 1)
    return points - distances * normal.reshape(1, 3)


def validate_cube_detection(
    points_3d: np.ndarray,
    normal: np.ndarray,
    center_on_plane: np.ndarray,
    expected_cube_size: float,
) -> bool:
    """Validate if the detected plane looks like a cube face.

    Args:
        points_3d: 3D points on the detected plane
        normal: Plane normal
        center_on_plane: 2D center on the plane
        expected_cube_size: Expected cube edge length

    Returns:
        Boolean indicating if detection is valid

    """
    # Check 1: Sufficient number of points
    if len(points_3d) < 20:
        return False

    # Check 2: Points should be roughly planar
    distances = np.abs(np.dot(points_3d - center_on_plane, normal))
    mean_distance = np.mean(distances)
    if mean_distance > expected_cube_size * 0.5:  # Too far from plane
        return False

    # Check 3: Spatial extent should be roughly square-like
    # Project to 2D coordinates on the plane
    points_on_plane = points_3d - np.dot(points_3d - center_on_plane, normal).reshape(-1, 1) * normal

    # Create 2D coordinate frame on the plane
    u_axis = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u_axis, normal)) > 0.9:
        u_axis = np.array([0.0, 1.0, 0.0])
    u_axis = u_axis - np.dot(u_axis, normal) * normal
    u_axis = u_axis / np.linalg.norm(u_axis)

    v_axis = np.cross(normal, u_axis)

    # Project points to 2D
    points_2d = np.column_stack(
        [
            np.dot(points_on_plane - center_on_plane, u_axis),
            np.dot(points_on_plane - center_on_plane, v_axis),
        ]
    )

    # Check aspect ratio and size
    extent_u = np.max(np.abs(points_2d[:, 0]))
    extent_v = np.max(np.abs(points_2d[:, 1]))

    aspect_ratio = max(extent_u, extent_v) / (min(extent_u, extent_v) + 1e-6)
    if aspect_ratio > 2.0:  # Too elongated
        return False

    # Check if size is reasonable
    avg_extent = (extent_u + extent_v) / 2
    return not (avg_extent < expected_cube_size * 0.2 or avg_extent > expected_cube_size * 3)


def transform_cube_centers_to_world(
    cube_centers_camera: list[np.ndarray], camera_pos: np.ndarray, camera_quat: np.ndarray
) -> np.ndarray:
    """Transform cube centers from the camera frame to the world frame.

    Args:
        cube_centers_camera: List of 3D points (x, y, z) in the camera frame.
        camera_pos:      Camera position in the world frame, shape (3,).
        camera_quat:     Camera orientation as quaternion (w, x, y, z), shape (4,).

    Returns:
        np.ndarray of shape (N,3) of 3D points (x, y, z) in the world frame.

    """
    R = quaternion_to_rotation_matrix(camera_quat)

    cube_centers_world = []
    for p_cam in cube_centers_camera:
        p_world = R @ np.asarray(p_cam) + camera_pos
        cube_centers_world.append(p_world)

    return np.asarray(cube_centers_world)


def quaternion_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = wxyz / np.linalg.norm(wxyz)  # normalise for safety
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def filter_cube_centers(points: list[np.ndarray], max_cubes: int, distance_threshold: float = 0.05) -> list[np.ndarray]:
    """Filter a list of candidate cube centers down to at most `max_cubes`.

    Filters unique cubes by merging points that are within `distance_threshold`
    of each other.

    Strategy: greedy furthest-first selection on cluster means.

    Args:
        points:             Candidate cube centers in the world frame.
        max_cubes:          Maximum number of cubes in the scene.
        distance_threshold: Points closer than this (metres) are merged
                            into a single cube center.

    Returns:
        List of up to `max_cubes` representative cube centers.

    """
    if not points:
        return []

    pts = np.array(points)  # (N, 3)

    # --- 1. Merge nearby points into clusters ---
    assigned = np.full(len(pts), -1, dtype=int)
    cluster_id = 0

    for i in range(len(pts)):
        if assigned[i] != -1:
            continue
        # Find all unassigned points within threshold of pts[i]
        dists = np.linalg.norm(pts - pts[i], axis=1)
        mask = (dists < distance_threshold) & (assigned == -1)
        assigned[mask] = cluster_id
        cluster_id += 1

    # --- 2. Compute each cluster's mean position ---
    cluster_means = np.array([pts[assigned == cid].mean(axis=0) for cid in range(cluster_id)])  # (C, 3)

    # --- 3. If we already have <= max_cubes clusters, return them all ---
    if len(cluster_means) <= max_cubes:
        return [cluster_means[i] for i in range(len(cluster_means))]

    # --- 4. Otherwise pick `max_cubes` via furthest-point sampling ---
    # Start with the cluster closest to the centroid of all means
    centroid = cluster_means.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(cluster_means - centroid, axis=1)))]

    while len(selected) < max_cubes:
        dists_to_selected = cdist(cluster_means, cluster_means[selected]).min(axis=1)
        dists_to_selected[selected] = -1  # exclude already selected
        selected.append(int(np.argmax(dists_to_selected)))

    return [cluster_means[i] for i in selected]
