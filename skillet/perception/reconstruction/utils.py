from typing import Literal

import numpy as np
import open3d as o3d
import torch
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from skillet.core.math import np_convert_quat
from skillet.scene.base import Scene, SceneObject


def assign_objects_to_id_hungarian(
    positions: np.ndarray | torch.Tensor,
    detections: np.ndarray | torch.Tensor,
    max_distance: float = 10.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """Match detections to known objects (cubes).

    Occluded cubes (no confident match) retain their last known position and are flagged in self.occluded.

    Args:
        positions: np.ndarray of shape (N, 3) of current cube positions
        detections: np.ndarray of shape (N, 3) of new cube positions
        ids: np.ndarray of shape (N,) of the ids of the cubes in the scene
        max_distance: the maximum distance for a cube to be considered not visible
        threhsold: the distance threshold a cube needs to have moved from its previous center

    Returns:
        tuple of: np.ndarray of positions and cube ids of those positions

    """
    device = None
    if isinstance(positions, torch.Tensor):
        device = positions.device
        positions = positions.cpu().numpy()
        detections = detections.cpu().numpy()
    # Create cost matrix
    diff = positions[:, None, :] - detections[~np.isnan(detections).any(axis=1)][None, :, :]  # (K, D, 3)
    cost = np.linalg.norm(diff, axis=-1)  # (K, D)
    # Assignment using Hungarian Algorithm
    cube_idx, det_idx = linear_sum_assignment(cost)
    valid_cube_idx = []
    valid_det_idx = []
    for k, d in zip(cube_idx, det_idx, strict=True):
        if cost[k, d] <= max_distance:
            valid_cube_idx.append(k)
            valid_det_idx.append(d)
    if device is None:
        return np.asarray(valid_cube_idx, dtype=np.int32), np.asarray(valid_det_idx, dtype=np.int32)
    return torch.as_tensor(valid_cube_idx, dtype=torch.int32, device=device), torch.as_tensor(
        valid_det_idx, dtype=torch.int32, device=device
    )


def assign_objects_to_id_mean(
    positions: np.ndarray, detections: np.ndarray, ids: np.ndarray | None = None, max_distance: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Match detections to known objects (cubes).

    Occluded cubes (no confident match) retain their last known position and are flagged in self.occluded.

    Args:
        positions: np.ndarray of shape (N, 3) of current cube positions
        detections: np.ndarray of shape (N, 3) of new cube positions
        ids: np.ndarray of shape (N,) of the ids of the cubes in the scene
        max_distance: the maximum distance for a cube to be considered not visible
        threhsold: the distance threshold a cube needs to have moved from its previous center

    Returns:
        tuple of: np.ndarray of positions and cube ids of those positions

    """
    # Create cost matrix
    diff = positions[:, None, :] - detections[~np.isnan(detections).any(axis=1)][None, :, :]  # (K, D, 3)
    cost = np.linalg.norm(diff, axis=-1)  # (K, D)
    candidates = [(i, j, cost[i, j]) for j in range(cost.shape[1]) for i in range(cost.shape[0])]
    candidates.sort(key=lambda x: x[2])

    valid_cube_idx = []
    valid_det_idx = []
    for i, j, c in candidates:
        if i not in valid_cube_idx and j not in valid_det_idx:
            valid_cube_idx.append(i)
            valid_det_idx.append(j)

    return np.asarray(valid_cube_idx, dtype=np.int32), np.asarray(valid_det_idx, dtype=np.int32)


def get_sorted_object_poses(scene: Scene, obj: tuple[SceneObject]) -> np.ndarray:
    """Return a list of scene objects of a specific type sorted by their ID.

    Args:
        scene: The scene to get poses from
        obj: Object instance to grab all instances of

    Returns:
        np.ndarray of shape (N, 7) of the object poses

    """
    id_list = []
    pose_list = []
    for ob in scene.objects:
        if not isinstance(ob, obj):
            continue
        id_list.append(ob.object_id)
        pose_list.append(ob.pose.cpu().numpy())

    sorted_ids = np.argsort(id_list)

    return np.asarray(pose_list)[sorted_ids], np.asarray(id_list)[sorted_ids]


def assign_poses_to_objects(
    scene: Scene,
    obj: SceneObject,
    poses: np.ndarray,
    ids: np.ndarray,
    obj_idx: np.ndarray,
    det_idx: np.ndarray,
    device: str = "cuda",
) -> None:
    """Return a list of scene objects of a specific type sorted by their ID.

    Args:
        scene: The scene to get poses from
        obj: Object instance to grab all instances of
        poses: np.ndarray of poses for SceneObjects
        ids: np.ndarray of sorted object ids
        obj_idx: Sorted indexes of object scene ids according to poses
        det_idx: The detection index of which pose to assign to which object
        device: CUDA device to create tensor on

    """
    for ob in scene.objects:
        if not isinstance(ob, obj):
            continue
        idx = np.where(ob.object_id == ids[obj_idx])[0]
        if idx.size > 0:
            idx = idx[0]
        else:
            # TODO handle occlusion more robustly.
            continue
        if np.isnan(poses[det_idx[idx]]).any():
            continue
        ob.pose = torch.as_tensor(np.concatenate((poses[det_idx[idx]], [1, 0, 0, 0])), device=device)


def percentile_clip(pts: torch.Tensor | np.ndarray, lo: float, hi: float) -> torch.Tensor | np.ndarray:
    """Percentile clip piont clouds.

    Args:
        pts: Points in shape (N,3)
        lo: low percentile
        hi: high percentile

    """
    if isinstance(pts, torch.Tensor):
        lo_vals = torch.quantile(pts, lo / 100.0, dim=0)
        hi_vals = torch.quantile(pts, hi / 100.0, dim=0)
    elif isinstance(pts, np.ndarray):
        lo_vals, hi_vals = np.percentile(pts, [lo / 100.0, hi / 100.0], dim=0)
    mask = ((pts >= lo_vals) & (pts <= hi_vals)).all(dim=1)
    return pts[mask]


def find_obj_centers_mean(
    masks: torch.Tensor,
    depth: torch.Tensor,
    camera_matrix: torch.Tensor,
    camera_pos: torch.Tensor,
    camera_quat: torch.Tensor,
    depth_scale: float = 1.0,
    obj_size: float | np.ndarray = 0.044,
    frame: Literal["world", "camera"] = "camera",
) -> torch.Tensor:
    """Find object centers from segmentation masks and depth map in the camera frame.

    Assumes that each object face is parallel to the camera, meaning we know the plane equation

    Args:
        masks: Binary masks for each object, shape (N, H, W) or list of (H, W) arrays
        depth: Depth map, shape (1, H, W) in meters or scaled units
        camera_matrix: 3x3 camera intrinsics matrix
        camera_pos: (3,) array of camera position in world frame
        camera_quat: (4,) array of quaternion in wxyz of camera orientation in world frame
        depth_scale: Scale factor for depth values (if depth is in mm, use 1/1000)
        obj_size: Expected object size in meters (used for validation)
        frame: frame in which to compute RANSAC in (world or camera)

    Returns:
        Centers: List of 3D object centers in camera frame (N, 3)

    """
    depth = depth.squeeze(0)
    centers = []
    for i, mask in enumerate(masks):
        mask = mask.to(torch.bool)

        if mask.sum() < 10:
            continue

        # Get 3D points from mask and depth
        points_3d = percentile_clip(mask_to_3d_points(mask, depth, camera_matrix, depth_scale), lo=10, hi=90)

        # Transform 3d points into world frame
        if frame == "world":
            points_3d = transform_xyz_to_world(points_3d, camera_pos=camera_pos, camera_quat=camera_quat)

        o_size = obj_size if isinstance(obj_size, float) else obj_size[i]
        proj_u = points_3d[:, 0]
        proj_v = points_3d[:, 1]
        proj_w = points_3d[:, 2]
        centroid_u = torch.sort(proj_u)[0][len(proj_u) // 2]
        centroid_v = torch.sort(proj_v)[0][len(proj_v) // 2]
        mask_v = torch.abs(centroid_v - proj_v) < 0.008
        mask_u = torch.abs(centroid_u - proj_u) < 0.008
        centroid_w = proj_w[mask_u & mask_v].mean() + (o_size * (3 / 4))  # helps with partial occlusion
        centroid = torch.as_tensor([centroid_u, centroid_v, centroid_w], device=masks.device)
        centers.append(centroid)

        # normals = compute_normals(points_3d.cpu().numpy())
        # dirs = pca_normal_directions(normals, n=3)
        # quats.append(torch.as_tensor(roll_quaternion_from_directions(dirs), device=masks.device))

    return torch.stack(centers, dim=0)


def find_spill_centers_mean_bbox(
    masks: torch.Tensor,
    depth: torch.Tensor,
    camera_matrix: torch.Tensor,
    camera_pos: torch.Tensor,
    camera_quat: torch.Tensor,
    depth_scale: float = 1.0,
    frame: Literal["world", "camera"] = "camera",
) -> torch.Tensor:
    """Find object centers from segmentation masks and depth map in the camera frame.

    Assumes that each object face is parallel to the camera, meaning we know the plane equation

    Args:
        masks: Binary masks for each object, shape (N, H, W) or list of (H, W) arrays
        depth: Depth map, shape (1, H, W) in meters or scaled units
        camera_matrix: 3x3 camera intrinsics matrix
        camera_pos: (3,) array of camera position in world frame
        camera_quat: (4,) array of quaternion in wxyz of camera orientation in world frame
        depth_scale: Scale factor for depth values (if depth is in mm, use 1/1000)
        frame: frame in which to compute RANSAC in (world or camera)

    Returns:
        Centers: List of 3D object centers in camera frame (N, 3)

    """
    depth = depth.squeeze(0)
    centers = []
    bboxes = []
    for i, mask in enumerate(masks):
        mask = mask.to(torch.bool)

        if mask.sum() < 10:
            continue

        # Get 3D points from mask and depth
        points_3d = percentile_clip(mask_to_3d_points(mask, depth, camera_matrix, depth_scale), lo=10, hi=90)

        # Transform 3d points into world frame
        if frame == "world":
            points_3d = transform_xyz_to_world(points_3d, camera_pos=camera_pos, camera_quat=camera_quat)

        proj_u = points_3d[:, 0]
        proj_v = points_3d[:, 1]
        proj_w = points_3d[:, 2]
        centroid_u = torch.sort(proj_u)[0][len(proj_u) // 2]
        centroid_v = torch.sort(proj_v)[0][len(proj_v) // 2]
        centroid_w = torch.sort(proj_w)[0][len(proj_w) // 2]
        centroid = torch.as_tensor([centroid_u, centroid_v, centroid_w], device=masks.device)
        centers.append(centroid)
        bboxes.append([points_3d.min(dim=0), points_3d.max(dim=0)])

    return torch.stack(centers, dim=0) if len(centers) != 0 else None, (
        torch.stack(bboxes, dim=0) if len(bboxes) != 0 else None
    )


def compute_normals(points: np.ndarray) -> np.ndarray:
    """Compute the normal vectors of each point using local neighbor clustering."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return np.asarray(pcd.normals)


def pca_normal_directions(normals: np.ndarray, n: int = 3) -> np.ndarray:
    """Compute the primary normals using PCA."""
    unit_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    cov = unit_normals.T @ unit_normals
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n]].T


def normals_to_quaternion_scipy(d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Convert the normal vectors to quaternion in [x,y,z,w]."""
    x = d0 / np.linalg.norm(d0)
    y = d1 - np.dot(d1, x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)

    R = np.column_stack([x, y, z])

    r = Rotation.from_matrix(R)
    return np_convert_quat(r.as_quat(), to="wxyz")


FORWARD_COL = None


def pitch_quaternion_from_directions(dirs: np.ndarray) -> np.ndarray:
    """Extract pitch-only quaternion from direction vectors."""
    # Find the most horizontal column (lowest Z component) = "forward" vector
    global FORWARD_COL
    z_components = np.abs(dirs[2, :])
    if FORWARD_COL is None:
        FORWARD_COL = np.argmin(z_components)
    forward = dirs[:, FORWARD_COL]

    # Project onto XZ plane and normalize (pitch = rotation around Y axis)
    forward_xz = np.array([forward[0], 0.0, forward[2]])
    forward_xz /= np.linalg.norm(forward_xz)

    # Pitch = angle of "forward" vector from world X, in the XZ plane
    pitch = np.arctan2(-forward_xz[2], forward_xz[0])

    # Build quaternion from pitch only (rotation around Y axis)
    q = Rotation.from_euler("y", pitch)
    return np_convert_quat(q.as_quat(), to="wxyz")


def roll_quaternion_from_directions(dirs: np.ndarray) -> np.ndarray:
    """Extract roll-only quaternion from direction vectors."""
    # Find the most vertical column (highest Z component) = "up" vector
    global FORWARD_COL
    z_components = np.abs(dirs[2, :])
    if FORWARD_COL is None:
        FORWARD_COL = np.argmax(z_components)
    up = dirs[:, FORWARD_COL]

    # Project onto YZ plane and normalize (roll = rotation around X axis)
    up_yz = np.array([0.0, up[1], up[2]])
    up_yz /= np.linalg.norm(up_yz)

    # Roll = angle of "up" vector from world Z, in the YZ plane
    roll = np.arctan2(up_yz[1], up_yz[2])

    # Build quaternion from roll only (rotation around X axis)
    q = Rotation.from_euler("x", roll)
    return np_convert_quat(q.as_quat(), to="wxyz")


def yaw_quaternion_from_directions(dirs: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from a rotation matrix."""
    global FORWARD_COL
    z_components = np.abs(dirs[2, :])
    if FORWARD_COL is None:
        FORWARD_COL = np.argmin(z_components)

    forward = dirs[:, FORWARD_COL]

    # Project onto XY plane and normalize
    forward_xy = np.array([forward[0], forward[1], 0.0])
    forward_xy /= np.linalg.norm(forward_xy)

    yaw = np.arctan2(forward_xy[1], forward_xy[0])

    # Build quaternion from yaw only (rotation around Z)
    q = Rotation.from_euler("z", yaw)
    return np_convert_quat(q.as_quat(), to="wxyz")


def find_cube_centers_ransac(
    masks: np.ndarray,
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    camera_pos: np.ndarray,
    camera_quat: np.ndarray,
    depth_scale: float = 1.0,
    cube_size: float = 0.044,
    frame: Literal["world", "camera"] = "camera",
) -> dict[str, np.ndarray]:
    """Find cube centers from segmentation masks and depth map in the camera frame.

    Args:
        masks: Binary masks for each cube, shape (N, H, W) or list of (H, W) arrays
        depth: Depth map, shape (1, H, W) in meters or scaled units
        camera_matrix: 3x3 camera intrinsics matrix
        camera_pos: (3,) array of camera position in world frame
        camera_quat: (4,) array of quaternion in wxyz of camera orientation in world frame
        depth_scale: Scale factor for depth values (if depth is in mm, use 1/1000)
        cube_size: Expected cube size in meters (used for validation)
        frame: frame in which to compute RANSAC in (world or camera)

    Returns:
        Centers: List of 3D cube centers in camera frame (N, 3)

    """
    depth = depth.squeeze(0)

    centers = []
    for mask in masks:
        mask = mask.astype(bool)

        if np.sum(mask) < 10:  # Skip if too few pixels
            continue

        # Get 3D points from mask and depth
        points_3d = mask_to_3d_points(mask, depth, camera_matrix, depth_scale)

        # Transform 3d points into world frame
        if frame == "world":
            points_3d = transform_xyz_to_world(points_3d, camera_pos=camera_pos, camera_quat=camera_quat)

        # Fit plane to visible cube face with RANSAC
        normal, plane_eq, _ = fit_plane_to_points(points_3d, frame=frame)

        # Project points onto the fitted plane
        points_on_plane = project_points_to_plane(points_3d, normal, plane_eq)

        # Find 2D center of the points on the plane
        center_on_plane = np.mean(points_on_plane, axis=0)

        # Move back along the normal to find cube center
        # Assume cube center is at distance cube_size/2 from the visible face
        cube_center = center_on_plane + normal * (cube_size / 2)

        centers.append(cube_center)

    return np.asarray(centers)


def mask_to_3d_points(
    mask: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    camera_matrix: np.ndarray | torch.Tensor,
    depth_scale: float = 1.0,
) -> np.ndarray | torch.Tensor:
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
    v, u = np.where(mask) if isinstance(mask, np.ndarray) else torch.where(mask)

    # Get depth values at masked pixels
    z = depth[v, u] * depth_scale

    # Backproject to 3D using camera intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack([x, y, z]) if isinstance(mask, np.ndarray) else torch.column_stack([x, y, z])


def fit_plane_to_points(
    points: np.ndarray,
    max_iterations: int = 500,
    inlier_threshold: float = 0.01,
    frame: Literal["world", "camera"] = "world",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a plane to 3D points using RANSAC + least squares.

    Args:
        points: 3D points, shape (N, 3)
        max_iterations: Maximum RANSAC iterations
        inlier_threshold: Distance threshold for inliers (in meters)
        frame: Frame to compute RANSAC in

    Returns:
        Tuple of:
            - normal: Unit normal vector (3,)
            - plane_eq: [a, b, c, d] where ax + by + cz + d = 0
            - mean_distance: Mean distance of points to plane

    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    best_inliers = None
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
    if frame == "camera" and normal[2] < 0:
        normal = -normal

    # normal[2] = 0  # zero out z component
    # normal = normal / np.linalg.norm(normal)
    # Plane equation: ax + by + cz + d = 0
    a, b, c = normal
    d = -np.dot(normal, mean)

    # Mean distance to plane
    distances = np.abs(np.dot(points - mean, normal))
    mean_distance = np.mean(distances)

    return normal, np.array([a, b, c, d]), mean_distance


def project_points_to_plane(
    points: np.ndarray | torch.Tensor,
    normal: np.ndarray | torch.Tensor,
    plane_eq: np.ndarray | torch.Tensor,
    threshold: float = 0.004,
) -> np.ndarray | torch.Tensor:
    """Project 3D points onto a plane and filter out points on it that exceed a maximum distance.

    Args:
        points: 3D points, shape (N, 3)
        normal: Unit normal vector (3,)
        plane_eq: [a, b, c, d] for plane ax + by + cz + d = 0
        threshold: distance threshold for a point not being on the plane

    Returns:
        Projected points on the plane, shape (N, 3)

    """
    # Distance from each point to plane
    a, b, c, d = plane_eq
    distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

    # Project along normal
    distances = distances.reshape(-1, 1)
    mask = (distances < threshold).flatten()
    return points[mask] - distances[mask] * normal.reshape(1, 3)


def transform_xyz_to_world(
    centers: np.ndarray | torch.Tensor, camera_pos: np.ndarray | torch.Tensor, camera_quat: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Transform points (xyz) from the camera frame to the world frame.

    Args:
        centers: np.ndarray 3D points (N, x, y, z) in the camera frame.
        camera_pos:      Camera position in the world frame, shape (3,).
        camera_quat:     Camera orientation as quaternion (w, x, y, z), shape (4,).

    Returns:
        np.ndarray of shape (N,3) of 3D points (x, y, z) in the world frame.

    """
    R = quaternion_to_rotation_matrix(camera_quat)

    if isinstance(centers, np.ndarray):
        xyz_world = np.zeros_like(centers)
        for i in range(centers.shape[0]):
            xyz_world[i] = R @ np.asarray(centers[i]) + camera_pos
    else:
        xyz_world = torch.zeros_like(centers, device=centers.device)
        for i in range(centers.shape[0]):
            xyz_world[i] = R @ torch.as_tensor(centers[i]) + camera_pos

    return xyz_world


def transform_plane_to_world(
    plane_eq: np.ndarray | torch.Tensor, camera_pos: np.ndarray | torch.Tensor, camera_quat: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Transform a plane equation from camera frame to world frame.

    Args:
        plane_eq: (4,) array [a, b, c, d] where ax + by + cz + d = 0 in camera frame
        camera_pos:      Camera position in the world frame, shape (3,).
        camera_quat:     Camera orientation as quaternion (w, x, y, z), shape (4,).

    Returns:
        (4,) array [a', b', c', d'] in world frame

    """
    n_cam = plane_eq[:3]  # normal in camera frame
    d_cam = plane_eq[3]
    R = quaternion_to_rotation_matrix(camera_quat)
    # Rotate the normal: n_world = R @ n_cam
    n_world = R @ n_cam

    # The plane passes through any point p_cam that satisfies the equation.
    # A convenient point is the closest point to the origin: p_cam = -d * n_cam
    p_cam = -d_cam * n_cam

    # Transform that point to world frame
    p_world = R @ p_cam + camera_pos

    # Recompute d in world frame: d' = -n_world . p_world
    d_world = -n_world @ p_world

    return (
        np.asarray([n_world[0], n_world[1], n_world[2], d_world])
        if isinstance(camera_pos, np.ndarray)
        else torch.as_tensor([n_world[0], n_world[1], n_world[2], d_world], device=camera_pos.device)
    )


def quaternion_to_rotation_matrix(wxyz: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    if isinstance(wxyz, np.ndarray):
        w, x, y, z = wxyz / np.linalg.norm(wxyz)
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
    w, x, y, z = wxyz / torch.linalg.norm(wxyz)
    return torch.as_tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        device=wxyz.device,
    )
