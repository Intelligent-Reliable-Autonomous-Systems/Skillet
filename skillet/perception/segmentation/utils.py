import numpy as np
import o3d
import trimesh
from jaxtyping import Float


def get_o3d_pcd(
    xyz_map: Float[np.ndarray, "*n 3"], rgb_map: Float[np.ndarray, "*n 3"], voxel_size: float | None = None
) -> o3d.geometry.PointCloud:
    """Get open3d point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_map.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(rgb_map.reshape(-1, 3))
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def depth_to_xyz(
    depth: Float[np.ndarray, "h w"],
    k: Float[np.ndarray, "3 3"],
) -> Float[np.ndarray, "h w 3"]:
    """Convert depth map to XYZ point cloud using camera intrinsics.

    Uses the pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    Args:
        depth: Depth map in meters.
        k: Camera intrinsics matrix (3x3).

    Returns:
        XYZ map where each pixel contains (X, Y, Z) coordinates in meters.

    """
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    h, w = depth.shape

    # Create pixel coordinate grids using broadcasting
    u = np.arange(w, dtype=np.float32)  # (w,)
    v = np.arange(h, dtype=np.float32)  # (h,)
    u_grid, v_grid = np.meshgrid(u, v)  # Both (h, w)

    # Compute XYZ using vectorized operations
    z = depth
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy

    # Stack into (h, w, 3) array
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def aabb_to_cuboid(aabb: np.ndarray, name: str) -> trimesh.primitives.Box:
    """Convert AABB to trimesh Box.

    Args:
        aabb: Axis-aligned bounding box as np.ndarray of shape (2, 3)
              where aabb[0] is min point and aabb[1] is max point
        name: Name to associate with the box

    Returns:
        A trimesh.primitives.Box representing the AABB

    """
    # Calculate box dimensions
    extents = aabb[1] - aabb[0]  # [width, depth, height]
    center = aabb.mean(0)  # Center point

    # Create a box centered at origin with the right dimensions
    box = trimesh.primitives.Box(extents=extents)

    # Move box to the correct position
    box.apply_translation(center)

    # Store name as metadata
    box.metadata = {"name": name}

    # Set a default color (light gray)
    box.visual.face_colors = [200, 200, 200, 255]

    return box


def _object_contact_points(xyz_world: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Estimate the 3D contact point of each object with the surface it rests on.

    For each object mask, takes the bottom 10th-percentile z points and returns their centroid.
    These points approximate where the object touches the table.

    Args:
        xyz_world: (H, W, 3) structured point cloud in world frame.
        masks: (N, 1, H, W) segmentation masks from SAM.

    Returns:
        (M, 3) array of contact points, one per object with enough valid points (M <= N).

    """
    masks_2d = masks.squeeze(1).astype(bool)  # (N, H, W)
    contacts = []
    for mask in masks_2d:
        obj_xyz = xyz_world[mask]
        valid = ~np.isnan(obj_xyz).any(axis=1)
        if valid.sum() < 10:
            continue
        obj_xyz = obj_xyz[valid]
        z_thresh = np.percentile(obj_xyz[:, 2], 10)
        bottom_pts = obj_xyz[obj_xyz[:, 2] <= z_thresh]
        contacts.append(bottom_pts.mean(axis=0))
    return np.array(contacts) if contacts else np.empty((0, 3))


def segment_table_with_ransac(
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    masks: np.ndarray,
    valid_mask: np.ndarray = None,
    max_planes: int = 5,
    contact_threshold: float = 0.03,
) -> trimesh.primitives.Box:
    """Segment the table by finding the plane that the most detected objects rest on.

    Runs iterative RANSAC to find candidate planes, then scores each by how many object
    contact points (bottom of each object's point cloud) lie within `contact_threshold`
    of the plane. The winning plane is the table surface.

    Args:
        xyz_world: (H, W, 3) structured point cloud in world frame.
        rgb: (H, W, 3) RGB colors in [0, 1].
        masks: (N, 1, H, W) segmentation masks from SAM for detected objects.
        valid_mask: Optional boolean mask of valid points over the spatial dims.
        max_planes: Maximum number of RANSAC iterations to run.
        contact_threshold: Distance (metres) within which an object contact point
                           is considered to lie on a candidate plane.

    Returns:
        table_box: trimesh.primitives.Box representing the table.

    """
    if len(xyz_world.shape) != 3:
        raise ValueError(f"Expected structured (H, W, 3) point cloud, got shape {xyz_world.shape}")

    # Estimate where each object contacts its supporting surface
    contact_pts = _object_contact_points(xyz_world, masks)
    if len(contact_pts) == 0:
        raise RuntimeError("No object contact points found — ensure objects are detected before calling this function.")
    print(f"[INFO] Object contact points (world frame):\n{contact_pts}")

    # Build point cloud from valid points
    if valid_mask is None:
        valid_mask = ~np.isnan(xyz_world).any(axis=2)
    xyz_valid = xyz_world[valid_mask]
    rgb_valid = rgb[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    pcd.colors = o3d.utility.Vector3dVector(rgb_valid)
    voxel_size = 0.005
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Iterative RANSAC: score each candidate plane by number of objects resting on it
    remaining_pcd = pcd
    best_score = -1
    best_pcd = None

    for i in range(max_planes):
        if len(remaining_pcd.points) < 50:
            break

        plane_model, inlier_idxs = remaining_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        norm = np.linalg.norm([a, b, c])

        # Distance from each contact point to this plane
        dists = np.abs(contact_pts @ np.array([a, b, c]) + d) / norm
        score = int((dists < contact_threshold).sum())

        inlier_pcd = remaining_pcd.select_by_index(inlier_idxs)
        print(f"[DEBUG] Plane {i}: model={plane_model}, objects_on_plane={score}/{len(contact_pts)}")

        if score > best_score:
            best_score = score
            best_pcd = inlier_pcd

        remaining_pcd = remaining_pcd.select_by_index(inlier_idxs, invert=True)

    if best_pcd is None or best_score == 0:
        raise RuntimeError(
            f"No plane found with objects resting on it (tried {max_planes} planes, "
            f"{len(contact_pts)} contact points, threshold={contact_threshold}m)."
        )

    print(f"[INFO] Selected table plane with {best_score}/{len(contact_pts)} objects")
    table_pcd = best_pcd

    # Remove statistical outliers to eliminate distant points that happen to lie on the plane
    table_pcd, _ = table_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # DBSCAN clustering: keep only the largest cluster (the actual table surface).
    # eps is tied to voxel_size so no independent hyperparameter is introduced.
    dbscan_eps = 3 * voxel_size
    labels = np.array(table_pcd.cluster_dbscan(eps=dbscan_eps, min_points=10))
    n_clusters = int(labels.max()) + 1 if len(labels) > 0 and labels.max() >= 0 else 0
    if n_clusters == 0:
        raise RuntimeError("DBSCAN found no clusters in table plane inliers after outlier removal.")
    if n_clusters > 2:
        print(
            f"[WARN] DBSCAN found {n_clusters} clusters in table plane inliers — expected 1–2. "
            f"Keeping largest cluster; point cloud may have significant noise."
        )
    else:
        print(f"[DEBUG] DBSCAN found {n_clusters} cluster(s) in table plane inliers.")
    largest_label = int(np.bincount(labels[labels >= 0]).argmax())
    table_pcd = table_pcd.select_by_index(np.where(labels == largest_label)[0])

    # Get table AABB using percentile-based bounds to handle remaining outliers
    table_pts = np.asarray(table_pcd.points)
    # Use 2nd and 98th percentiles for XY to avoid extreme outliers while keeping most of table
    xy_min = np.percentile(table_pts[:, :2], 2, axis=0)
    xy_max = np.percentile(table_pts[:, :2], 98, axis=0)
    # Use actual min/max for Z since height is well-defined by RANSAC
    z_min = table_pts[:, 2].min()
    z_max = table_pts[:, 2].max()

    table_aabb = np.stack([np.append(xy_min, z_min), np.append(xy_max, z_max)])
    surface_z = table_pts[:, 2].mean()

    # Create table box
    table_box = aabb_to_cuboid(table_aabb, "table")

    # Adjust height position so the top of the box aligns with detected surface
    # We need to adjust the transform directly since trimesh.Box works differently
    extents = table_box.extents
    table_center = table_box.center_mass
    # Offset the box down so its top surface aligns with the detected plane
    height_offset = surface_z - table_center[2] - extents[2] / 2 - 0.02  # small offset
    table_box.apply_translation([0, 0, height_offset])

    # Set color from point cloud
    table_color = (np.asarray(table_pcd.colors).mean(0) * 255).astype(np.uint8)
    table_color_rgba = np.append(table_color, 255)
    table_box.visual.face_colors = table_color_rgba

    print(f"[INFO] Table surface at z = {surface_z:.3f}, dims = {table_box.extents}")
    return table_box
