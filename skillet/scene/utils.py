import cv2
import numpy as np
import open3d as o3d
import torch
from jaxtyping import Int

from skillet.core.math import transform_points, unproject_depth
from skillet.scene.base import SceneObject, Scene
from skillet.scene.cube import Cube

_PALETTE_BGR: list[tuple[int, int, int]] = [
    (44, 44, 220),
    (44, 190, 44),
    (220, 110, 44),
    (0, 190, 240),
    (200, 44, 200),
    (210, 210, 44),
    (0, 130, 255),
    (170, 44, 240),
    (44, 240, 160),
    (240, 160, 44),
]
_OVERLAY_ALPHA = 0.35
_BBOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1


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


def get_object_geometry(obj: SceneObject) -> list[object]:
    """Get the geometry of an object in the scene.

    Defaults to an AABB wireframe. Unknown poses are ignored.
    """
    # TODO: add object-specific geometry visualization.
    if not obj.is_pose_known():
        return []
    if isinstance(obj, Cube):
        corners = obj.get_corners().cpu().numpy()
        return [create_box_lineset(corners)]
    aabb = obj.aabb.cpu().numpy()
    return [create_aabb_lineset(aabb)]


def quat_to_roll_pitch_yaw(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert a quaternion to roll, pitch, yaw.

    Args:
        quat: The quaternion in (w, x, y, z).

    Returns:
        A tuple containing roll, pitch, yaw.

    """
    roll = np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2]))
    pitch = np.arcsin(2 * (quat[0] * quat[2] - quat[3] * quat[1]))
    yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3]))
    return roll, pitch, yaw


def tilt_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Return tilt from quaternion."""
    w, x, y, z = q

    # right vector z-component from rotation matrix
    right_z = 2 * (x * z - y * w)

    # tilt angle
    tilt = np.arcsin(np.clip(right_z, -1.0, 1.0))

    return np.degrees(tilt)


def point_cloud_to_open3d(
    points: torch.Tensor,
    segment_indices: torch.Tensor | None = None,
    filter_zero: bool = True,
    world_bounds: tuple[float, float, float, float, float, float] | None = None,
) -> object | None:
    """Convert output of segmented_rgbd_to_point_cloud to an Open3D PointCloud.

    Args:
        points: (N, 3), (H, W, 3), (N, 6), or (H, W, 6) from
            segmented_rgbd_to_point_cloud. Last dim is XYZ or XYZRGB.
        segment_indices: Optional (N,) segment indices. If provided, points are colored
            by segment id, overriding any embedded RGB colors.
        filter_zero: If True, remove points at (0, 0, 0) (invalid/masked). Default True.
        world_bounds: Optional (x_min, y_min, z_min, x_max, y_max, z_max). Points
            outside this axis-aligned box are removed before visualization.

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
    keep = np.ones(xyz.shape[0], dtype=bool)

    if filter_zero:
        keep &= np.any(xyz != 0, axis=1)

    if world_bounds is not None:
        lo = np.array(world_bounds[:3], dtype=np.float64)
        hi = np.array(world_bounds[3:], dtype=np.float64)
        keep &= np.all((xyz >= lo) & (xyz <= hi), axis=1)

    xyz = xyz[keep]
    x = x[keep]
    if seg is not None:
        seg = seg[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # If segment indices are provided, color by segment id.
    if seg is not None:
        colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
        unique_seg = np.unique(seg)
        for sid in unique_seg:
            mask = seg == sid
            r = ((sid * 37) % 255) / 255.0
            g = ((sid * 57) % 255) / 255.0
            b = ((sid * 97) % 255) / 255.0
            colors[mask] = (r, g, b)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    # Otherwise, if embedded RGB is present, use it.
    elif x.shape[-1] >= 6:
        rgb = x[:, 3:6]
        if rgb.size > 0 and rgb.max() > 1.0:
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))

    return pcd


def make_point_marker(pos: np.ndarray, radius: float = 0.01, color: tuple[int] = (1, 0, 0)) -> object:
    """Make a spherical point marker."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(pos)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def create_aabb_lineset(
    bounds: tuple[float, float, float, float, float, float],
) -> object:
    """Create a wireframe box for the AABB with axis-colored edges.

    Args:
        bounds: (x_min, y_min, z_min, x_max, y_max, z_max).

    Returns:
        Open3D LineSet

    """
    x0, y0, z0, x1, y1, z1 = bounds
    # 8 corners ordered so bit pattern (z_bit, y_bit, x_bit) maps to index.
    corners = np.array(
        [
            [x0, y0, z0],  # 0
            [x1, y0, z0],  # 1
            [x0, y1, z0],  # 2
            [x1, y1, z0],  # 3
            [x0, y0, z1],  # 4
            [x1, y0, z1],  # 5
            [x0, y1, z1],  # 6
            [x1, y1, z1],  # 7
        ],
        dtype=np.float64,
    )

    return create_box_lineset(corners)


def create_box_lineset(
    corners: np.ndarray,
) -> object:
    """Create a wireframe box with axis-colored edges.

    Args:
        corners: (8, 3) array of box corners.

    Returns:
        Open3D LineSet

    """
    RED = [1.0, 0.0, 0.0]
    GREEN = [0.0, 1.0, 0.0]
    BLUE = [0.0, 0.0, 1.0]
    PURPLE = [1.0, 0.0, 1.0]

    pos_xyz_corner = corners[7] * 0.95 + corners[0] * 0.05
    corners = np.concatenate([corners, pos_xyz_corner.reshape(1, 3)], axis=0)

    # (edge_start, edge_end, color) grouped by the axis the edge is parallel to.
    edges_and_colors: list[tuple[list[int], list[float]]] = [
        # X-axis edges (differ only in x)
        ([0, 1], RED),
        ([2, 3], RED),
        ([4, 5], RED),
        ([6, 7], RED),
        # Y-axis edges (differ only in y)
        ([0, 2], GREEN),
        ([1, 3], GREEN),
        ([4, 6], GREEN),
        ([5, 7], GREEN),
        # Z-axis edges (differ only in z)
        ([0, 4], BLUE),
        ([1, 5], BLUE),
        ([2, 6], BLUE),
        ([3, 7], BLUE),
        ([7, 8], PURPLE),
    ]
    lines = [e for e, _ in edges_and_colors]
    colors = [c for _, c in edges_and_colors]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def create_camera_model(
    camera_pose: np.ndarray,
    face_width: float = 0.06,
    face_height: float = 0.04,
    face_depth: float = 0.008,
    body_width: float = 0.035,
    body_height: float = 0.03,
    body_depth: float = 0.045,
    marker_width: float = 0.015,
    marker_height: float = 0.008,
    marker_depth: float = 0.01,
) -> list[object] | None:
    """Build a simple 3-part camera model positioned at *camera_pose*.

    The camera looks down its local +Z axis (OpenCV convention).  The model
    consists of:
      1. Face plate -- a thin, wide box at the front (light gray).
      2. Body -- a narrower, longer box behind the face (dark gray).
      3. Top marker -- a small red box on top to indicate orientation.

    Args:
        camera_pose: 7-element array (x, y, z, qw, qx, qy, qz).
        face_*: Dimensions of the front face plate.
        body_*: Dimensions of the camera body.
        marker_*: Dimensions of the red orientation marker.

    Returns:
        List of three Open3D TriangleMesh objects, or None if open3d is
        not installed.

    """
    if o3d is None:
        return None

    from scipy.spatial.transform import Rotation

    pos = camera_pose[:3]
    quat_wxyz = camera_pose[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos

    def _make_box(
        w: float,
        h: float,
        d: float,
        offset: np.ndarray,
        color: tuple[float, float, float],
    ) -> object:
        mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
        # Center the box at the local origin then apply offset.
        mesh.translate(np.array([-w / 2, -h / 2, -d / 2]) + offset)
        mesh.paint_uniform_color(color)
        mesh.transform(T)
        mesh.compute_vertex_normals()
        return mesh

    # Face plate: centered at local origin (front of camera).
    face = _make_box(face_width, face_height, face_depth, np.array([0.0, 0.0, 0.0]), (0.7, 0.7, 0.7))

    # Body: behind the face along -Z.
    body = _make_box(
        body_width,
        body_height,
        body_depth,
        np.array([0.0, 0.0, -(face_depth / 2 + body_depth / 2)]),
        (0.35, 0.35, 0.35),
    )

    # Top marker: on top of the body (-Y in camera frame).
    marker = _make_box(
        marker_width,
        marker_height,
        marker_depth,
        np.array([0.0, -(body_height / 2 + marker_height / 2), -(face_depth / 2 + body_depth / 2)]),
        (0.9, 0.1, 0.1),
    )

    return [face, body, marker]


def draw_instance_annotations(image: np.ndarray, masks: torch.Tensor, segment_ids: torch.Tensor) -> np.ndarray:
    """Draw semi-transparent overlays, bounding boxes, and prompt labels."""
    out = image.copy()
    overlay = image.copy()
    masks_np = masks.detach().cpu().numpy()
    ids_np = segment_ids.detach().cpu().numpy()
    n = masks_np.shape[0]

    for i in range(n):
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        overlay[masks_np[i] > 0] = color
    cv2.addWeighted(overlay, _OVERLAY_ALPHA, out, 1.0 - _OVERLAY_ALPHA, 0, out)

    for i in range(n):
        seg_mask = masks_np[i] > 0
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]

        ys, xs = np.where(seg_mask)
        if len(ys) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(out, (x1, y1), (x2, y2), color, _BBOX_THICKNESS)

        label = f"#{i} obj_{int(ids_np[i])}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        tx, ty = x1, y1 - 6 if y1 - 6 - th >= 0 else y1 + th + 6
        cv2.rectangle(out, (tx - 1, ty - th - 4), (tx + tw + 5, ty + 4), color, cv2.FILLED)
        cv2.putText(out, label, (tx + 2, ty), _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICKNESS, cv2.LINE_AA)

    return out


def get_sorted_object_poses(scene: Scene, obj: SceneObject) -> np.ndarray:
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
            # TODO handle occlusion more robustly. The object closest to the camera
            # Should be the one that is not occluded
            continue
        ob.pose = torch.as_tensor(np.concatenate((poses[det_idx[idx]], [1, 0, 0, 0])), device=device)


def _arrange_panels(panels: list[np.ndarray], gap: int = 10) -> np.ndarray:
    """Arrange panels in a grid with whitespace gaps between them."""
    n = len(panels)
    if n == 0:
        return None
    if n == 1:
        return panels[0]
    if n == 2:
        # Side by side
        h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            if p.shape[0] < h:
                pad = np.zeros((h - p.shape[0], p.shape[1], p.shape[2]), dtype=p.dtype)
                p = np.concatenate([p, pad], axis=0)
            padded.append(p)
        divider = np.zeros((h, gap, 3), dtype=np.uint8)
        return np.concatenate([padded[0], divider, padded[1]], axis=1)

    # 2x2 grid for 3 or 4 panels
    if n == 3:
        # Pad with a blank panel
        blank = np.zeros_like(panels[0])
        panels = panels + [blank]

    top_left, top_right, bot_left, bot_right = panels[0], panels[1], panels[2], panels[3]

    # Normalize all panels to the same size (max dims across all)
    target_h = max(p.shape[0] for p in panels)
    target_w = max(p.shape[1] for p in panels)

    def resize_pad(p):
        """Ensure 3-channel BGR and pad to target size."""
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        elif p.shape[2] == 1:
            p = cv2.cvtColor(p[:, :, 0], cv2.COLOR_GRAY2BGR)
        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        h, w = p.shape[:2]
        out[:h, :w] = p[:target_h, :target_w]
        return out

    tl = resize_pad(top_left)
    tr = resize_pad(top_right)
    bl = resize_pad(bot_left)
    br = resize_pad(bot_right)

    h_gap = np.full((target_h, gap, 3), 255, dtype=np.uint8)
    v_gap = np.full((gap, target_w * 2 + gap, 3), 255, dtype=np.uint8)

    top_row = np.concatenate([tl, h_gap, tr], axis=1)
    bot_row = np.concatenate([bl, h_gap, br], axis=1)

    return np.concatenate([top_row, v_gap, bot_row], axis=0)
