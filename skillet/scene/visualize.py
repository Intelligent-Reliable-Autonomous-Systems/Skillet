"""Open3D visualization for point clouds from object localization."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from skillet.scene.base import Scene, SceneObject
from skillet.scene.cube import Cube

try:
    import open3d as o3d
    import open3d.visualization.gui as _gui
    import open3d.visualization.rendering as _rendering
except ImportError:
    o3d = None  # type: ignore[assignment]
    _gui = None  # type: ignore[assignment]
    _rendering = None  # type: ignore[assignment]


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
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    return pcd


def make_point_marker(pos, radius=0.01, color=(1, 0, 0)):
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


class Open3DVisualizer:
    """Stateful Open3D visualizer using the ``gui.Application`` API.

    Renders a point cloud with optional world-bounds wireframe, camera model,
    and a HUD label showing the current camera pose.  Designed to ``run()`` on
    the main thread while a perception pipeline pushes updates from a
    background thread via ``update()``.
    """

    _CAM_GEOMETRY_NAMES = ("cam_face", "cam_body", "cam_marker")

    def __init__(
        self,
        scene: Scene,
        window_name: str = "Table Scene",
        width: int = 1024,
        height: int = 768,
        get_tcp_pos: Callable[[], Sequence[float]] | None = None,
    ) -> None:
        self.scene = scene
        self._window_name = window_name
        self._width = width
        self._height = height

        self._app: Any | None = None
        self._window: Any | None = None
        self._scene_widget: Any | None = None
        self._hud_label: Any | None = None
        self._mat_unlit: Any | None = None
        self._mat_lit: Any | None = None
        self._mat_line: Any | None = None

        self._added_geometries: set[str] = set()
        self._needs_camera_setup = True
        self._closed = False
        self._target_pos: np.ndarray | None = None
        self._target_size: float = 0.007
        self._tcp_pos: np.ndarray | None = None
        self._get_tcp_pos = get_tcp_pos

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Initialize gui.Application, create window / scene / HUD / static geometry."""
        self._app = _gui.Application.instance
        self._app.initialize()

        self._window = self._app.create_window(
            self._window_name,
            self._width,
            self._height,
        )
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)

        self._scene_widget = _gui.SceneWidget()
        self._scene_widget.scene = _rendering.Open3DScene(self._window.renderer)
        self._window.add_child(self._scene_widget)

        self._hud_label = _gui.Label("Cam: waiting for data...")
        self._window.add_child(self._hud_label)

        # Materials
        self._mat_unlit = _rendering.MaterialRecord()
        self._mat_unlit.shader = "defaultUnlit"
        self._mat_unlit.point_size = 3 * self._window.scaling

        self._mat_lit = _rendering.MaterialRecord()
        self._mat_lit.shader = "defaultLit"

        self._mat_line = _rendering.MaterialRecord()
        self._mat_line.shader = "unlitLine"
        self._mat_line.line_width = 2 * self._window.scaling

        # Static geometries
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self._add_geometry("coord_frame", coord, self._mat_lit)

        if self.scene.bounds is not None:
            bounds_ls = create_aabb_lineset(self.scene.bounds)
            if bounds_ls is not None:
                self._add_geometry("scene_bounds", bounds_ls, self._mat_line)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _add_geometry(self, name: str, geom: Any, mat: Any) -> None:
        """Add or replace a named geometry in the scene."""
        scene = self._scene_widget.scene
        if name in self._added_geometries:
            scene.remove_geometry(name)
        if isinstance(geom, list):
            for g in geom:
                scene.add_geometry(name, g, mat)
        else:
            scene.add_geometry(name, geom, mat)
        self._added_geometries.add(name)

    def _remove_geometry(self, name: str) -> None:
        if name in self._added_geometries:
            self._scene_widget.scene.remove_geometry(name)
            self._added_geometries.discard(name)

    # ------------------------------------------------------------------
    # Layout / close callbacks (called on main thread by gui framework)
    # ------------------------------------------------------------------

    def _on_layout(self, layout_context: Any) -> None:
        r = self._window.content_rect
        self._scene_widget.frame = r
        pref = self._hud_label.calc_preferred_size(
            layout_context,
            _gui.Widget.Constraints(),
        )
        self._hud_label.frame = _gui.Rect(
            r.x + 10,
            r.y + 10,
            pref.width,
            pref.height,
        )

    def _on_close(self) -> bool:
        self._closed = True
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        point_cloud: torch.Tensor,
        segment_indices: torch.Tensor | None = None,
        camera_pose: torch.Tensor | None = None,
    ) -> None:
        """Push new data from any thread; the scene update runs on the GUI thread."""
        if self._closed or self._app is None or self._window is None:
            return

        # Prepare heavy geometry conversion on the calling (perception) thread.
        pcd = point_cloud_to_open3d(
            point_cloud,
            segment_indices=segment_indices,
            filter_zero=True,
            world_bounds=self.scene.bounds,
        )

        cam_meshes: list[Any] | None = None
        hud_text = ""
        if camera_pose is not None:
            cam_np = camera_pose.detach().cpu().numpy().astype(np.float64)
            cam_meshes = create_camera_model(cam_np)
            pos = cam_np[:3]
            q = cam_np[3:7]
            roll, pitch, yaw = quat_to_roll_pitch_yaw(q)
            hud_text = (
                f"Cam: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
                f"q=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}) \n"
                f"rpy=({roll:.3f}, {pitch:.3f}, {yaw:.3f})  "
                f"Tilt: {tilt_from_quat_wxyz(q):.3f}°"
            )

        do_camera_setup = self._needs_camera_setup

        def _do_update() -> None:
            if self._closed:
                return

            # Point cloud
            if pcd is not None:
                self._add_geometry("pcd", pcd, self._mat_unlit)
            else:
                self._remove_geometry("pcd")

            # Camera model (3 meshes)
            if cam_meshes is not None:
                for name, mesh in zip(
                    self._CAM_GEOMETRY_NAMES,
                    cam_meshes,
                    strict=True,
                ):
                    self._add_geometry(name, mesh, self._mat_lit)
            else:
                for name in self._CAM_GEOMETRY_NAMES:
                    self._remove_geometry(name)

            for obj in self.scene.objects:
                geometry = get_object_geometry(obj)
                if geometry is not None and len(geometry) > 0:
                    self._add_geometry(obj.identifier, geometry, self._mat_line)
                else:
                    self._remove_geometry(obj.identifier)

            # Target position sphere
            if self._target_pos is not None:
                sphere = make_point_marker(self._target_pos, radius=self._target_size, color=(1, 0.5, 0))
                self._add_geometry("target_pos", sphere, self._mat_lit)
            else:
                self._remove_geometry("target_pos")

            # TCP position sphere
            if self._get_tcp_pos is not None:
                xyz = self._get_tcp_pos()
                if isinstance(xyz, torch.Tensor):
                    xyz = xyz.detach().cpu().numpy().astype(np.float64)
                if xyz is not None:
                    self._tcp_pos = np.array(xyz, dtype=np.float64)
                    if self._tcp_pos.ndim > 1:
                        self._tcp_pos = self._tcp_pos[0]
            if self._tcp_pos is not None:
                sphere = make_point_marker(self._tcp_pos[:3], radius=0.007, color=(1, 0, 1))
                self._add_geometry("tcp_pos", sphere, self._mat_lit)
            else:
                self._remove_geometry("tcp_pos")

            # HUD
            if hud_text:
                self._hud_label.text = hud_text
                self._window.set_needs_layout()

            # Auto-fit view on first valid point cloud
            if do_camera_setup and pcd is not None:
                bounds = self._scene_widget.scene.bounding_box
                center = bounds.get_center()
                self._scene_widget.setup_camera(60, bounds, center)
                self._needs_camera_setup = False

        self._app.post_to_main_thread(self._window, _do_update)

    def run(self) -> None:
        """Set up the window and block on the GUI event loop (call on main thread)."""
        if o3d is None:
            raise ImportError("Open3D is required for visualization. Install with: pip install open3d")
        self._setup()
        self._app.run()

    def run_thread(self) -> None:
        """Run the visualizer in a thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="VisualizerThread", daemon=True)
        self._thread.start()

    def stop_thread(self) -> None:
        """Stop the visualizer thread."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

    def request_close(self) -> None:
        """Request the GUI to shut down (safe to call from any thread)."""
        self._closed = True
        if self._app is not None:
            self._app.quit()

    def set_target_pos(self, xyz: Sequence[float] | None, size: float = 0.007) -> None:
        """Set the target position sphere marker. Pass None to clear."""
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy().astype(np.float64)
        self._target_pos = np.array(xyz, dtype=np.float64) if xyz is not None else None
        if self._target_pos is not None:
            if self._target_pos.ndim > 1:
                self._target_pos = self._target_pos[0]
        self._target_size = size


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


def tilt_from_quat_wxyz(q):
    w, x, y, z = q

    # right vector z-component from rotation matrix
    right_z = 2 * (x * z - y * w)

    # tilt angle
    tilt = np.arcsin(np.clip(right_z, -1.0, 1.0))

    tilt_deg = np.degrees(tilt)
    return tilt_deg  # degrees
