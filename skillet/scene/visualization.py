"""Open3D visualization for point clouds from object localization."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from skillet.scene.utils import (
    create_aabb_lineset,
    create_camera_model,
    get_object_geometry,
    make_point_marker,
    point_cloud_to_open3d,
    quat_to_roll_pitch_yaw,
    tilt_from_quat_wxyz,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from skillet.core import BatchedEnvironment, Environment
    from skillet.scene.base import Scene

try:
    import open3d as o3d
    import open3d.visualization.gui as _gui
    import open3d.visualization.rendering as _rendering
except ImportError:
    o3d = None  # type: ignore[assignment]
    _gui = None  # type: ignore[assignment]
    _rendering = None  # type: ignore[assignment]


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
        env: Environment | BatchedEnvironment,
        window_name: str = "Skillet Table Scene",
        width: int = 640,
        height: int = 480,
        get_tcp_pos: Callable[[], Sequence[float]] | None = None,
    ) -> None:
        self.scene = scene
        self.env = env
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
        self._open3d_scene: np.ndarray | None = None
        self._tcp_pose: np.ndarray | None = None
        self._gripper_pos: np.ndarray | None = None
        self._get_tcp_pose = self.get_tcp_pose
        self._get_gripper_pos = self.get_gripper_pos
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._render_event = threading.Event()

    @property
    def open3d_scene(self) -> np.ndarray:
        return self._open3d_scene

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
        self._mat_unlit.shader = "defaultLitTransparency"
        self._mat_unlit.base_color = [1.0, 0.0, 0.0, 0.7]
        self._mat_unlit.point_size = 3 * self._window.scaling

        self._mat_lit = _rendering.MaterialRecord()
        self._mat_lit.shader = "defaultLit"

        self._mat_line = _rendering.MaterialRecord()
        self._mat_line.shader = "unlitLine"
        self._mat_line.line_width = 2 * self._window.scaling

        # Static geometries
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self._add_geometry("coord_frame", coord, self._mat_lit)

        if self.scene is not None and self.scene.bounds is not None:
            bounds_ls = create_aabb_lineset(self.scene.bounds)
            if bounds_ls is not None:
                self._add_geometry("scene_bounds", bounds_ls, self._mat_line)

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
            if self._get_tcp_pose is not None:
                pose = self._get_tcp_pose()
                if isinstance(pose, torch.Tensor):
                    pose = pose.detach().cpu().numpy().astype(np.float64)
                if pose is not None:
                    self._tcp_pose = np.asarray(pose, dtype=np.float64)
                    if self._tcp_pose.ndim > 1:
                        self._tcp_pose = self._tcp_pose[0]
            if self._get_gripper_pos is not None:
                pos = self._get_gripper_pos()
                if isinstance(pose, torch.Tensor):
                    pos = pos.detach().cpu().numpy().astype(np.float64)
                if pos is not None:
                    self._gripper_pos = np.asarray(pos, dtype=np.float64)
                    if self._gripper_pos.ndim > 1:
                        self._gripper_pos = self._gripper_pos[0]

            if self._tcp_pose is not None:
                sphere = make_point_marker(self._tcp_pose[:3], radius=0.007, color=(1, 0, 1))
                self._add_geometry("tcp_pos", sphere, self._mat_lit)
                width = (0.1 if self._gripper_pos < 0.5 else 0.03) if self._gripper_pos is not None else 0.08
                gripper = self.gripper_mesh(self._tcp_pose[0:3], self._tcp_pose[3:7], width=width)
                self._add_geometry("gripper", gripper, self._mat_unlit)
            else:
                self._remove_geometry("tcp_pos")
                self._remove_geometry("gripper")

            # HUD
            if hud_text:
                self._hud_label.text = hud_text
                self._window.set_needs_layout()

            # Auto-fit view on first valid point cloud
            if do_camera_setup and pcd is not None:
                bounds = self._scene_widget.scene.bounding_box
                center = bounds.get_center()
                self._scene_widget.setup_camera(60, bounds, center)
                eye = center + [1.2, -0.8, 0.0]
                up = [0.0, 0.0, 1.0]

                self._scene_widget.scene.camera.look_at(center, eye, up)
                self._needs_camera_setup = False

        def _on_image(img):
            self._open3d_scene = np.asarray(img)
            self._render_event.set()

        def _render() -> None:
            self._scene_widget.scene.scene.render_to_image(_on_image)
            self._scene_widget.force_redraw()
            self._render_event.wait(timeout=1.0)

        self._app.post_to_main_thread(self._window, _do_update)
        self._app.post_to_main_thread(self._window, _render)

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
        if self._target_pos is not None and self._target_pos.ndim > 1:
            self._target_pos = self._target_pos[0]
        self._target_size = size

    def gripper_mesh(
        self,
        xyz: np.ndarray,
        quat_wxyz: np.ndarray,
        width: float = 0.08,
        depth: float = 0.02,
        height: float = 0.02,
        finger_len: float = 0.06,
    ) -> object:
        """Create a simple parallel-jaw gripper mesh with the pose at the finger tips midpoint.

        Args:
            xyz: (3,) position of the midpoint between the fingers
            quat_wxyz: (4,) quaternion [w, x, y, z]
            width: distance between fingers
            depth: base depth
            height: thickness
            finger_len: finger length

        Returns:
            open3d.geometry.TriangleMesh

        """
        # --- Helper: quaternion → rotation matrix ---
        w, x, y, z = quat_wxyz
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

        # Base
        base = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        base.translate([-width / 2, -height / 2, -depth / 2])

        # Fingers
        finger_w = 0.01
        finger_h = height
        finger_d = finger_len

        left_finger = o3d.geometry.TriangleMesh.create_box(finger_w, finger_h, finger_d)
        right_finger = o3d.geometry.TriangleMesh.create_box(finger_w, finger_h, finger_d)

        # Position fingers relative to base
        left_finger.translate([-width / 2, -height / 2, depth / 2])
        right_finger.translate([width / 2 - finger_w, -height / 2, depth / 2])

        gripper = base + left_finger + right_finger

        # Compute offset to move fingers to midpoint
        # Finger tips are at z = depth/2 + finger_len
        # x coordinates of tips: left = -width/2 + finger_w/2, right = width/2 - finger_w/2
        tip_left_local = np.array([-width / 2 + finger_w / 2, 0, depth / 2 + finger_d])
        tip_right_local = np.array([width / 2 - finger_w / 2, 0, depth / 2 + finger_d])
        midpoint_local = (tip_left_local + tip_right_local) / 2.0

        # Apply rotation first, then translate to xyz
        gripper.translate(-midpoint_local)  # move midpoint to origin
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        gripper.transform(T)

        # Lighting and color
        gripper.paint_uniform_color([0.8, 0.2, 0.2])
        gripper.compute_vertex_normals()

        return gripper

    def get_tcp_pose(self) -> Sequence[float]:
        """Get the TCP post from the environment."""
        return (
            self.env.get_observation(self.env.batched_env.obs_spec_ikee.unbatched())["tcp_pose_b"]
            .detach()
            .cpu()
            .numpy()
        )

    def get_gripper_pos(self) -> Sequence[float]:
        """Get the gripper position from the environment."""
        return (
            self.env.get_observation(self.env.batched_env.obs_spec_ikee.unbatched())["gripper"].detach().cpu().numpy()
        )
