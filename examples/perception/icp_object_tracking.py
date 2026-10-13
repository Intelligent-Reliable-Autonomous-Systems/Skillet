"""Track SAM-segmented objects with ICP and watch their 3D models build up as they rotate.

Sources:
    synthetic  Raycast an asymmetric, face-colored object spinning in front of a virtual camera. Uses
               ground-truth masks by default and reports rotation error against ground truth.
    realsense  Live RealSense RGB-D stream segmented with SAM3.
    recording  Replay frames saved with `--record` (frame_*.npz files).

Examples:
    python examples/perception/icp_object_tracking.py --source synthetic
    python examples/perception/icp_object_tracking.py --source realsense --concepts "mug" --tsdf
    python examples/perception/icp_object_tracking.py --source realsense --concepts "box" --record captures/box
    python examples/perception/icp_object_tracking.py --source recording --recording captures/box --method colored

Keys (focus the "ICP tracking" image window):
    q / ESC  quit           space  pause/resume       r  reset tracker (and SAM session)
    c        toggle scene   t      tint models        m  show canonical models (and TSDF meshes)
    s        save models, meshes and trajectories to --out

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from skillet.perception.reconstruction.icp_object_tracker import (
    ICPObjectTracker,
    ICPTrackerConfig,
    TrackedObject,
    pose_to_matrix,
    shift_image,
    transform_points,
)
from skillet.perception.utils import invalidate_occlusions

if TYPE_CHECKING:
    from collections.abc import Iterator

Frame = dict[str, Any]
"""rgb (H, W, 3) uint8, depth (H, W) meters, intrinsic_k (3, 3), and optionally camera_pose, masks,
labels and gt_obj_to_cam."""

PALETTE = np.array(
    [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]],
    dtype=np.uint8,
)


def track_color(track_id: int) -> np.ndarray:
    """Return a stable RGB color for a track ID."""
    return PALETTE[track_id % len(PALETTE)]


# --------------------------------------------------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------------------------------------------------


def synthetic_object() -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Return an L-shaped mesh (a box body plus an offset handle, no rotational symmetry) and per-triangle colors.

    The body is 8 x 5 x 4 cm, centered on the origin, with its faces along the axes.
    """
    face_colors = np.array(
        [[220, 60, 60], [60, 200, 80], [60, 90, 220], [230, 200, 50], [200, 80, 200], [60, 200, 210]], dtype=np.float64
    )
    parts = []
    part_specs = [  # (size, offset, color tint)
        ((0.08, 0.05, 0.04), (-0.04, -0.025, -0.02), 1.0),
        ((0.02, 0.06, 0.03), (0.04, -0.05, -0.02), 0.6),
    ]
    for size, offset, tint in part_specs:
        part = o3d.geometry.TriangleMesh.create_box(*size).translate(offset)
        part.compute_triangle_normals()
        normals = np.asarray(part.triangle_normals)
        axis = np.abs(normals).argmax(axis=1)
        face = 2 * axis + (normals[np.arange(len(axis)), axis] > 0)
        parts.append((part, face_colors[face] * tint))
    return parts[0][0] + parts[1][0], np.concatenate([parts[0][1], parts[1][1]]).astype(np.uint8)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return an OpenCV-convention (x right, y down, z forward) camera-to-world matrix, with world z up."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    cam_to_world[:3, 3] = eye
    return cam_to_world


def synthetic_frames(
    num_frames: int,
    width: int = 640,
    height: int = 480,
    deg_per_frame: float = 3.0,
    motion: str = "tumble",
    camera_pose: bool = False,
) -> Iterator[Frame]:
    """Yield raycast frames of the synthetic object.

    Args:
        num_frames: number of frames
        width: image width
        height: image height
        deg_per_frame: rotation speed
        motion: `tumble` spins the object about a tilted axis in front of a backdrop, facing the camera.
            `turntable` rests it level on a table, seen from above at an angle, spinning about the vertical.
        camera_pose: include the (world = table frame) camera pose in each frame (turntable only)

    """
    intrinsic_k = np.array([[600.0, 0.0, width / 2], [0.0, 600.0, height / 2], [0.0, 0.0, 1.0]])
    mesh, tri_colors = synthetic_object()

    if motion == "turntable":
        cam_to_world = look_at(np.array([0.0, -0.4, 0.3]), np.array([0.0, 0.0, 0.02]))
        table = o3d.geometry.TriangleMesh.create_box(1.5, 1.5, 0.01).translate((-0.75, -0.75, -0.01))
    else:
        cam_to_world = np.eye(4)
        table = o3d.geometry.TriangleMesh.create_box(1.5, 1.2, 0.01).translate((-0.75, -0.6, 0.75))
    world_to_cam = np.linalg.inv(cam_to_world)
    table_t = o3d.t.geometry.TriangleMesh.from_legacy(table)

    axis = np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2])
    initial = Rotation.from_euler("xyz", [20, -30, 10], degrees=True)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(intrinsic_k), o3d.core.Tensor(world_to_cam), width, height
    )
    ray_np = rays.numpy()
    dirs = ray_np[..., 3:] / np.linalg.norm(ray_np[..., 3:], axis=-1, keepdims=True)
    rng = np.random.default_rng(0)

    for t in range(num_frames):
        obj_to_world = np.eye(4)
        if motion == "turntable":
            obj_to_world[:3, :3] = Rotation.from_euler("z", 25 + deg_per_frame * t, degrees=True).as_matrix()
            obj_to_world[:3, 3] = [0.01 * np.sin(t / 20), 0.01 * np.cos(t / 25), 0.02]  # resting on the table
        else:
            obj_to_world[:3, :3] = (Rotation.from_rotvec(np.radians(deg_per_frame * t) * axis) * initial).as_matrix()
            obj_to_world[:3, 3] = [0.02 * np.sin(t / 20), 0.015 * np.cos(t / 25), 0.5]

        scene = o3d.t.geometry.RaycastingScene()
        posed = o3d.geometry.TriangleMesh(mesh).transform(obj_to_world)
        obj_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(posed))
        scene.add_triangles(table_t)
        hit = scene.cast_rays(rays)

        t_hit = hit["t_hit"].numpy()
        geom = hit["geometry_ids"].numpy()
        prim = hit["primitive_ids"].numpy()
        hit_mask = np.isfinite(t_hit)
        points = ray_np[..., :3] + ray_np[..., 3:] * np.where(hit_mask, t_hit, 0)[..., None]
        depth = np.where(hit_mask, (points - cam_to_world[:3, 3]) @ cam_to_world[:3, 2], 0.0).astype(np.float32)
        depth[hit_mask] += rng.normal(0.0, 0.001, hit_mask.sum()).astype(np.float32)

        shade = 0.35 + 0.65 * np.abs((hit["primitive_normals"].numpy() * dirs).sum(-1))
        obj_mask = hit_mask & (geom == obj_id)
        rgb = np.zeros((height, width, 3), np.float64)
        rgb[hit_mask & ~obj_mask] = [150, 140, 130]
        rgb[obj_mask] = tri_colors[prim[obj_mask]]
        rgb = (rgb * shade[..., None]).clip(0, 255).astype(np.uint8)

        frame = {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": intrinsic_k,
            "masks": obj_mask[None],
            "labels": ["object"],
            "gt_obj_to_cam": world_to_cam @ obj_to_world,
        }
        if camera_pose and motion == "turntable":
            frame["camera_pose"] = cam_to_world
        yield frame


REALSENSE_PRESETS = {"default": 1, "high_accuracy": 3, "high_density": 4}


def realsense_frames(
    width: int,
    height: int,
    fps: int,
    apriltag: dict | None = None,
    laser_power: float | None = None,
    preset: str | None = None,
    depth_filters: str = "temporal",
    depth_resolution: tuple[int, int] = (848, 480),
) -> Iterator[Frame]:
    """Yield depth-aligned frames from a RealSense camera.

    With `apriltag` (kwargs for `CameraLocalizer`), frames carry a world-frame `camera_pose` once the localizer has
    settled on the tag; until then they carry only `tag_detections`, the localizer's progress.

    Depth quality: on a D435i at 0.76 m, the default settings gave 4.9 mm frame-to-frame depth noise on a block
    (47% valid pixels); max laser power with spatial + temporal filtering gave 2.3 mm (90% valid), and the
    high-accuracy preset with filtering 1.8 mm (63% valid). Sensor settings are restored on exit.

    Args:
        width: color (and depth) width
        height: color (and depth) height
        fps: frame rate
        apriltag: kwargs for `CameraLocalizer`, or None to stay in the camera frame
        laser_power: IR projector power (None: the sensor's maximum)
        preset: visual preset name from `REALSENSE_PRESETS`, or None to leave it
        depth_filters: `none`, `spatial`, `temporal` or `spatial_temporal` (smoothing in disparity space). The
            spatial filter smears depth across silhouettes (a 42 mm cube at 0.8 m measured ~60 mm deep in a single
            view, vs ~49 mm with temporal only), so `temporal` is the default
        depth_resolution: depth stream size; 848x480 is the D435's best depth mode (needs USB 3), and depth is
            aligned to the color image either way

    """
    import pyrealsense2 as rs

    localizer = None
    if apriltag is not None:
        from skillet.perception.localization.camera_localizer import CameraLocalizer

        localizer = CameraLocalizer(**apriltag)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, *depth_resolution, rs.format.z16, fps)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    saved = {o: depth_sensor.get_option(o) for o in (rs.option.visual_preset, rs.option.laser_power)}
    if preset is not None:
        depth_sensor.set_option(rs.option.visual_preset, REALSENSE_PRESETS[preset])
    if laser_power is None:
        laser_power = depth_sensor.get_option_range(rs.option.laser_power).max
    depth_sensor.set_option(rs.option.laser_power, laser_power)
    filters = []
    if depth_filters != "none":
        filters = [rs.disparity_transform(True)]
        if "spatial" in depth_filters:
            filters.append(rs.spatial_filter())
        if "temporal" in depth_filters:
            filters.append(rs.temporal_filter())
        filters.append(rs.disparity_transform(False))
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intrinsic_k = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
    align = rs.align(rs.stream.color)
    # Color camera position in the depth camera's frame, for removing alignment occlusion artifacts
    extrinsics = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
    rotation = np.array(extrinsics.rotation).reshape(3, 3).T  # librealsense stores it column-major
    baseline = float(-(rotation.T @ np.array(extrinsics.translation))[0])
    try:
        for _ in range(30):  # let auto-exposure settle
            pipeline.wait_for_frames()
        while True:
            frames = pipeline.wait_for_frames()
            for depth_filter in filters:  # filter in the depth camera's frame, before alignment
                frames = depth_filter.process(frames).as_frameset()
            frames = align.process(frames)
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            frame = {
                "rgb": np.asanyarray(color.get_data()).copy(),
                "depth": invalidate_occlusions(
                    np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale, intrinsic_k[0, 0], baseline
                ),
                "intrinsic_k": intrinsic_k,
            }
            if localizer is not None:
                pose_xyzw = localizer.get_camera_pose(frame["rgb"], intrinsic_k)
                frame["tag_detections"] = len(localizer._pose_buffer)
                if localizer.is_localized:
                    # CameraLocalizer returns (x, y, z, qx, qy, qz, qw); the tracker uses (x, y, z, qw, qx, qy, qz)
                    frame["camera_pose"] = np.concatenate([pose_xyzw[:3], pose_xyzw[[6, 3, 4, 5]]])
            yield frame
    finally:
        for option, value in saved.items():
            depth_sensor.set_option(option, value)
        pipeline.stop()


def recording_has_poses(directory: Path | None) -> bool:
    """Whether a recording made with --apriltag carries camera poses (in its last frame)."""
    paths = sorted(directory.glob("frame_*.npz")) if directory is not None else []
    if not paths:
        return False
    with np.load(paths[-1]) as data:
        return "camera_pose" in data.files


def recording_frames(directory: Path, loop: bool) -> Iterator[Frame]:
    """Yield frames saved by `--record`."""
    paths = sorted(directory.glob("frame_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.npz files in {directory}")
    while True:
        for path in paths:
            with np.load(path) as data:
                yield {k: data[k] for k in data.files}
        if not loop:
            return


# --------------------------------------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------------------------------------


class SamSegmenter:
    """Segment frames with a SAM3 client and return per-instance masks."""

    def __init__(self, model: str, concepts: list[str], local: bool) -> None:
        """Create the SAM client (spawning the SAM server if needed, unless `local`)."""
        from skillet.perception.segmentation.sam import get_sam_client

        self.concepts = concepts
        self.client = get_sam_client(model)(use_server=not local)
        # Track IDs only come back from an in-process SAM3 streaming client; otherwise associate by position
        self.has_track_ids = model == "sam3_streaming" and local

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, list[str], list[int] | None]:
        """Return (N, H, W) masks, N labels and N track IDs (or None)."""
        masks, _, _, concept_indices = self.client.segment_concepts(rgb.transpose(2, 0, 1), self.concepts)
        masks = masks.detach().cpu().numpy().reshape(-1, *rgb.shape[:2]) > 0.5
        labels = [self.concepts[i] for i in concept_indices.cpu().tolist()]
        track_ids = self.client.object_ids.cpu().tolist() if self.has_track_ids else None
        keep = self._drop_contained(masks, labels)
        masks, labels = masks[keep], [labels[i] for i in keep]
        if track_ids is not None:
            track_ids = [track_ids[i] for i in keep]
        return masks, labels, track_ids

    @staticmethod
    def _drop_contained(masks: np.ndarray, labels: list[str], threshold: float = 0.8) -> list[int]:
        """Drop masks mostly inside a larger mask of the same label (SAM often also returns object parts)."""
        areas = masks.sum(axis=(1, 2))
        keep: list[int] = []
        for i in np.argsort(-areas):
            inside = any(
                labels[j] == labels[i] and (masks[i] & masks[j]).sum() > threshold * max(areas[i], 1) for j in keep
            )
            if not inside and areas[i] > 0:
                keep.append(int(i))
        return sorted(keep)

    def reset(self) -> None:
        """Reset any streaming state."""
        self.client.reset()


# --------------------------------------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------------------------------------


def backproject(
    depth: np.ndarray, intrinsic_k: np.ndarray, rgb: np.ndarray, stride: int = 3
) -> o3d.geometry.PointCloud:
    """Back-project a strided depth image into a colored point cloud in the camera frame."""
    d = depth[::stride, ::stride]
    v, u = np.nonzero((d > 0.1) & (d < 2.0))
    z = d[v, u].astype(np.float64)
    u, v = u * stride, v * stride
    points = np.stack(
        [(u - intrinsic_k[0, 2]) * z / intrinsic_k[0, 0], (v - intrinsic_k[1, 2]) * z / intrinsic_k[1, 1], z], axis=1
    )
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(rgb[v, u] / 255.0)
    return pcd


class TrackingViewer:
    """Open3D window showing the scene cloud, each object's fused model at its pose, its axes and trajectory."""

    def __init__(self) -> None:
        """Create the Open3D window."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("ICP object models", width=1280, height=720)
        opt = self.vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.12])
        self.show_scene = True
        self.tint = False
        self._names: list[o3d.geometry.Geometry] = []
        self._view_set = False
        # World origin and axes (x red, y green, z blue)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
        origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        origin.paint_uniform_color((1.0, 1.0, 1.0))
        self.vis.add_geometry(origin)

    def reset_view(self) -> None:
        """Re-fit the view on the next update (e.g. once the camera pose becomes known)."""
        self._view_set = False

    def update(self, frame: Frame, tracker: ICPObjectTracker, cam_to_world: np.ndarray) -> None:
        """Redraw all geometry for this frame."""
        for geom in self._names:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self._names = []

        depth = shift_image(frame["depth"], *tracker.last_depth_shift)
        scene = backproject(depth, frame["intrinsic_k"], frame["rgb"]).transform(cam_to_world)
        self._add(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(cam_to_world))
        if self.show_scene:
            # Grayscale so the colored models stand out
            gray = np.asarray(scene.colors).mean(axis=1, keepdims=True) * 0.6
            scene.colors = o3d.utility.Vector3dVector(np.repeat(gray, 3, axis=1))
            self._add(scene)

        for obj in tracker.objects.values():
            model = obj.model_in_world()
            if self.tint:
                model.paint_uniform_color(track_color(obj.track_id) / 255.0)
            self._add(model)
            size = max(0.03, 0.6 * float(np.max(obj.model.get_max_bound() - obj.model.get_min_bound())))
            self._add(o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(obj.obj_to_world))
            obb = obj.oriented_bbox()
            obb.color = track_color(obj.track_id) / 255.0 if obj.status == "tracking" else (0.5, 0.5, 0.5)
            self._add(obb)
            if len(obj.trajectory) > 1:
                center = obj.model.get_center()[None]
                points = np.concatenate([transform_points(T, center) for T in obj.trajectory[-200:]])
                trail = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(points),
                    o3d.utility.Vector2iVector(np.stack([np.arange(len(points) - 1), np.arange(1, len(points))], 1)),
                )
                trail.paint_uniform_color(track_color(obj.track_id) / 255.0)
                self._add(trail)

        if not self._view_set and len(scene.points) > 0:
            self._set_camera_view(scene, cam_to_world)
            self._view_set = True

    def _add(self, geom: o3d.geometry.Geometry) -> None:
        self.vis.add_geometry(geom, reset_bounding_box=not self._view_set)
        self._names.append(geom)

    def _set_camera_view(self, scene: o3d.geometry.PointCloud, cam_to_world: np.ndarray) -> None:
        """Look at the scene from (roughly) the real camera's viewpoint."""
        ctr = self.vis.get_view_control()
        ctr.set_lookat(np.median(np.asarray(scene.points), axis=0))
        ctr.set_front(-cam_to_world[:3, 2])
        ctr.set_up(-cam_to_world[:3, 1])
        ctr.set_zoom(0.4)

    def poll(self) -> bool:
        """Process window events; returns False once the window is closed."""
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self) -> None:
        """Destroy the window."""
        self.vis.destroy_window()


def draw_overlay(
    frame: Frame, masks: np.ndarray, track_ids: list[int], tracker: ICPObjectTracker, lines: list[str]
) -> np.ndarray:
    """Draw masks colored by track and a status line per object on the RGB image (returns BGR)."""
    image = frame["rgb"].copy()
    for mask, track_id in zip(masks, track_ids, strict=False):
        color = track_color(track_id)
        image[mask] = (0.5 * image[mask] + 0.5 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color.tolist(), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lines.extend(object_summary(obj) for obj in tracker.objects.values())
    for i, line in enumerate(lines):
        y = 22 + 20 * i
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def object_summary(obj: TrackedObject) -> str:
    """One-line status of a tracked object."""
    rotvec = obj.rotation_from_initial.as_rotvec(degrees=True)
    return (
        f"#{obj.track_id} {obj.label}  {obj.status}  fit={obj.fitness:.2f} rmse={1000 * obj.inlier_rmse:.1f}mm  "
        f"rot={np.linalg.norm(rotvec):.0f}deg  pts={len(obj.model.points)} fused={obj.num_fusions}"
    )


def show_canonical_models(tracker: ICPObjectTracker) -> None:
    """Show every model (and TSDF mesh) in its own object frame, side by side. Blocks until closed."""
    geometries = []
    offset = 0.0
    for obj in tracker.objects.values():
        extent = obj.model.get_max_bound() - obj.model.get_min_bound()
        shift = np.array([offset, 0.0, 0.0])
        geometries.append(o3d.geometry.PointCloud(obj.model).translate(shift))
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).translate(shift))
        if (mesh := obj.mesh()) is not None and len(mesh.triangles) > 0:
            geometries.append(mesh.translate(shift + np.array([0.0, 1.5 * extent[1], 0.0])))
        offset += 1.5 * float(extent.max())
    if geometries:
        o3d.visualization.draw_geometries(geometries, window_name="Canonical object models (q to close)")


# --------------------------------------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------------------------------------


def rotation_error_deg(
    obj: TrackedObject, gt_initial: np.ndarray, gt_now: np.ndarray, cam_to_world: np.ndarray
) -> float:
    """Angle between the estimated and ground-truth rotation since the object was first seen."""
    gt_rel = (cam_to_world[:3, :3] @ gt_now[:3, :3]) @ (cam_to_world[:3, :3] @ gt_initial[:3, :3]).T
    est_rel = obj.rotation_from_initial.as_matrix()
    return float(np.degrees(Rotation.from_matrix(est_rel.T @ gt_rel).magnitude()))


def main() -> None:
    """Run the demo."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["synthetic", "realsense", "recording"], default="synthetic")
    parser.add_argument("--recording", type=Path, help="Directory of frame_*.npz files for --source recording")
    parser.add_argument("--loop", action="store_true", help="Loop the recording")
    parser.add_argument("--record", type=Path, help="Save every frame to this directory (for later replay)")
    parser.add_argument("--frames", type=int, default=240, help="Number of synthetic frames")
    parser.add_argument(
        "--motion", choices=["tumble", "turntable"], default="turntable", help="Synthetic object motion"
    )
    parser.add_argument("--canonical", choices=["face", "world"], default="face", help="Initial object frame")
    parser.add_argument(
        "--upright", action="store_true", help="Only allow rotation about up (objects slid or spun on the table)"
    )
    parser.add_argument(
        "--reanchor",
        choices=["conservative", "aggressive", "off"],
        default="conservative",
        help="Periodic re-anchoring to the confident model core; `aggressive` suits symmetric objects like cubes",
    )
    parser.add_argument(
        "--depth-registration",
        choices=["auto", "fixed", "none"],
        default="none",
        help="Correct depth misregistered with the RGB masks: per-frame search, a fixed --depth-shift, or none",
    )
    parser.add_argument("--depth-shift", type=int, nargs=2, default=[0, 0], help="Fixed depth shift (du dv) in pixels")
    parser.add_argument(
        "--background-filter",
        action="store_true",
        help="Drop mask-edge pixels lying on the background surface (for data without occlusion invalidation)",
    )
    parser.add_argument(
        "--depth-resolution", type=int, nargs=2, default=[848, 480], help="Depth stream size (848 480 needs USB 3)"
    )
    parser.add_argument("--laser-power", type=float, help="RealSense IR projector power (default: maximum)")
    parser.add_argument("--preset", choices=sorted(REALSENSE_PRESETS), help="RealSense visual preset")
    parser.add_argument(
        "--depth-filters", choices=["none", "spatial", "temporal", "spatial_temporal"], default="temporal"
    )
    parser.add_argument(
        "--apriltag", action="store_true", help="Localize the RealSense with an AprilTag; track only once localized"
    )
    parser.add_argument("--tag-id", type=int, default=1)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=0.1, help="Tag side length in meters")
    parser.add_argument(
        "--tag-pose",
        type=float,
        nargs=7,
        default=[0.14, -0.01, 0.0, 0.0, 0.0, 0.7071068, -0.7071068],
        help="Tag pose in the world frame: x y z qx qy qz qw",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--segmenter",
        choices=["auto", "gt", "sam3", "sam3_streaming"],
        default="auto",
        help="auto = ground truth for synthetic, sam3 otherwise",
    )
    parser.add_argument("--concepts", default="object", help="Comma-separated SAM concepts, e.g. 'mug,box'")
    parser.add_argument("--sam-local", action="store_true", help="Run SAM in-process instead of via the SAM server")
    parser.add_argument(
        "--method", choices=["point_to_point", "point_to_plane", "colored", "generalized"], default="point_to_plane"
    )
    parser.add_argument("--voxel", type=float, default=0.003, help="Voxel size in meters")
    parser.add_argument("--keyframe-deg", type=float, default=10.0, help="Rotation between model fusions")
    parser.add_argument(
        "--fuse-rmse", type=float, default=0.003, help="Max ICP inlier RMSE (m) to fuse a view; raise for noisy depth"
    )
    parser.add_argument("--tsdf", action="store_true", help="Also fuse a TSDF volume per object (press m to view)")
    parser.add_argument("--no-relocalize", action="store_true", help="Disable FPFH + RANSAC recovery")
    parser.add_argument("--out", type=Path, default=Path("output/icp_models"), help="Where `s` saves models")
    parser.add_argument("--no-vis", action="store_true", help="Run headless and print per-frame status")
    args = parser.parse_args()

    config = ICPTrackerConfig(
        voxel_size=args.voxel,
        method=args.method,
        keyframe_rotation_deg=args.keyframe_deg,
        fuse_max_rmse=args.fuse_rmse,
        canonical_frame=args.canonical,
        require_camera_pose=args.apriltag or recording_has_poses(args.recording),
        upright=args.upright,
        reanchor_every=0 if args.reanchor == "off" else ICPTrackerConfig.reanchor_every,
        reanchor_jumps="aggressive" if args.reanchor == "aggressive" else "conservative",
        depth_registration=args.depth_registration,
        depth_shift_px=tuple(args.depth_shift),
        background_tolerance=0.005 if args.background_filter else 0.0,
        tsdf=args.tsdf,
        relocalize=not args.no_relocalize,
    )
    tracker = ICPObjectTracker(config)

    if args.source == "synthetic":
        frames = synthetic_frames(args.frames, args.width, args.height, motion=args.motion, camera_pose=True)
    elif args.source == "realsense":
        apriltag = None
        if args.apriltag:
            apriltag = {
                "apriltag_pose": np.asarray(args.tag_pose),
                "apriltag_size_m": args.tag_size,
                "apriltag_id": args.tag_id,
                "apriltag_fam": args.tag_family,
            }
        frames = realsense_frames(
            args.width,
            args.height,
            args.fps,
            apriltag,
            args.laser_power,
            args.preset,
            args.depth_filters,
            tuple(args.depth_resolution),
        )
    else:
        if args.recording is None:
            parser.error("--source recording requires --recording DIR")
        frames = recording_frames(args.recording, args.loop)

    segmenter_name = args.segmenter
    if segmenter_name == "auto":
        # Synthetic frames and recordings carry masks; re-segment a recording with --segmenter sam3
        segmenter_name = "gt" if args.source in ("synthetic", "recording") else "sam3"
    segmenter = None
    if segmenter_name != "gt":
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
        segmenter = SamSegmenter(segmenter_name, concepts, args.sam_local)

    if args.record is not None:
        args.record.mkdir(parents=True, exist_ok=True)

    viewer = None if args.no_vis else TrackingViewer()
    window = "ICP tracking"
    paused = False
    frame = None
    gt_initial: dict[int, np.ndarray] = {}
    localized = False
    errors: list[float] = []

    try:
        frame_idx = 0
        while True:
            if not paused or frame is None:
                frame = next(frames, None)
                if frame is None:
                    break

                t0 = time.perf_counter()
                if segmenter is None:
                    masks, labels, track_ids = frame["masks"], list(frame["labels"]), None
                else:
                    masks, labels, track_ids = segmenter(frame["rgb"])
                t_seg = time.perf_counter() - t0
                if args.record is not None:
                    # Save the masks used too, so a replay can reproduce the tracking without re-running SAM
                    recorded = {**frame, "masks": masks, "labels": np.asarray(labels, dtype=str)}
                    np.savez_compressed(args.record / f"frame_{frame_idx:05d}.npz", **recorded)

                camera_pose = frame.get("camera_pose")
                cam_to_world = np.eye(4) if camera_pose is None else pose_to_matrix(camera_pose)

                t0 = time.perf_counter()
                tracker.update(
                    frame["rgb"], frame["depth"], frame["intrinsic_k"], masks, labels, track_ids, camera_pose
                )
                t_icp = time.perf_counter() - t0

                # Recover which track each mask went to, for coloring the overlay
                if track_ids is None:
                    track_ids = [_nearest_track(tracker, m, frame, cam_to_world) for m in masks]

                du, dv = tracker.last_depth_shift
                lines = [
                    f"frame {frame_idx}  seg {1000 * t_seg:.0f}ms  icp {1000 * t_icp:.0f}ms  depth shift ({du},{dv})px"
                ]
                if args.apriltag and camera_pose is None:
                    lines.append(f"waiting for AprilTag localization ({frame.get('tag_detections', 0)}/10)")
                if viewer is not None and (camera_pose is not None) != localized:
                    viewer.reset_view()
                localized = camera_pose is not None
                if "gt_obj_to_cam" in frame and segmenter is None and tracker.objects:
                    obj = next(iter(tracker.objects.values()))
                    gt_initial.setdefault(obj.track_id, frame["gt_obj_to_cam"])
                    err = rotation_error_deg(obj, gt_initial[obj.track_id], frame["gt_obj_to_cam"], cam_to_world)
                    errors.append(err)
                    lines.append(f"ground-truth rotation error {err:.2f}deg")

                if args.no_vis:
                    print(" | ".join(lines + [object_summary(o) for o in tracker.objects.values()]))
                frame_idx += 1

            if viewer is None:
                continue

            viewer.update(frame, tracker, cam_to_world)
            if not viewer.poll():
                break
            cv2.imshow(window, draw_overlay(frame, masks, track_ids, tracker, list(lines)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                tracker.reset()
                gt_initial.clear()
                if segmenter is not None:
                    segmenter.reset()
                print("[INFO] Tracker reset")
            elif key == ord("c"):
                viewer.show_scene = not viewer.show_scene
            elif key == ord("t"):
                viewer.tint = not viewer.tint
            elif key == ord("m"):
                show_canonical_models(tracker)
            elif key == ord("s"):
                for path in tracker.save(args.out):
                    print(f"[INFO] Saved {path}")
    finally:
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()

    if errors:
        print(f"[INFO] Rotation error vs ground truth: mean {np.mean(errors):.2f}deg, max {np.max(errors):.2f}deg")
    for obj in tracker.objects.values():
        print(f"[INFO] {object_summary(obj)}")


def _nearest_track(tracker: ICPObjectTracker, mask: np.ndarray, frame: Frame, cam_to_world: np.ndarray) -> int:
    """Find the tracked object whose model projects closest to the mask's centroid (for display only)."""
    v, u = np.nonzero(mask)
    if len(u) == 0 or not tracker.objects:
        return -1
    k = frame["intrinsic_k"]
    world_to_cam = np.linalg.inv(cam_to_world)
    best, best_dist = -1, np.inf
    for obj in tracker.objects.values():
        center = transform_points(world_to_cam, obj.center[None])[0]
        if center[2] <= 0:
            continue
        pixel = np.array([k[0, 0] * center[0] / center[2] + k[0, 2], k[1, 1] * center[1] / center[2] + k[1, 2]])
        dist = np.linalg.norm(pixel - [u.mean(), v.mean()])
        if dist < best_dist:
            best, best_dist = obj.track_id, dist
    return best


if __name__ == "__main__":
    main()
