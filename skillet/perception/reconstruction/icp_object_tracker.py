"""icp_object_tracker.py.

Track SAM-segmented objects with ICP while incrementally building their 3D models.

Each tracked object owns a canonical model (a point cloud, and optionally a TSDF volume) expressed in its
own object frame. Every frame, the masked depth of the object is registered against that model with
coarse-to-fine ICP to recover its full 6-DoF pose. When the registration is confident and the object shows
the camera a sufficiently new side, the observation is fused into the model, so the model fills in as the
object rotates and later frames have more surface to lock onto.

The object frame is fixed by the first observation, with its origin at the observed centroid. By default its
axes snap to the object's faces closest to the world axes: z to the top face (objects usually start resting
level on the table) and x to the side face whose normal is closest to world x; see
`ICPTrackerConfig.canonical_frame`. No shape or size priors are needed.

Conventions:
    - ``rgb``: (H, W, 3) or (3, H, W) uint8
    - ``depth``: (H, W) float depth in meters
    - ``camera_pose``: 4x4 camera-to-world matrix or 7D (x, y, z, qw, qx, qy, qz); identity if omitted
    - ``obj_to_world``: 4x4 matrix mapping object-frame points to the world frame
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from collections.abc import Sequence

reg = o3d.pipelines.registration

ICPMethod = Literal["point_to_point", "point_to_plane", "colored", "generalized"]


@dataclass
class ICPTrackerConfig:
    """Tuning parameters for `ICPObjectTracker`. Distances are in meters."""

    # Depth-to-color registration
    depth_registration: Literal["none", "fixed", "auto"] = "none"
    """Correct depth that is misregistered with the RGB image the masks come from (bad depth-color extrinsics,
    or color and depth captured at slightly different times while things move). `fixed` shifts the depth image
    by `depth_shift_px`; `auto` searches, every frame, for the shift that puts the masked objects in front of
    their surroundings."""
    depth_shift_px: tuple[int, int] = (0, 0)
    """(du, dv) shift applied to the depth image for `fixed` registration: depth moves right/down by du/dv."""
    max_depth_shift_px: int = 24
    """Search radius for `auto` registration."""
    depth_shift_window: int = 15
    """`auto` applies the median of the last N per-frame estimates: the calibration part of the error is constant,
    and single-frame estimates are noisy."""

    # Preprocessing
    voxel_size: float = 0.003
    """Resolution at `noise_reference_depth`; with `noise_adaptive`, scaled with the expected depth noise."""
    noise_adaptive: bool = True
    """Scale each object's resolution and every threshold tied to it (ICP distances, merge RMSE, background
    tolerance, normal radius) with the expected depth noise. Stereo depth noise grows with distance squared,
    so settings tuned at arm's length would chase noise for an object twice as far away."""
    noise_reference_depth: float = 0.45
    """Depth (m) at which the configured sizes apply unscaled."""
    max_noise_scale: float = 4.0
    max_resolution_scale: float = 1.0
    """Cap on how much the model resolution coarsens with noise. Noise tolerances scale fully, but a coarse
    model of a small object has too few points to pin its orientation."""
    """Resolution of the observations and of the fused model."""
    depth_min: float = 0.1
    depth_max: float = 2.0
    mask_erosion: int = 2
    """Iterations of 3x3 erosion applied to each mask, dropping flying pixels at depth discontinuities."""
    background_tolerance: float = 0.0
    """Drop mask pixels within this distance of the local surface just outside the mask: background seen through
    the mask edge (RGB-D misalignment or loose masks). 0 disables. Off by default: invalidating alignment
    occlusions at the camera (`skillet.perception.utils.invalidate_occlusions`) removes most such pixels, and on
    small or distant objects the filter also eats into the object. 0.005 suits data without that step."""
    background_band_px: int = 20
    """Only mask pixels within this many pixels of the mask border are tested against the background (at
    `noise_reference_depth`; the band narrows with distance like the misregistration it targets)."""
    flying_pixel_threshold: float = 0.0
    """Drop pixels whose depth differs from a 4-neighbour by more than this (m, at `noise_reference_depth`, growing
    linearly with depth like a pixel's footprint): mixed foreground/background pixels at silhouettes. 0 disables.
    Off by default: a threshold tight enough to catch edge smears also deletes steep, nearly edge-on faces."""
    max_depth_spread: float = 0.15
    """Drop masked points farther than this from the median depth (background leaking through the mask)."""
    largest_cluster: bool = True
    """Keep only the largest DBSCAN cluster of each observation."""
    min_points: int = 50
    """Observations with fewer points after preprocessing are ignored."""

    # Registration
    method: ICPMethod = "point_to_plane"
    """`colored` can disambiguate geometrically symmetric objects (cubes, cylinders) that have texture, but
    assumes their colors don't change with lighting as they turn."""
    scales: tuple[float, ...] = (1.0,)
    """Coarse-to-fine ICP voxel sizes, as multiples of `voxel_size`, e.g. (2.0, 1.0). A single fine scale is the
    default: on small objects coarse levels tend to slide newly revealed faces onto the wrong part of the model."""
    iterations: tuple[int, ...] = (50,)
    yaw_candidates_deg: tuple[float, ...] = ()
    """Extra initial guesses: the prediction rotated by +/- each angle about up (through the model center), tried
    when the prediction doesn't fit confidently. Helps with fast spins at low frame rates. Keep these below half
    the object's symmetry angle (45 deg for a cube)."""
    reacquire_scales: tuple[float, ...] = (4.0, 2.0, 1.0)
    """Wider coarse-to-fine scales used when re-acquiring an object, where the motion prior is stale."""
    reacquire_iterations: tuple[int, ...] = (40, 30, 30)
    correspondence_factor: float = 2.0
    """Max correspondence distance at each scale, as a multiple of that scale's voxel size."""
    robust_kernel: bool = True
    """Down-weight outliers (occluding fingers, segmentation noise, newly revealed surface) with a Tukey loss."""
    kernel_factor: float = 1.0
    """Tukey loss cutoff, as a multiple of each scale's voxel size."""

    # Acceptance and fusion
    track_min_fitness: float = 0.3
    """Minimum ICP fitness (fraction of the observation with a model correspondence) to accept the pose."""
    fuse_min_fitness: float = 0.5
    """Minimum ICP fitness to fuse the observation into the model. Keep this moderate: newly revealed surface
    has no model correspondences yet, so a high threshold would stop the model from ever growing."""
    fuse_max_rmse: float = 0.003
    """Maximum inlier RMSE to fuse. This is the main guard against fusing a misaligned view; raise it for
    noisier depth sensors."""
    fuse_min_stable_frames: int = 3
    """Consecutive fusable frames required before fusing, so a bad re-acquisition can't corrupt the model."""
    keyframe_rotation_deg: float = 10.0
    """Fuse once the object has turned this much relative to the camera since the last fusion..."""
    keyframe_translation: float = 0.03
    """...or moved this much relative to the camera."""
    max_model_points: int = 30000
    confirm_within: int = 4
    """A newly fused model point must be re-observed by another fused view within this many fusions, or it is
    pruned. Views fused at a wrong pose rarely agree with later views, so their points fade out instead of
    becoming part of the object."""
    max_confidence: float = 20.0
    """Cap on a point's confirmation count, so long-lived points can still adapt."""
    core_min_confidence: float = 3.0
    """Points confirmed by at least this many views form the model's confident core, used for re-anchoring."""
    outlier_removal_every: int = 5
    """Run statistical outlier removal on the model every N fusions (0 disables)."""

    # Re-anchoring against the confident core
    reanchor_every: int = 10
    """Every N tracked frames, re-register the view against the confident core, both locally and globally, and
    snap to the closest well-fitting pose. Corrects drift; for symmetric objects the snap may jump by a
    symmetry, but the axes stay attached to the same kind of faces. 0 disables."""
    reanchor_margin: float = 0.1
    """A global candidate must fit the core better than the local re-registration by this much to cause a jump;
    otherwise only the local correction is applied. While the core covers just part of the object, a wrong
    global pose can lay the view onto the known faces nearly as well as the right one."""
    reanchor_jumps: Literal["conservative", "aggressive"] = "conservative"
    """When a global candidate may replace the tracked pose. `aggressive`: whenever it fits the core clearly
    better; recovers well, but for asymmetric objects can jump to a wrong pose that lays a view showing new
    surface onto the known faces. `conservative`: it must also fit the full model (including recently seen,
    not yet confirmed surface) as well as the current pose. Use `aggressive` for symmetric objects such as
    cubes, where a symmetric jump keeps the axes on the faces."""
    reanchor_full_model_tolerance: float = 0.02
    """For `conservative` jumps: how much worse than the current pose a jump may fit the full model."""
    reanchor_fitness_tolerance: float = 0.05
    """Global candidates within this fitness of the best one count as equally good; the closest to the current
    pose wins (for symmetric objects, the smallest symmetric jump)."""

    seed: int | None = 0
    """Seed for the random sampling in global registration (None for nondeterministic)."""

    # Data association and recovery
    max_association_distance: float = 0.1
    """Max distance between a detection and an object's predicted center to associate them."""
    relocalize: bool = True
    """Also try global FPFH + RANSAC registration when an object is re-acquired or lost."""
    relocalize_below_fitness: float = 0.5
    """Only relocalize when ICP (from the last pose, with the wide scales) fits worse than this."""
    relocalize_trials: int = 5
    """Independent RANSAC hypotheses per relocalization; the best-fitting refined one is kept."""
    relocalize_min_fitness: float = 0.5
    relocalize_margin: float = 0.15
    """Fitness by which a global registration must beat ICP from the last pose to override it. Partial views
    of near-symmetric objects often fit a flipped pose almost as well, so the motion-consistent result wins ties."""

    # Canonical object frame
    require_camera_pose: bool = False
    """Ignore frames without a `camera_pose` (e.g. until an AprilTag camera localizer has locked on), so objects
    are only created once their orientation can be expressed in the world frame."""
    canonical_frame: Literal["face", "world"] = "face"
    """How to orient a new object's frame. `face`: z along up, snapped to the object's top face if it has one,
    and x along the side face whose normal (or its opposite) is closest to world x; without a flat side face
    (e.g. a cylinder) x is world x made horizontal. `world`: axes aligned with the world frame."""
    up_source: Literal["auto", "world", "table", "camera"] = "auto"
    """Where "up" comes from. `world`: `world_up` (meaningful when `camera_pose` is given). `table`: normal of the
    dominant plane in the depth image outside the masks. `camera`: image up. `auto`: `world` if a camera pose
    is given, else `table`, falling back to `camera` if no plane is found."""
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    top_face_snap_deg: float = 15.0
    """Snap up to the mean normal of the object's top-facing points when they are within this angle of up."""
    face_tolerance_deg: float = 15.0
    """Side normals within this angle of each other count as the same face."""
    min_face_fraction: float = 0.2
    """Fraction of the side-facing normals a face needs to count as a face rather than a curved surface."""
    table_min_inlier_fraction: float = 0.3
    """Fraction of the background points the table plane must explain to be trusted."""
    upright: bool = False
    """Only allow rotation about the up axis relative to the first observation (objects slid or spun on a table).
    Removes tilt drift, but is wrong for objects that get picked up and turned over."""

    # Optional TSDF fusion, for a mesh of the object
    tsdf: bool = False
    tsdf_voxel_size: float = 0.002
    tsdf_trunc_factor: float = 4.0
    """TSDF truncation distance, as a multiple of `tsdf_voxel_size`."""


@dataclass
class TrackedObject:
    """A tracked object: its canonical model and current pose."""

    track_id: int
    label: str
    obj_to_world: np.ndarray
    model: o3d.geometry.PointCloud
    """Fused point cloud in the object frame."""
    tsdf: o3d.pipelines.integration.ScalableTSDFVolume | None = None
    status: Literal["tracking", "lost"] = "tracking"
    fitness: float = 1.0
    inlier_rmse: float = 0.0
    num_fusions: int = 1
    frames_unseen: int = 0
    trajectory: list[np.ndarray] = field(default_factory=list)
    """History of accepted `obj_to_world` poses."""

    _prev_obj_to_world: np.ndarray | None = None
    _keyframe_obj_to_cam: np.ndarray | None = None
    _pyramid: dict[float, o3d.geometry.PointCloud] = field(default_factory=dict)
    _stable_frames: int = 0
    _up: np.ndarray | None = None
    confidence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Per model point: how many fused views confirmed it."""
    _birth: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    _frames_since_anchor: int = 0
    voxel: float = 0.003
    """Model resolution, fixed when the object is created."""
    noise: float = 0.003
    """Distance unit for noise-dependent thresholds (noise-scaled `voxel_size`), fixed at creation."""

    @property
    def position(self) -> np.ndarray:
        """Object frame origin in the world frame, (3,)."""
        return self.obj_to_world[:3, 3].copy()

    @property
    def pose(self) -> np.ndarray:
        """7D world pose (x, y, z, qw, qx, qy, qz)."""
        return matrix_to_pose(self.obj_to_world)

    @property
    def rotation_from_initial(self) -> Rotation:
        """Rotation of the object (in the world frame) since it was first seen."""
        return Rotation.from_matrix(self.obj_to_world[:3, :3] @ self.trajectory[0][:3, :3].T)

    @property
    def center(self) -> np.ndarray:
        """Centroid of the fused model in the world frame, (3,)."""
        return transform_points(self.obj_to_world, self.model.get_center()[None])[0]

    def model_in_world(self) -> o3d.geometry.PointCloud:
        """Return a copy of the fused model at the object's current pose."""
        return o3d.geometry.PointCloud(self.model).transform(self.obj_to_world)

    def oriented_bbox(self) -> o3d.geometry.OrientedBoundingBox:
        """Return a box fit to the model in the object frame, placed at the current pose."""
        obb = self.model.get_minimal_oriented_bounding_box(robust=True)
        rot = self.obj_to_world[:3, :3]
        return o3d.geometry.OrientedBoundingBox(
            transform_points(self.obj_to_world, np.asarray(obb.center)[None])[0], rot @ obb.R, obb.extent
        )

    def mesh(self) -> o3d.geometry.TriangleMesh | None:
        """Extract the TSDF mesh in the object frame (None if TSDF fusion is disabled)."""
        if self.tsdf is None:
            return None
        mesh = self.tsdf.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


@dataclass
class _Observation:
    """A preprocessed object observation."""

    pcd: o3d.geometry.PointCloud
    """Points in the world frame, with colors and camera-facing normals."""
    pixel_mask: np.ndarray
    """Pixels that produced the points, (H, W) bool."""
    centroid: np.ndarray
    scale: float = 1.0
    """Noise scale the observation was processed at (see `ICPTrackerConfig.noise_adaptive`)."""


class ICPObjectTracker:
    """Track objects from per-frame instance masks and build their 3D models with ICP."""

    def __init__(self, config: ICPTrackerConfig | None = None) -> None:
        """Initialize the tracker.

        Args:
            config: Tracker parameters; defaults to `ICPTrackerConfig()`

        """
        self.config = config or ICPTrackerConfig()
        if self.config.seed is not None:
            o3d.utility.random.seed(self.config.seed)
        self.objects: dict[int, TrackedObject] = {}
        self._next_id = 0
        self.last_depth_shift: tuple[int, int] = (0, 0)
        """(du, dv) depth shift applied on the last frame (see `ICPTrackerConfig.depth_registration`)."""
        self._shift_history: deque[tuple[int, int]] = deque(maxlen=self.config.depth_shift_window)

    def reset(self) -> None:
        """Forget every tracked object."""
        self.objects = {}
        self._next_id = 0
        self._shift_history.clear()

    def update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic_k: np.ndarray,
        masks: np.ndarray,
        labels: Sequence[str] | None = None,
        track_ids: Sequence[int] | None = None,
        camera_pose: np.ndarray | None = None,
    ) -> dict[int, TrackedObject]:
        """Update the tracked objects with a new RGB-D frame and its instance masks.

        Args:
            rgb: (H, W, 3) or (3, H, W) uint8 image
            depth: (H, W) depth in meters
            intrinsic_k: 3x3 camera intrinsics
            masks: (N, H, W) instance masks, bool or float in [0, 1]
            labels: N class labels, used to keep association within a class
            track_ids: N stable instance IDs (e.g. from SAM3 streaming); associated by position if omitted
            camera_pose: 4x4 camera-to-world matrix or 7D (x, y, z, qw, qx, qy, qz); camera frame if omitted

        Returns:
            All tracked objects, keyed by track ID

        """
        rgb = _to_hwc_uint8(rgb)
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        intrinsic_k = np.asarray(intrinsic_k, dtype=np.float64).reshape(3, 3)
        masks = np.asarray(masks).reshape(-1, *depth.shape) > 0.5
        labels = list(labels) if labels is not None else ["object"] * len(masks)
        if camera_pose is None and self.config.require_camera_pose:
            return self.objects
        if self.config.depth_registration == "fixed":
            self.last_depth_shift = tuple(self.config.depth_shift_px)
        elif self.config.depth_registration == "auto" and len(masks) > 0:
            self._shift_history.append(self._estimate_depth_shift(depth, masks))
            du, dv = np.median(np.array(self._shift_history), axis=0)
            self.last_depth_shift = (round(du), round(dv))
        if self.last_depth_shift != (0, 0):
            depth = shift_image(depth, *self.last_depth_shift)
        cam_to_world = np.eye(4) if camera_pose is None else pose_to_matrix(camera_pose)

        observations = [self._preprocess(rgb, depth, m, intrinsic_k, cam_to_world) for m in masks]
        if track_ids is None:
            track_ids = self._associate(observations, labels)

        seen = set()
        up = None
        for obs, label, track_id in zip(observations, labels, track_ids, strict=True):
            if obs is None:
                continue
            track_id = int(track_id)
            if track_id in self.objects:
                self._track(self.objects[track_id], obs, rgb, depth, intrinsic_k, cam_to_world)
            else:
                if up is None and self.config.canonical_frame == "face":
                    up = self._up_direction(depth, masks, intrinsic_k, cam_to_world, camera_pose is not None)
                self._create(track_id, label, obs, rgb, depth, intrinsic_k, cam_to_world, up)
            seen.add(track_id)

        for track_id, obj in self.objects.items():
            if track_id not in seen:
                obj.frames_unseen += 1
                obj._prev_obj_to_world = None
        return self.objects

    def save(self, out_dir: str | Path) -> list[Path]:
        """Save each object's model, mesh (if any) and pose trajectory to `out_dir`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for obj in self.objects.values():
            stem = f"{obj.label.replace(' ', '_')}_{obj.track_id}"
            path = out_dir / f"{stem}_model.ply"
            o3d.io.write_point_cloud(str(path), obj.model)
            written.append(path)
            if (mesh := obj.mesh()) is not None:
                path = out_dir / f"{stem}_mesh.ply"
                o3d.io.write_triangle_mesh(str(path), mesh)
                written.append(path)
            path = out_dir / f"{stem}_trajectory.npy"
            np.save(path, np.stack(obj.trajectory))
            written.append(path)
        return written

    def _preprocess(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsic_k: np.ndarray,
        cam_to_world: np.ndarray,
    ) -> _Observation | None:
        """Back-project a mask into a clean, downsampled point cloud with normals in the world frame."""
        cfg = self.config
        if cfg.mask_erosion > 0:
            mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=cfg.mask_erosion) > 0
        valid = mask & np.isfinite(depth) & (depth > cfg.depth_min) & (depth < cfg.depth_max)
        if valid.sum() < cfg.min_points:
            return None
        if cfg.flying_pixel_threshold > 0:
            threshold = cfg.flying_pixel_threshold * max(
                1.0, float(np.median(depth[valid])) / cfg.noise_reference_depth
            )
            valid &= ~flying_pixels(depth, threshold)
            if valid.sum() < cfg.min_points:
                return None
        scale = self._noise_scale(float(np.median(depth[valid])))
        voxel = min(scale, cfg.max_resolution_scale) * cfg.voxel_size
        noise = scale * cfg.voxel_size  # distance unit for noise-dependent radii
        if cfg.background_tolerance > 0:
            depth_m = float(np.median(depth[valid]))
            band = max(3, round(cfg.background_band_px * min(1.0, cfg.noise_reference_depth / depth_m)))
            tolerance = scale * cfg.background_tolerance
            valid &= ~self._background_pixels(depth, mask, valid, intrinsic_k, tolerance, band)
            if valid.sum() < cfg.min_points:
                return None
        valid &= np.abs(depth - np.median(depth[valid])) < cfg.max_depth_spread

        v, u = np.nonzero(valid)
        z = depth[v, u].astype(np.float64)
        fx, fy, cx, cy = intrinsic_k[0, 0], intrinsic_k[1, 1], intrinsic_k[0, 2], intrinsic_k[1, 2]
        points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(rgb[v, u].astype(np.float64) / 255.0)
        pcd = pcd.voxel_down_sample(voxel)

        if cfg.largest_cluster and len(pcd.points) >= cfg.min_points:
            clusters = np.asarray(pcd.cluster_dbscan(eps=3 * max(voxel, noise), min_points=5))
            if clusters.max() >= 0:
                largest = np.bincount(clusters[clusters >= 0]).argmax()
                pcd = pcd.select_by_index(np.flatnonzero(clusters == largest))
        if len(pcd.points) < cfg.min_points:
            return None

        # Estimate normals in the camera frame so they can be oriented toward the camera
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * noise, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.zeros(3))
        pcd.transform(cam_to_world)
        return _Observation(pcd=pcd, pixel_mask=valid, centroid=pcd.get_center(), scale=scale)

    def _noise_scale(self, depth: float) -> float:
        """Return the expected depth noise at `depth` relative to the reference depth (grows with depth^2)."""
        cfg = self.config
        if not cfg.noise_adaptive:
            return 1.0
        return float(np.clip((depth / cfg.noise_reference_depth) ** 2, 1.0, cfg.max_noise_scale))

    def _estimate_depth_shift(self, depth: np.ndarray, masks: np.ndarray) -> tuple[int, int]:
        """Find the depth image shift that best registers it with the masks.

        A correctly registered object sits in front of the ring of pixels around its mask, so a depth threshold
        separates the two cleanly; misregistration pulls background depth into the mask and object depth out of
        it. Edges where the object meets its support carry no depth step, so the shift is best constrained across
        the object's free edges.
        """
        kernel = np.ones((3, 3), np.uint8)
        inner_v, inner_u, ring_v, ring_u = [], [], [], []
        for mask in masks:
            m = mask.astype(np.uint8)
            inner = np.nonzero(cv2.erode(m, kernel, iterations=2))
            ring = np.nonzero((cv2.dilate(m, kernel, iterations=6) > 0) & ~mask)
            inner_v.append(inner[0])
            inner_u.append(inner[1])
            ring_v.append(ring[0])
            ring_u.append(ring[1])
        inner_v, inner_u = np.concatenate(inner_v), np.concatenate(inner_u)
        ring_v, ring_u = np.concatenate(ring_v), np.concatenate(ring_u)
        h, w = depth.shape

        def score(du: int, dv: int) -> float:
            # Depth shifted by (du, dv) has, at pixel (v, u), the original depth at (v - dv, u - du)
            a = depth[np.clip(inner_v - dv, 0, h - 1), np.clip(inner_u - du, 0, w - 1)]
            b = depth[np.clip(ring_v - dv, 0, h - 1), np.clip(ring_u - du, 0, w - 1)]
            a, b = a[a > 0], b[b > 0]
            if len(a) < 50 or len(b) < 50:
                return -np.inf
            # How cleanly a depth threshold separates mask (in front) from ring (behind)
            threshold = 0.5 * (np.median(a) + np.median(b))
            return float((a < threshold).mean() + (b > threshold).mean())

        # Ties (the score is flat near the optimum) go to the smallest shift
        def key(shift: tuple[int, int]) -> tuple[float, int]:
            return round(score(*shift), 3), -abs(shift[0]) - abs(shift[1])

        r = self.config.max_depth_shift_px
        du0, dv0 = max(((du, dv) for du in range(-r, r + 1, 2) for dv in range(-r, r + 1, 2)), key=key)
        return max(((du0 + i, dv0 + j) for i in (-1, 0, 1) for j in (-1, 0, 1)), key=key)

    def _background_pixels(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        valid: np.ndarray,
        intrinsic_k: np.ndarray,
        tolerance: float,
        band_px: int,
    ) -> np.ndarray:
        """Flag mask pixels near the border that lie on the background surface just outside the mask.

        RGB-D misalignment and loose masks let background (table, wall, other objects) leak in along mask edges.
        Leaked pixels continue the local surface of the nearest background, while object pixels stand off it.
        Testing against that local plane, rather than a single global table plane or a depth threshold, works
        for any background and keeps concave objects intact: a cup's inner back wall is not on the surface
        beyond its rim. Only the object's bottom few mm, where it rests on the support, are lost; objects
        thinner than `background_tolerance` cannot be separated from what they lie on.
        """
        cfg = self.config
        h, w = depth.shape
        in_range = (depth > cfg.depth_min) & (depth < cfg.depth_max)
        # Background reference pixels, kept a couple of pixels clear of the mask edge
        outside = ~(cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & in_range
        if outside.sum() < 10:
            return np.zeros_like(mask)
        dist, labels = cv2.distanceTransformWithLabels(
            (~outside).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        band = valid & (dist <= band_px)
        if not band.any():
            return np.zeros_like(mask)

        # Pixel coordinates of each outside label, and the unique references used by the band
        oy, ox = np.nonzero(outside)
        label_yx = np.zeros((labels.max() + 1, 2), dtype=np.int64)
        label_yx[labels[oy, ox]] = np.stack([oy, ox], axis=1)
        by, bx = np.nonzero(band)
        refs, ref_index = np.unique(labels[by, bx], return_inverse=True)

        def backproject(v: np.ndarray, u: np.ndarray) -> np.ndarray:
            z = depth[v, u].astype(np.float64)
            fx, fy, cx, cy = intrinsic_k[0, 0], intrinsic_k[1, 1], intrinsic_k[0, 2], intrinsic_k[1, 2]
            return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)

        # Fit a plane to the outside pixels in a window around each reference
        r = 6
        dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
        wy = np.clip(label_yx[refs, 0, None] + dy.ravel(), 0, h - 1)
        wx = np.clip(label_yx[refs, 1, None] + dx.ravel(), 0, w - 1)
        weight = outside[wy, wx].astype(np.float64)
        pts = backproject(wy, wx)
        count = weight.sum(axis=1)
        mean = (weight[..., None] * pts).sum(axis=1) / np.maximum(count, 1)[:, None]
        centered = (pts - mean[:, None]) * weight[..., None]
        cov = np.einsum("nki,nkj->nij", centered, pts - mean[:, None]) / np.maximum(count, 1)[:, None, None]
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, :, 0]
        # Trust only well-supported, roughly planar neighborhoods
        planar = (count >= 15) & (eigvals[:, 0] < 0.1 * eigvals[:, 1])

        offset = np.abs(((backproject(by, bx) - mean[ref_index]) * normal[ref_index]).sum(axis=1))
        on_background = planar[ref_index] & (offset < tolerance)
        flagged = np.zeros_like(mask)
        flagged[by[on_background], bx[on_background]] = True
        return flagged

    def _associate(self, observations: list[_Observation | None], labels: list[str]) -> list[int]:
        """Match observations to tracked objects of the same label by distance to their predicted centers."""
        objects = list(self.objects.values())
        big = 1e6
        cost = np.full((len(observations), len(objects)), big)
        for i, (obs, label) in enumerate(zip(observations, labels, strict=True)):
            if obs is None:
                continue
            for j, obj in enumerate(objects):
                if obj.label != label:
                    continue
                predicted = transform_points(self._predict(obj), obj.model.get_center()[None])[0]
                dist = np.linalg.norm(obs.centroid - predicted)
                # Lost objects may have moved anywhere; let relocalization sort them out
                if dist <= self.config.max_association_distance or obj.status == "lost":
                    cost[i, j] = dist

        track_ids = [-1] * len(observations)
        if objects:
            for i, j in zip(*linear_sum_assignment(cost), strict=True):
                if cost[i, j] < big:
                    track_ids[i] = objects[j].track_id
        for i, obs in enumerate(observations):
            if track_ids[i] == -1 and obs is not None:
                track_ids[i] = self._next_id
                self._next_id += 1
        return track_ids

    def _create(
        self,
        track_id: int,
        label: str,
        obs: _Observation,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic_k: np.ndarray,
        cam_to_world: np.ndarray,
        up: np.ndarray | None = None,
    ) -> None:
        """Start a new object whose frame sits at the observation's centroid (oriented per `canonical_frame`)."""
        cfg = self.config
        obj_to_world = np.eye(4)
        obj_to_world[:3, 3] = obs.centroid
        if up is not None:
            obj_to_world[:3, :3] = self._canonical_rotation(obs, up)
        model = o3d.geometry.PointCloud(obs.pcd).transform(np.linalg.inv(obj_to_world))
        tsdf = None
        if cfg.tsdf:
            tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=cfg.tsdf_voxel_size,
                sdf_trunc=cfg.tsdf_trunc_factor * cfg.tsdf_voxel_size,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
        obj = TrackedObject(track_id=track_id, label=label, obj_to_world=obj_to_world, model=model, tsdf=tsdf)
        obj.voxel = min(obs.scale, cfg.max_resolution_scale) * cfg.voxel_size
        obj.noise = obs.scale * cfg.voxel_size
        obj.confidence = np.ones(len(model.points))
        obj._birth = np.ones(len(model.points), dtype=np.int64)
        obj.trajectory.append(obj_to_world.copy())
        obj._up = np.asarray(cfg.world_up, dtype=np.float64) if up is None else up
        obj._keyframe_obj_to_cam = np.linalg.inv(cam_to_world) @ obj_to_world
        if tsdf is not None:
            self._integrate_tsdf(obj, rgb, depth, obs.pixel_mask, intrinsic_k, cam_to_world)
        self.objects[track_id] = obj

    def _up_direction(
        self,
        depth: np.ndarray,
        masks: np.ndarray,
        intrinsic_k: np.ndarray,
        cam_to_world: np.ndarray,
        has_camera_pose: bool,
    ) -> np.ndarray:
        """Return the world-frame up direction according to `up_source`."""
        source = self.config.up_source
        if source == "auto":
            source = "world" if has_camera_pose else "table"
        if source == "world":
            up = np.asarray(self.config.world_up, dtype=np.float64)
            return up / np.linalg.norm(up)
        if source == "table" and (normal := self._table_normal(depth, masks, intrinsic_k)) is not None:
            return cam_to_world[:3, :3] @ normal
        return -cam_to_world[:3, 1]  # image up (camera -y)

    def _table_normal(self, depth: np.ndarray, masks: np.ndarray, intrinsic_k: np.ndarray) -> np.ndarray | None:
        """Fit the dominant plane outside the object masks; return its camera-frame normal facing the camera."""
        stride = 4
        background = ~masks.any(axis=0) & np.isfinite(depth) & (depth > self.config.depth_min)
        background &= depth < self.config.depth_max
        v, u = np.nonzero(background[::stride, ::stride])
        z = depth[::stride, ::stride][v, u].astype(np.float64)
        u, v = u * stride, v * stride
        if len(z) < 200:
            return None
        points = np.stack(
            [(u - intrinsic_k[0, 2]) * z / intrinsic_k[0, 0], (v - intrinsic_k[1, 2]) * z / intrinsic_k[1, 1], z], 1
        )
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.01)
        if len(pcd.points) < 50:
            return None
        plane, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=300)
        if len(inliers) < self.config.table_min_inlier_fraction * len(pcd.points):
            return None
        normal = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
        # The camera (origin) is on the side the normal should point to: n . 0 + d > 0
        return normal if plane[3] > 0 else -normal

    def _canonical_rotation(self, obs: _Observation, up: np.ndarray) -> np.ndarray:
        """Orient a new object along its faces closest to the world axes.

        z is up, snapped to the top face; x is the side-face axis closest to world x.
        """
        cfg = self.config
        normals = np.asarray(obs.pcd.normals)
        z = up / np.linalg.norm(up)

        top = normals[normals @ z > np.cos(np.radians(cfg.top_face_snap_deg))]
        if len(top) >= max(10, 0.05 * len(normals)):
            z = top.mean(axis=0)
            z /= np.linalg.norm(z)

        # Reference x: world x made horizontal (world y if x is nearly vertical, e.g. without a camera pose)
        reference = np.array([1.0, 0.0, 0.0]) - z[0] * z
        if np.linalg.norm(reference) < 0.3:
            reference = np.array([0.0, 1.0, 0.0]) - z[1] * z
        reference /= np.linalg.norm(reference)
        x = reference

        # Cluster the horizontal components of side-facing normals into faces; a face's axis is its normal or the
        # opposite (the unseen face across a box), whichever is closer to the reference
        side = normals[np.abs(normals @ z) < np.sin(np.radians(45.0))]
        if len(side) >= 10:
            side = side[np.random.default_rng(0).permutation(len(side))[:1000]]
            side = side - np.outer(side @ z, z)
            side /= np.linalg.norm(side, axis=1, keepdims=True)
            similar = side @ side.T > np.cos(np.radians(cfg.face_tolerance_deg))
            unassigned = np.ones(len(side), dtype=bool)
            best_alignment = -np.inf
            while unassigned.sum() >= cfg.min_face_fraction * len(side):
                votes = (similar & unassigned).sum(axis=1) * unassigned
                members = similar[votes.argmax()] & unassigned
                unassigned &= ~members
                if members.sum() < cfg.min_face_fraction * len(side):
                    break
                face = side[members].mean(axis=0)
                face /= np.linalg.norm(face)
                # Either of the face's two in-plane axes (its normal, or the perpendicular) can serve as x
                for axis in (face, np.cross(z, face)):
                    alignment = abs(axis @ reference)
                    if alignment > best_alignment:
                        best_alignment, x = alignment, np.sign(axis @ reference) * axis

        x = x - (x @ z) * z
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z], axis=1)

    def _track(
        self,
        obj: TrackedObject,
        obs: _Observation,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic_k: np.ndarray,
        cam_to_world: np.ndarray,
    ) -> None:
        """Register the observation to the object's model, update its pose and fuse it if warranted."""
        cfg = self.config
        reacquired = obj.frames_unseen > 0 or obj.status == "lost"
        obj.frames_unseen = 0

        # Try a constant-velocity prediction first, then the last pose
        candidates = [self._predict(obj)]
        if obj._prev_obj_to_world is not None:
            candidates.append(obj.obj_to_world)
        best = None
        for obj_to_world in candidates:
            result = self._register(obs.pcd, obj, np.linalg.inv(obj_to_world))
            if best is None or (result.fitness, -result.inlier_rmse) > (best.fitness, -best.inlier_rmse):
                best = result
            if best.fitness >= cfg.fuse_min_fitness:
                break
        max_rmse = cfg.fuse_max_rmse * obj.noise / cfg.voxel_size
        confident = best.fitness >= cfg.fuse_min_fitness and best.inlier_rmse <= 0.75 * max_rmse
        if cfg.yaw_candidates_deg and not confident:
            center = obj.model.get_center()
            for angle in cfg.yaw_candidates_deg:
                for sign in (-1.0, 1.0):
                    result = self._register(
                        obs.pcd, obj, np.linalg.inv(self._yawed(obj, candidates[0], center, sign * angle))
                    )
                    if (result.fitness, -result.inlier_rmse) > (best.fitness, -best.inlier_rmse):
                        best = result

        # After a gap (or a failed fit) the motion prior is stale: widen the search. Mediocre fits during
        # tracking are left alone, since they are usually just newly revealed surface, and global
        # registration happily snaps partially symmetric shapes into a flipped pose.
        if reacquired or best.fitness < cfg.track_min_fitness:
            result = self._register(obs.pcd, obj, np.linalg.inv(obj.obj_to_world), wide=True)
            if (result.fitness, -result.inlier_rmse) > (best.fitness, -best.inlier_rmse):
                best = result
            if cfg.relocalize and best.fitness < cfg.relocalize_below_fitness:
                result = self._relocalize(obs.pcd, obj)
                if (
                    result is not None
                    and result.fitness >= cfg.relocalize_min_fitness
                    and result.fitness > best.fitness + cfg.relocalize_margin
                ):
                    best = result

        obj.fitness, obj.inlier_rmse = best.fitness, best.inlier_rmse
        if best.fitness < cfg.track_min_fitness:
            obj.status = "lost"
            obj._prev_obj_to_world = None
            obj._stable_frames = 0
            return
        fusable = best.fitness >= cfg.fuse_min_fitness and best.inlier_rmse <= max_rmse
        obj._stable_frames = obj._stable_frames + 1 if fusable and not reacquired else 0

        obj.status = "tracking"
        obj._prev_obj_to_world = obj.obj_to_world
        obj.obj_to_world = np.linalg.inv(best.transformation)
        if cfg.upright:
            obj.obj_to_world = self._make_upright(obj, obj.obj_to_world)
        obj._frames_since_anchor += 1
        if cfg.reanchor_every and obj._frames_since_anchor >= cfg.reanchor_every:
            obj._frames_since_anchor = 0
            self._reanchor(obj, obs)
        obj.trajectory.append(obj.obj_to_world.copy())

        if obj._stable_frames >= cfg.fuse_min_stable_frames and self._is_keyframe(obj, cam_to_world):
            self._fuse(obj, obs, np.linalg.inv(obj.obj_to_world))
            obj._keyframe_obj_to_cam = np.linalg.inv(cam_to_world) @ obj.obj_to_world
            if obj.tsdf is not None:
                self._integrate_tsdf(obj, rgb, depth, obs.pixel_mask, intrinsic_k, cam_to_world)

    @staticmethod
    def _yawed(obj: TrackedObject, obj_to_world: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate a pose about the object's up axis through its model center."""
        rotation = Rotation.from_rotvec(np.radians(angle_deg) * obj._up).as_matrix()
        pivot = transform_points(obj_to_world, center[None])[0]
        yawed = obj_to_world.copy()
        yawed[:3, :3] = rotation @ obj_to_world[:3, :3]
        yawed[:3, 3] = rotation @ (obj_to_world[:3, 3] - pivot) + pivot
        return yawed

    def _make_upright(self, obj: TrackedObject, obj_to_world: np.ndarray) -> np.ndarray:
        """Keep only the rotation about up (since the first observation), pivoting about the model's center."""
        initial = obj.trajectory[0][:3, :3]
        quat = Rotation.from_matrix(obj_to_world[:3, :3] @ initial.T).as_quat(scalar_first=True)
        # Swing-twist decomposition: the twist about `up` is the quaternion's projection onto it
        w, v = quat[0], quat[1:]
        twist = np.concatenate([[w], (v @ obj._up) * obj._up])
        if np.linalg.norm(twist) < 1e-9:
            return obj_to_world
        rotation = Rotation.from_quat(twist / np.linalg.norm(twist), scalar_first=True).as_matrix() @ initial
        center = obj.model.get_center()
        upright = np.eye(4)
        upright[:3, :3] = rotation
        upright[:3, 3] = transform_points(obj_to_world, center[None])[0] - rotation @ center
        return upright

    def _predict(self, obj: TrackedObject) -> np.ndarray:
        """Predict the object's pose by repeating its last motion, expressed in the object frame."""
        if obj._prev_obj_to_world is None:
            return obj.obj_to_world
        body_motion = np.linalg.inv(obj._prev_obj_to_world) @ obj.obj_to_world
        return obj.obj_to_world @ body_motion

    def _register(
        self,
        source: o3d.geometry.PointCloud,
        obj: TrackedObject,
        init_world_to_obj: np.ndarray,
        wide: bool = False,
        core: bool = False,
    ) -> reg.RegistrationResult:
        """Coarse-to-fine ICP of the observation (world frame) onto the model (object frame).

        The partial observation is the source so every observed point looks for support on the more
        complete model, rather than hidden model surfaces pulling toward the visible side.
        """
        cfg = self.config
        scales = cfg.reacquire_scales if wide else cfg.scales
        iterations_per_scale = cfg.reacquire_iterations if wide else cfg.iterations
        transform = init_world_to_obj
        result = None
        for scale, iterations in zip(scales, iterations_per_scale, strict=True):
            target = self._model_at_scale(obj, scale, core)
            voxel = scale * obj.voxel
            src = source.voxel_down_sample(voxel) if scale > 1.0 else source
            unit = max(voxel, scale * obj.noise)
            max_dist = cfg.correspondence_factor * unit
            criteria = reg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iterations)
            estimation = self._estimation(unit)
            try:
                if cfg.method == "colored":
                    result = reg.registration_colored_icp(src, target, max_dist, transform, estimation, criteria)
                elif cfg.method == "generalized":
                    result = reg.registration_generalized_icp(src, target, max_dist, transform, estimation, criteria)
                else:
                    result = reg.registration_icp(src, target, max_dist, transform, estimation, criteria)
            except RuntimeError:
                # Raised e.g. by colored ICP when there are no correspondences at all
                result = reg.RegistrationResult()
                result.transformation = transform
                return result
            transform = result.transformation
        return result

    def _estimation(self, unit: float) -> Any:  # noqa: ANN401
        """Build the ICP transformation estimator for the configured method."""
        method = self.config.method
        kernel = reg.TukeyLoss(k=self.config.kernel_factor * unit) if self.config.robust_kernel else None
        if method == "point_to_point":
            return reg.TransformationEstimationPointToPoint()
        if method == "point_to_plane":
            if kernel:
                return reg.TransformationEstimationPointToPlane(kernel)
            return reg.TransformationEstimationPointToPlane()
        if method == "colored":
            if kernel:
                return reg.TransformationEstimationForColoredICP(lambda_geometric=0.968, kernel=kernel)
            return reg.TransformationEstimationForColoredICP(lambda_geometric=0.968)
        if method == "generalized":
            if kernel:
                return reg.TransformationEstimationForGeneralizedICP(epsilon=1e-3, kernel=kernel)
            return reg.TransformationEstimationForGeneralizedICP(epsilon=1e-3)
        raise ValueError(f"Unknown ICP method `{method}`")

    def _model_at_scale(self, obj: TrackedObject, scale: float, core: bool = False) -> o3d.geometry.PointCloud:
        """Return the model (or its confident core) downsampled for an ICP scale, cached until the model changes."""
        key = (scale, core)
        if key not in obj._pyramid:
            model = self._core(obj) if core else obj.model
            obj._pyramid[key] = model.voxel_down_sample(scale * obj.voxel) if scale > 1.0 else model
        return obj._pyramid[key]

    def _core(self, obj: TrackedObject) -> o3d.geometry.PointCloud:
        """Return the model points confirmed by enough views, or the whole model while the core is small."""
        keep = np.flatnonzero(obj.confidence >= self.config.core_min_confidence)
        if len(keep) < max(50, 0.25 * len(obj.model.points)):
            return obj.model
        return obj.model.select_by_index(keep)

    def _relocalize(
        self, source: o3d.geometry.PointCloud, obj: TrackedObject, core: bool = False, all_results: bool = False
    ) -> reg.RegistrationResult | list[reg.RegistrationResult] | None:
        """Globally register the observation to the model with FPFH features + RANSAC, then refine with ICP.

        RANSAC often lands near, but not inside, the true pose's basin, so several hypotheses are each refined
        with the wide coarse-to-fine ICP and the best fit is kept.
        """
        voxel = 2 * max(obj.voxel, obj.noise)
        src, src_fpfh = _fpfh(source, voxel)
        tgt, tgt_fpfh = _fpfh(self._core(obj) if core else obj.model, voxel)
        if len(src.points) < 10 or len(tgt.points) < 10:
            return [] if all_results else None
        max_dist = 1.5 * voxel
        best, results = None, []
        for _ in range(self.config.relocalize_trials):
            coarse = reg.registration_ransac_based_on_feature_matching(
                src,
                tgt,
                src_fpfh,
                tgt_fpfh,
                True,
                max_dist,
                reg.TransformationEstimationPointToPoint(False),
                3,
                [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(max_dist)],
                reg.RANSACConvergenceCriteria(100000, 0.999),
            )
            if coarse.fitness == 0.0:
                continue
            result = self._register(source, obj, coarse.transformation, wide=True, core=core)
            results.append(result)
            if best is None or (result.fitness, -result.inlier_rmse) > (best.fitness, -best.inlier_rmse):
                best = result
        return results if all_results else best

    def _reanchor(self, obj: TrackedObject, obs: _Observation) -> None:
        """Re-register the view against the confident core and snap to the closest well-fitting pose."""
        cfg = self.config
        local = self._register(obs.pcd, obj, np.linalg.inv(obj.obj_to_world), core=True)
        chosen = local
        global_candidates = self._relocalize(obs.pcd, obj, core=True, all_results=True)
        best_fitness = max((c.fitness for c in global_candidates), default=0.0)
        if best_fitness > local.fitness + cfg.reanchor_margin:
            max_dist = cfg.correspondence_factor * max(obj.voxel, obj.noise)

            def full_fitness(world_to_obj: np.ndarray) -> float:
                return reg.evaluate_registration(obs.pcd, obj.model, max_dist, world_to_obj).fitness

            good = [c for c in global_candidates if c.fitness >= best_fitness - cfg.reanchor_fitness_tolerance]
            if cfg.reanchor_jumps == "conservative":
                current = full_fitness(np.linalg.inv(obj.obj_to_world))
                floor = current - cfg.reanchor_full_model_tolerance
                good = [c for c in good if full_fitness(c.transformation) >= floor]

            def rotation_change(result: reg.RegistrationResult) -> float:
                delta = np.linalg.inv(result.transformation)[:3, :3] @ obj.obj_to_world[:3, :3].T
                return Rotation.from_matrix(delta).magnitude()

            chosen = min(good, key=rotation_change, default=local)
        if chosen.fitness < cfg.track_min_fitness:
            return
        obj_to_world = np.linalg.inv(chosen.transformation)
        if cfg.upright:
            obj_to_world = self._make_upright(obj, obj_to_world)
        if chosen is not local:
            obj._prev_obj_to_world = None  # the motion prior doesn't hold across a jump
        obj.obj_to_world = obj_to_world
        obj.fitness, obj.inlier_rmse = chosen.fitness, chosen.inlier_rmse

    def _is_keyframe(self, obj: TrackedObject, cam_to_world: np.ndarray) -> bool:
        """Whether the object has turned or moved enough relative to the camera to show new surface."""
        obj_to_cam = np.linalg.inv(cam_to_world) @ obj.obj_to_world
        if obj._keyframe_obj_to_cam is None:
            return True
        delta = np.linalg.inv(obj._keyframe_obj_to_cam) @ obj_to_cam
        angle = np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())
        shift = np.linalg.norm(obj_to_cam[:3, 3] - obj._keyframe_obj_to_cam[:3, 3])
        return angle >= self.config.keyframe_rotation_deg or shift >= self.config.keyframe_translation

    def _fuse(self, obj: TrackedObject, obs: _Observation, world_to_obj: np.ndarray) -> None:
        """Merge the registered view into the model, confirming the model points it re-observes.

        Each view point within a voxel of a model point confirms it (and is averaged into it); the rest are added
        as new, unconfirmed points. New points that no later view confirms within `confirm_within` fusions are
        pruned.
        """
        cfg = self.config
        obj.num_fusions += 1
        view = o3d.geometry.PointCloud(obs.pcd).transform(world_to_obj)
        q, q_colors, q_normals = (np.asarray(view.points), np.asarray(view.colors), np.asarray(view.normals))
        points, colors, normals = (
            np.asarray(obj.model.points).copy(),
            np.asarray(obj.model.colors).copy(),
            np.asarray(obj.model.normals).copy(),
        )
        dist, nearest = cKDTree(points).query(q, distance_upper_bound=max(obj.voxel, obj.noise))
        hit = np.isfinite(dist)

        # Running weighted average of the matched points, then one confirmation per view
        idx = nearest[hit]
        counts = np.bincount(idx, minlength=len(points)).astype(np.float64)
        weight = obj.confidence
        for values, new in ((points, q), (colors, q_colors), (normals, q_normals)):
            sums = np.zeros_like(values)
            np.add.at(sums, idx, new[hit])
            values[:] = (values * weight[:, None] + sums) / (weight + counts)[:, None]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        confidence = np.minimum(weight + (counts > 0), cfg.max_confidence)

        points = np.concatenate([points, q[~hit]])
        colors = np.concatenate([colors, q_colors[~hit]])
        normals = np.concatenate([normals, q_normals[~hit]])
        confidence = np.concatenate([confidence, np.ones((~hit).sum())])
        birth = np.concatenate([obj._birth, np.full((~hit).sum(), obj.num_fusions)])

        keep = (confidence >= 2) | (birth > obj.num_fusions - cfg.confirm_within)
        if keep.sum() > cfg.max_model_points:
            keep[np.argsort(-confidence)[cfg.max_model_points :]] = False
        model = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep]))
        model.colors = o3d.utility.Vector3dVector(colors[keep])
        model.normals = o3d.utility.Vector3dVector(normals[keep])
        confidence, birth = confidence[keep], birth[keep]
        if cfg.outlier_removal_every and obj.num_fusions % cfg.outlier_removal_every == 0:
            model, inliers = model.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            confidence, birth = confidence[inliers], birth[inliers]
        obj.model, obj.confidence, obj._birth = model, confidence, birth
        obj._pyramid = {}

    def _integrate_tsdf(
        self,
        obj: TrackedObject,
        rgb: np.ndarray,
        depth: np.ndarray,
        pixel_mask: np.ndarray,
        intrinsic_k: np.ndarray,
        cam_to_world: np.ndarray,
    ) -> None:
        """Integrate the masked RGB-D frame into the object's TSDF volume (in the object frame)."""
        h, w = depth.shape
        masked_depth = np.where(pixel_mask, depth, 0.0).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(masked_depth),
            depth_scale=1.0,
            depth_trunc=self.config.depth_max,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, intrinsic_k[0, 0], intrinsic_k[1, 1], intrinsic_k[0, 2], intrinsic_k[1, 2]
        )
        # The volume lives in the object frame, so its "extrinsic" maps object -> camera
        obj.tsdf.integrate(rgbd, intrinsic, np.linalg.inv(cam_to_world) @ obj.obj_to_world)


def _fpfh(pcd: o3d.geometry.PointCloud, voxel: float) -> tuple[o3d.geometry.PointCloud, reg.Feature]:
    """Downsample a cloud and compute its FPFH features."""
    down = pcd.voxel_down_sample(voxel)
    if not down.has_normals():
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
    fpfh = reg.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
    return down, fpfh


def _to_hwc_uint8(rgb: Any) -> np.ndarray:  # noqa: ANN401
    """Convert a (3, H, W) or (H, W, 3) numpy array or torch tensor to an (H, W, 3) uint8 array."""
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.transpose(1, 2, 0)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def flying_pixels(depth: np.ndarray, threshold: float) -> np.ndarray:
    """Flag valid pixels whose depth jumps by more than `threshold` to any valid 4-neighbour."""
    valid = depth > 0
    flagged = np.zeros_like(valid)
    for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
        neighbour = np.roll(depth, step, axis=axis)
        flagged |= valid & (neighbour > 0) & (np.abs(depth - neighbour) > threshold)
    return flagged


def shift_image(image: np.ndarray, du: int, dv: int) -> np.ndarray:
    """Shift an image right by `du` and down by `dv` pixels, filling uncovered pixels with zeros."""
    shifted = np.zeros_like(image)
    h, w = image.shape[:2]
    shifted[max(dv, 0) : h + min(dv, 0), max(du, 0) : w + min(du, 0)] = image[
        max(-dv, 0) : h + min(-dv, 0), max(-du, 0) : w + min(-du, 0)
    ]
    return shifted


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a 7D (x, y, z, qw, qx, qy, qz) pose, or a 4x4 matrix, to a 4x4 matrix."""
    if hasattr(pose, "detach"):
        pose = pose.detach().cpu().numpy()
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:7], scalar_first=True).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 matrix to a 7D (x, y, z, qw, qx, qy, qz) pose."""
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)
    return np.concatenate([matrix[:3, 3], quat])
