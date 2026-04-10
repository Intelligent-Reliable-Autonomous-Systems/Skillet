"""sam_reconstructor.py.

Reconstruct the scene from SAM bounding boxes.
"""

import pickle
import time
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.perception.reconstruction.utils import (
    assign_objects_to_id,
    find_cube_centers,
    transform_xyz_to_world,
)
from skillet.perception.segmentation.sam import get_sam_client
from skillet.perception.segmentation.vlm import GeminiClient
from skillet.scene import THREE_CUBE_APRIL_SCENE, Cube
from skillet.scene.base import Scene
from skillet.scene.utils import assign_poses_to_objects, get_sorted_object_poses


class SAMReconstructor(ReconstructorBase):
    """Main class for reconstruction with SAM Client.

    Finds the bounding boxes of the cubes, segments point cloud, and projects normal
    to find the center of the cube.

    """

    def __init__(
        self,
        scene: Scene | None = None,
        model: Literal["sam2", "sam3", "sam3_streaming"] = "sam3",
        mode: Literal["text", "bboxes"] = "text",
        device: str = "cuda",
        build_scene: bool = True,
        visualize: bool = True,
    ) -> None:
        super().__init__(scene)
        self._model = model
        self._mode = mode
        self._sam_model = get_sam_client(model)
        self._device = device
        self._build_scene_flag = build_scene
        self._vlm_client = GeminiClient() if build_scene else None
        self._visualize = visualize

        self._bbox_frame = None
        self._mask_frame = None
        self._masks = None
        self._segment_indices = None

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def masks(self) -> torch.Tensor:
        return self._masks

    @property
    def segment_indices(self) -> torch.Tensor:
        return self._segment_indices

    def update_state(
        self, obs: dict[str, Any], update: bool = True, frame: Literal["world", "camera"] = "camera"
    ) -> None:
        """Update the state of the scene by finding cube centers.

        Args:
            obs: RGB-D obs spec from the environment
            update: If to update the state of the scene or not

        """
        if not self._scene.contains_objects:
            print("[INFO][SAM RECONSTRUCTOR] Building scene...")
            self._build_scene(obs)
            print("[INFO][SAM RECONSTRUCTOR] Successfully built scene.")

        if not update:
            return
        rgb = obs["rgb"]
        depth = obs["depth"]
        intrinsic_k = obs["intrinsic_k"]
        camera_pose = obs["camera_pose"]

        if self._mode == "text":
            concepts = ["block", "robot_arm"]  # Can potentially segment on "cube face" for redundancy
            masks, boxes, scores, concept_indices = self._sam_model.segment_from_concepts(rgb, concepts)
        elif self._mode == "bboxes":
            bboxes = [
                [100, 100, 150, 150],
                [200, 200, 250, 250],
                [300, 300, 350, 350],
                [200, 100, 250, 150],
                [100, 200, 150, 250],
            ]
            masks, scores = self._sam_model.segment_from_bboxes(rgb, bboxes)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

        self._masks = masks
        self._segment_indices = torch.arange(masks.shape[0], device=masks.device)

        if isinstance(depth, torch.Tensor):
            rgb = rgb.cpu().numpy()
            depth = depth.cpu().numpy()
            intrinsic_k = intrinsic_k.cpu().numpy()
            camera_pose = camera_pose.cpu().numpy()

        # TODO make sure we are always grabbing the blocks
        cube_masks = masks[concept_indices == 0].cpu().numpy()
        masks = masks.cpu().numpy()

        # Find cube centers in the camera frame
        dc = find_cube_centers(
            cube_masks,
            depth,
            intrinsic_k,
            cube_size=0.041,
            camera_pos=camera_pose[0:3],
            camera_quat=camera_pose[3:7],
            frame=frame,
        )

        centers = (
            transform_xyz_to_world(dc["centers"], camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7])
            if frame == "camera"
            else dc["centers"]
        )

        poses, ids = get_sorted_object_poses(self._scene, Cube)
        cube_idx, det_idx = assign_objects_to_id(poses[:, 0:3], centers)

        assign_poses_to_objects(self._scene, Cube, centers, ids, cube_idx, det_idx)

        if self._visualize:
            self._bbox_frame = SAMReconstructor.show_bounding_boxes(
                rgb, masks, concept_indices=concept_indices, concepts=concepts
            )
            self._mask_frame = SAMReconstructor.show_cube_masks(rgb, cube_masks, self._scene, ids, cube_idx, det_idx)

    def get_observation(self) -> Scene:
        """Return the scene."""
        return self._scene

    def _build_scene(
        self,
        obs: dict[str, torch.Tensor],
        call_vlm: bool = False,
        vis_scene: bool = False,
        task_instruction: str = "Put the red block on the purple block.",
    ) -> None:
        """Build the scene using an API call to a VLM by creating bounding boxes for each object.

        Args:
            obs: RGBD obs spec observation
            call_vlm: If to call VLM or load scene from defaults
            vis_scene: If to visualize the scene after parsing
            task_instruction: string for the task instruction to seed the VLM with

        """
        rgb = obs["rgb"]
        depth = obs["depth"]
        camera_pose = obs["camera_pose"]
        intrinsic_k = obs["intrinsic_k"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
            depth = depth.cpu().numpy()
            camera_pose = camera_pose.cpu().numpy()
            intrinsic_k = intrinsic_k.cpu().numpy()
        if call_vlm:
            rgb_pil = Image.fromarray(rgb.transpose(1, 2, 0))
            rgb_pil_resized = rgb_pil.resize(
                (800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS
            )
            bboxes, grounded_goal_atoms, _ = self._vlm_client.detect_and_translate(rgb_pil_resized, task_instruction)

            for bbox in bboxes:
                bbox["label"] = bbox["label"].replace(" ", "_")
            for atom in grounded_goal_atoms:
                atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]
            with open("data/test/vlm_out_2.pkl", "wb") as f:
                pickle.dump(bboxes, f)
        with open("data/test/vlm_out.pkl", "rb") as f:
            out = pickle.load(f)
        labels = []
        boxes = []
        # Parse boxes + labels from VLM
        for d in out:
            if "block" in d["label"]:
                labels.append(d["label"])
                # BBoxes in format  [ymin, xmin, ymax, xmax]
                box = d["box_2d"]
                box[0] = (box[0] / 1000) * rgb.shape[1]
                box[2] = (box[2] / 1000) * rgb.shape[1]
                box[1] = (box[1] / 1000) * rgb.shape[2]
                box[3] = (box[3] / 1000) * rgb.shape[2]
                boxes.append(box)

        # Find cube centers from SAM3 with bounding boxes
        masks, scores = self._sam_model.segment_from_bboxes(rgb, np.asarray(boxes))

        if vis_scene:
            SAMReconstructor.show_image_and_masks(rgb.transpose(1, 2, 0), masks.cpu().numpy(), labels)
        dc = find_cube_centers(
            masks.cpu().numpy(),
            depth,
            intrinsic_k,
            cube_size=0.041,
            camera_pos=camera_pose[0:3],
            camera_quat=camera_pose[3:7],
        )
        centers = transform_xyz_to_world(dc["centers"], camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7])

        # Reconstruct scene
        cubes = []
        for i, l in enumerate(labels):
            c = Cube(
                size=0.036, init_pose=torch.as_tensor(np.concatenate((centers[i], [1, 0, 0, 0])), device=self._device)
            )
            c.name = l
            cubes.append(c)
        self._scene.add_objects(cubes)
        self._scene.contains_objects = True
        print(f"[INFO] Reconstructed Scene with VLM.\n{self._scene}")

    @staticmethod
    def show_image_and_masks(
        image: np.ndarray,
        masks: np.ndarray,
        labels: list[str],
    ) -> None:
        """Show RGB image and a single overlay image with all masks + labels.

        Args:
            image: (H, W, 3) RGB image, uint8 or float
            masks: (N, H, W) boolean or {0,1} masks
            labels: list of length N containing labels for each mask

        """
        num_masks = masks.shape[0]
        if len(labels) != num_masks:
            raise ValueError(f"Expected {num_masks} labels, got {len(labels)}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(image)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")

        overlay = image.copy().astype(float)

        # Generate distinct colors
        rng = np.random.default_rng(0)
        colors = rng.integers(0, 255, size=(num_masks, 3))

        alpha = 0.7
        for i in range(num_masks):
            mask = masks[i].astype(bool)
            if mask.sum() == 0:
                continue

            color = colors[i].astype(float)
            overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

            # Compute centroid for placing label
            ys, xs = np.where(mask)
            cy = int(np.mean(ys))
            cx = int(np.mean(xs))
            cy = max(cy - 10, 0)

            axes[1].text(
                cx,
                cy,
                labels[i],
                color="white",
                fontsize=10,
                ha="center",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
            )

        if image.dtype == np.uint8:
            overlay = overlay.astype(np.uint8)
        else:
            overlay = overlay / 255.0 if overlay.max() > 1 else overlay

        axes[1].imshow(overlay)
        axes[1].set_title("Masks + Labels")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_cube_detection(
        results: dict[str, Any],
        masks: np.ndarray,
        depth: np.ndarray,
        camera_matrix: np.ndarray,
        camera_pose: np.ndarray,
        frame: Literal["world", "camera"] = "world",
        depth_scale: float = 1.0,
        max_points: int = 500,
    ) -> None:
        """Visualize cube detection results with live streaming at ~2Hz.

        Args:
            results:        Output dict from find_cube_centers().
            masks:          Binary masks (N, H, W).
            depth:          Depth map (1, H, W) or (H, W).
            camera_matrix:  3x3 intrinsics.
            camera_pose:    Pose of camera in xyz wxyz (quat) in world frame
            frame:          Frame to visualize in (world or camera)
            depth_scale:    Same scale used in find_cube_centers().
            max_points:     Max depth cloud points to render (downsampled for speed).

        """
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        COLORS = plt.cm.tab10.colors

        def deproject(mask: np.ndarray, depth_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Back-project masked pixels into 3D camera-frame points."""
            ys, xs = np.where(mask & (depth_img > 0))
            z = depth_img[ys, xs] * depth_scale
            x = (xs - cx) * z / fx
            y = (ys - cy) * z / fy
            all_pts = np.stack([x, y, z], axis=1)

            # Downsample for rendering speed
            if len(all_pts) > max_points:
                idx = np.random.choice(len(all_pts), max_points, replace=False)
                pts = all_pts[idx]
            else:
                pts = all_pts
            return pts, all_pts

        def plane_patch(
            center: np.ndarray,
            normal: np.ndarray,
            pts_3d: np.ndarray,
            plane_eq: np.ndarray,
            padding: float = 0.000,
            threshold: float = 0.004,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Build a plane patch bounded by the min/max x and y of the point cloud."""
            # Build local (u, v) frame on the plane
            ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(normal, ref)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            v /= np.linalg.norm(v)

            # Compute plane distances and mask out far
            a, b, c, d = plane_eq
            distances = a * pts_3d[:, 0] + b * pts_3d[:, 1] + c * pts_3d[:, 2] + d

            # Project along normal
            distances = distances.reshape(-1, 1)
            mask = (distances < threshold).flatten()

            # Project points into 2D (u, v) coords
            origin = center
            coords_u = (pts_3d[mask] - origin) @ u
            coords_v = (pts_3d[mask] - origin) @ v

            # Bounding box with optional padding
            u_min, u_max = coords_u.min() - padding, coords_u.max() + padding
            v_min, v_max = coords_v.min() - padding, coords_v.max() + padding

            # Four corners lifted back to 3D
            uu, vv = np.meshgrid(
                np.linspace(u_min, u_max, 10),
                np.linspace(v_min, v_max, 10),
            )
            pts = origin[:, None, None] + u[:, None, None] * uu + v[:, None, None] * vv
            return pts[0], pts[1], pts[2]

        # Figure setup
        fig = plt.figure(figsize=(10, 7), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333")
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.set_xlabel("X (m)", color="#888", fontsize=8)
        ax.set_ylabel("Y (m)", color="#888", fontsize=8)
        ax.set_zlabel("Z (m)", color="#888", fontsize=8)
        ax.set_title("", color="white", fontsize=10, pad=8)

        def draw(res: dict[str, np.ndarray], msk: np.ndarray, dep: np.ndarray) -> None:
            """Draw the result of RANSAC with cube centers."""
            ax.cla()
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")
            ax.tick_params(colors="#888", labelsize=7)
            ax.set_xlabel("X (m)", color="#888", fontsize=8)
            ax.set_ylabel("Y (m)", color="#888", fontsize=8)
            ax.set_zlabel("Z (m)", color="#888", fontsize=8)

            d2d = dep.squeeze() if dep.ndim == 3 else dep
            legend_handles = []

            for i, (mask, center, normal, plane_eq, plane_center) in enumerate(
                zip(msk, res["centers"], res["normals"], res["plane_equations"], res["plane_centers"])
            ):
                color = COLORS[i % len(COLORS)]
                pts, all_pts = deproject(mask.astype(bool), d2d)

                if len(pts) == 0:
                    continue
                if frame == "world":
                    pts = transform_xyz_to_world(pts, camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7])
                # Depth cloud with cube outline
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    s=3,
                    c="black",
                    depthshade=False,
                    zorder=1,
                )
                # Faint colored halo so you can tell cubes apart
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    s=10,
                    c=[color],
                    alpha=0.15,
                    depthshade=False,
                    zorder=1,
                )

                # Fitted Plane

                Xp, Yp, Zp = plane_patch(plane_center, normal, all_pts, plane_eq)
                surf = ax.plot_surface(
                    Xp,
                    Yp,
                    Zp,
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                    antialiased=True,
                    zorder=2,
                )

                # Normal vector
                ax.quiver(
                    plane_center[0],
                    plane_center[1],
                    plane_center[2],
                    normal[0],
                    normal[1],
                    normal[2],
                    length=0.035,
                    normalize=False,
                    color=color,
                    linewidth=2,
                    arrow_length_ratio=0.35,
                    zorder=4,
                )
                # Label "n̂"
                tip = plane_center + normal * 0.038
                ax.text(
                    tip[0],
                    tip[1],
                    tip[2],
                    "n̂",
                    color=color,
                    fontsize=8,
                    zorder=5,
                )

                # Cube center: sphere with cross
                ax.scatter(
                    center[0],
                    center[1],
                    center[2],
                    s=120,
                    c=[color],
                    edgecolors="black",
                    linewidths=1.5,
                    depthshade=False,
                    zorder=6,
                    marker="o",
                )
                # tiny cross
                d = 0.008
                for dx, dy, dz in [(d, 0, 0), (0, d, 0), (0, 0, d)]:
                    ax.plot(
                        [center[0] - dx, center[0] + dx],
                        [center[1] - dy, center[1] + dy],
                        [center[2] - dz, center[2] + dz],
                        color="black",
                        linewidth=1,
                        alpha=0.8,
                        zorder=7,
                    )
                ax.text(
                    center[0],
                    center[1],
                    center[2] + d * 2,
                    f"C{i}",
                    color="black",
                    fontsize=7,
                    zorder=8,
                )

                lbl = f"Cube {i}  z={center[2]:.3f}m"
                legend_handles.append(mpatches.Patch(color=color, label=lbl))

            ax.set_title(
                "Cube Plane Detection",
                color="black",
                fontsize=10,
                pad=8,
            )
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    loc="upper left",
                    fontsize=7,
                    facecolor="#222",
                    edgecolor="#444",
                    labelcolor="black",
                )

            ax.grid(False)
            fig.canvas.draw_idle()

        draw(results, masks, depth)
        plt.tight_layout()
        plt.show()
        return None

    @staticmethod
    def masks_to_bboxes(masks: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Convert binary masks of shape (N, H, W) to bounding boxes (x1, y1, x2, y2)."""
        bboxes = []
        for mask in masks:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        return bboxes

    @staticmethod
    def show_cube_masks(
        rgb_image: np.ndarray,
        masks: np.ndarray,
        scene: Scene,
        ids: np.ndarray,
        obj_idx: np.ndarray,
        det_idx: np.ndarray,
    ) -> np.ndarray:
        """Show the masks and the corresponding labels.

        Args:
            rgb_image: RGB image from camera
            masks: masks produced by SAM
            scene: the current scene to obtain
            ids: np.ndarray of sorted object ids
            obj_idx: Sorted indexes of object scene ids according to poses
            det_idx: The detection index of which pose to assign to which object

        """
        rgb_image = rgb_image.transpose((1, 2, 0))
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()

        # Generate distinct colors for each object
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(100, 255, size=(len(scene.objects), 3))]

        for color_idx, ob in enumerate(scene.objects):
            if not isinstance(ob, Cube):
                continue

            idx = np.where(ob.object_id == ids[obj_idx])[0]
            if idx.size > 0:
                idx = idx[0]
            else:
                continue

            mask = masks[det_idx[idx]]  # shape (H, W), bool or 0/1
            color = colors[color_idx]

            # Overlay colored mask with transparency
            overlay = display.copy()
            overlay[mask.astype(bool)] = color
            cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

            # Draw contour around mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color, 2)

            # Place label at centroid of mask
            padding = 3
            baseline = 3
            ys, xs = np.where(mask.astype(bool))
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(ob.name, font, font_scale, thickness)
                tx, ty = cx - text_w // 2, int(ys.min()) - padding
                tx = max(0, min(tx, display.shape[1] - text_w))
                ty = max(text_h + padding, ty)

                cv2.putText(display, ob.name, (tx, ty), font, font_scale, (255, 255, 255), thickness)

        return display

    @staticmethod
    def show_bounding_boxes(
        rgb_image: np.ndarray,
        masks: np.ndarray,
        concept_indices: np.ndarray | None = None,
        concepts: list | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes and semi-transparent mask overlays on an RGB image.

        Args:
            rgb_image: HxWx3 numpy array in RGB format.
            masks:     NxHxW binary (or boolean) numpy array from SAM.
            concept_indices: mask indices of each concept
            concepts: list of concepts from segmentation

        Returns:
            BGR image ready for cv2.imshow.

        """
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.40
        THICKNESS = 1
        PADDING = 4
        # cv2 works in BGR
        rgb_image = rgb_image.transpose((1, 2, 0))
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()
        overlay = display.copy()

        # Generate a distinct colour per mask
        rng = np.random.default_rng(seed=42)
        colors = [tuple(int(c) for c in rng.integers(80, 230, size=3)) for _ in range(len(masks))]

        for i, mask in enumerate(masks):
            # Semi-transparent fill
            overlay[mask > 0] = colors[i]

            # Bounding box
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(display, (x1, y1), (x2, y2), colors[i], thickness=2)

            # Concept label
            label = f"{concepts[concept_indices[i]]} {i}" if concept_indices is not None else f"Object {i}"

            (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)

            pill_x1 = x1
            pill_y1 = max(0, y1 - text_h - baseline - PADDING * 2)
            pill_x2 = x1 + text_w + PADDING * 2
            pill_y2 = max(text_h + baseline + PADDING * 2, y1)

            cv2.rectangle(display, (pill_x1, pill_y1), (pill_x2, pill_y2), colors[i], cv2.FILLED)

            text_x = pill_x1 + PADDING
            text_y = pill_y2 - PADDING - baseline
            cv2.putText(display, label, (text_x, text_y), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)

        # Blend fill at 35 % opacity
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
        return display


def main() -> None:
    """Run live SAM benchmark with RGB/depth view and mask overlay."""
    scene = THREE_CUBE_APRIL_SCENE

    env = RealsenseEnv()
    reconstructor = SAMReconstructor(scene=None, build_scene=True)
    while True:
        rgbd_obs = env.get_observation()
        reconstructor.update_state(rgbd_obs)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
