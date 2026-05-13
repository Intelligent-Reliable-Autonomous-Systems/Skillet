"""data_collector.py.

Controls logging of data and messages within Skillet.
"""

import copy
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from unified_planning.plans import ActionInstance

from skillet.agents.base_agent import Agent
from skillet.core import ObservationSpec
from skillet.envs import SkilletEnv
from skillet.perception.perception import SkilletPerception
from skillet.planning import AbstractModel
from skillet.planning.abstract.trace_io import PDDLTraceIO
from skillet.planning.abstract.up_utils import AbstractAction
from skillet.scene.base import Scene
from skillet.scene.utils import depth_to_colormap_np


class SkilletDataLogger:
    def __init__(
        self,
        log_dir: str,
        env: SkilletEnv,
        scene: Scene,
        perception: SkilletPerception | None = None,
        abs_model: AbstractModel | None = None,
        agent: Agent | None = None,
        obs_spec: ObservationSpec | None = None,
        visualize: bool = False,
    ):
        self._log_dir = log_dir
        self._scene = scene
        self._env = env
        self._perception = perception
        self._abs_model = abs_model
        self._agent = agent
        self._obs_spec = obs_spec

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._num_points = 0
        self._exp_id = -1
        self._start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pddl_trace = PDDLTraceIO(self._abs_model._problem)

        # For display
        self._write_video = False
        self._fps = 15
        self.figsize = (20, 11)
        self._writer = None
        self._data_window_name = "Skillet Visualization"
        self._window_active = False
        self._visualize = visualize
        self._width = 1920
        self._height = 1080
        matplotlib.use("Agg")

        self.fig = plt.figure(figsize=self.figsize, facecolor="#aaaaaa")
        self._build_layout()

        self._cap = cv2.VideoCapture(0)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # .mp4 output
        self._writer = None

        self.reset_logging()
        Path(self._log_dir).mkdir(exist_ok=True, parents=True)

    @property
    def write_video(self) -> bool:
        return self._write_video

    @write_video.setter
    def write_video(self, write: bool) -> None:
        self._write_video = write

    def save_video(self):
        if self._write_video:
            self._cap.release()
            self._writer.release()
            self._write_video = False
            self._writer = None

    def setup_logging(self) -> None:
        """Set up logging."""
        pass

    def reset_logging(self, log_dir: str | None = None) -> None:
        """Reset the data collector by dumping everything to the log file and resetting buffers."""
        self._log_dir = log_dir if log_dir is not None else self._log_dir
        self._num_points = 0
        self._exp_id += 1
        self._rgbd_obs: np.ndarray = None
        self._depth_obs: np.ndarray = None
        self._camera_pose: np.ndarray = None
        self._tcp_pose: np.ndarray = None
        self._gripper_pos: np.ndarray = None
        self._time_stamps: np.ndarray = None
        self._abs_state: np.ndarray = None
        self._obj_poses: np.ndarray = None
        self._obj_ids: np.ndarray = None
        self._obj_names: np.ndarray = None
        self._intrinsic_k: np.ndarray = None
        self._world_bounds: np.ndarray = None

    def add_datapoint(self) -> None:
        """Add a datapoint to the logger by querying the environment for relevant observations."""
        print("[INFO][LOGGER] Logging datapoint.")
        time_stamp = time.perf_counter()
        obs = self._env.get_observation(self._obs_spec)
        scene_dict = self._scene.serialize_scene_poses()
        # abstract_state = self._abs_model.get_abstract_state()
        if self._time_stamps is None:
            self._time_stamps = np.array([time_stamp])
            self._rgbd_obs = obs["rgb"].cpu().numpy().squeeze()[None, ...]
            self._depth_obs = obs["depth"].cpu().numpy().squeeze()[None, ...]
            self._camera_pose = obs["camera_pose"].cpu().numpy().squeeze()[None, ...]
            self._intrinsic_k = obs["intrinsic_k"].cpu().numpy().squeeze()[None, ...]
            self._tcp_pose = obs["tcp_pose_b"].cpu().numpy().squeeze()[None, ...]
            self._gripper_pos = obs["gripper"].cpu().numpy().squeeze()[None, ...]
            self._perception_frame = self._perception.perception_frame[None, ...]
            # self._abs_state = np.array([abstract_state])
            self._obj_ids = scene_dict["ids"][None, ...]
            self._obj_poses = scene_dict["poses"][None, ...]
            self._obj_names = scene_dict["names"][None, ...]
            self._world_bounds = scene_dict["bounds"][None, ...]
        else:
            self._time_stamps = np.concatenate((self._time_stamps, np.array([time_stamp])), axis=0)
            self._rgbd_obs = np.concatenate((self._rgbd_obs, obs["rgb"].cpu().numpy().squeeze()[None, ...]), axis=0)
            self._depth_obs = np.concatenate((self._depth_obs, obs["depth"].cpu().numpy().squeeze()[None, ...]), axis=0)
            self._camera_pose = np.concatenate(
                (self._camera_pose, obs["camera_pose"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._intrinsic_k = np.concatenate(
                (self._intrinsic_k, obs["intrinsic_k"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._tcp_pose = np.concatenate(
                (self._tcp_pose, obs["tcp_pose_b"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._gripper_pos = np.concatenate(
                (self._gripper_pos, obs["gripper"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._perception_frame = np.concatenate(
                (self._perception_frame, self._perception.perception_frame[None, ...]), axis=0
            )
            # self._abs_state = np.concatenate((self._abs_state, np.array([abstract_state])), axis=-1)
            self._obj_ids = np.concatenate((self._obj_ids, scene_dict["ids"][None, ...]), axis=0)
            self._obj_poses = np.concatenate((self._obj_poses, scene_dict["poses"][None, ...]), axis=0)
            self._obj_names = np.concatenate((self._obj_names, scene_dict["names"][None, ...]), axis=0)
            self._world_bounds = np.concatenate((self._world_bounds, scene_dict["bounds"][None, ...]), axis=0)

        self._num_points += 1

    def save_log(self) -> None:
        """Save the log to a file."""
        print("[INFO][LOGGER] Saving datafile")
        fpath = Path(f"{self._log_dir}/{self._start_time}/exp_{self._exp_id}")
        fpath.mkdir(exist_ok=True, parents=True)
        if self._write_video:
            self._cap.release()
            self._writer.release()
        with h5py.File(f"{fpath}/data.h5", "w") as f:
            ep = f.create_group("episode")
            ep.create_dataset("time_stamps", data=self._time_stamps, compression="gzip")
            ep.create_dataset("rgb", data=self._rgbd_obs, compression="gzip")
            ep.create_dataset("depth", data=self._depth_obs, compression="gzip")
            ep.create_dataset("camera_pose", data=self._camera_pose, compression="gzip")
            ep.create_dataset("intrinsic_k", data=self._intrinsic_k, compression="gzip")
            ep.create_dataset("tcp_pose", data=self._tcp_pose, compression="gzip")
            ep.create_dataset("perception_frame", data=self._perception_frame, compression="gzip")
            # ep.create_dataset("abs_state", data=self._abs_state, compression="gzip")
            ep.create_dataset("obj_ids", data=self._obj_ids, compression="gzip")
            ep.create_dataset("obj_poses", data=self._obj_poses, compression="gzip")
            ep.create_dataset(
                "obj_names",
                data=self._obj_names.astype(object),
                compression="gzip",
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            ep.create_dataset("world_bounds", data=self._world_bounds, compression="gzip")
        f.close()

    def _build_layout(self) -> None:
        """Build the layout of the grid for plotting."""
        # Outer: 3 columns — [main rgb | 2x2 grid | text column]
        outer = gridspec.GridSpec(
            1, 3, figure=self.fig, width_ratios=[2, 2, 1], wspace=0.05, left=0.02, right=0.98, top=0.95, bottom=0.02
        )

        # Left: main RGB
        left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[0.05, 1], hspace=0.08)
        self.ax_main = self.fig.add_subplot(left_gs[1])
        self.ax_main_title = self.fig.add_subplot(left_gs[0])
        self.ax_main_title.axis("off")
        """self.ax_main_title.text(
            0.5,
            0.5,
            "Main Image",
            ha="center",
            va="center",
            color="black",
            fontsize=20,
            transform=self.ax_main_title.transAxes,
        )"""

        # Middle: 2x2 grid of images
        mid_gs = gridspec.GridSpecFromSubplotSpec(
            3,
            2,
            subplot_spec=outer[1],
            height_ratios=[0.05, 1, 1],
            hspace=0.08,
            wspace=0.05,
        )
        self.ax_grid_title = self.fig.add_subplot(mid_gs[0, :])
        self.ax_grid_title.axis("off")
        """self.ax_grid_title.text(
            0.5,
            0.5,
            "Perception",
            ha="center",
            va="center",
            color="black",
            fontsize=20,
            transform=self.ax_grid_title.transAxes,
        )"""
        self.ax_grid = [
            self.fig.add_subplot(mid_gs[1, 0]),
            self.fig.add_subplot(mid_gs[1, 1]),
            self.fig.add_subplot(mid_gs[2, 0]),
            self.fig.add_subplot(mid_gs[2, 1]),
        ]

        # Right: inital scene (from VLM), plan text (middle) + action text (bottom)
        right_gs = gridspec.GridSpecFromSubplotSpec(
            5, 1, subplot_spec=outer[2], height_ratios=[1, 1, 0.5, 1, 0.5], hspace=0.08
        )
        self.ax_vlm = self.fig.add_subplot(right_gs[0])
        self.ax_text = [
            self.fig.add_subplot(right_gs[1]),
            self.fig.add_subplot(right_gs[2]),
            self.fig.add_subplot(right_gs[3]),
            self.fig.add_subplot(right_gs[4]),
        ]

        # Style all axes
        for ax in [self.ax_main, self.ax_vlm, *self.ax_text, *self.ax_grid]:
            ax.axis("off")
            ax.set_facecolor("#aaaaaa")

    def _show_image(self, ax: Axes, img_bgr: np.ndarray, title: str | None = None) -> None:
        """Show the image on the corresponding axis.

        Args:
            ax: matplotlib.axes.Axes object to show image on
            img_bgr: np.ndarray in shape (3, h, w)

        """
        ax.clear()
        ax.axis("off")
        ax.imshow(img_bgr)
        if title:
            ax.set_title(title, color="black", fontsize=14, pad=3)

    def _show_text_box(self, ax: Axes, title: str, body: str, title_color: str = "#000000") -> None:
        """Show a text box with title and formatted lines.

        Args:
            ax: matplotlib.axes.Axes object to show text on
            title: title of text box
            body: body of text to show
            title_color: color of the title

        """
        ax.clear()
        ax.axis("off")
        ax.set_facecolor("#aaaaaa")
        ax.text(0.05, 0.97, title, transform=ax.transAxes, color=title_color, fontsize=14, fontweight="bold", va="top")
        ax.text(
            0.05,
            0.82,
            body,
            transform=ax.transAxes,
            color="black",
            fontsize=10,
            va="top",
            wrap=True,
            multialignment="left",
        )

    def update(
        self,
        main_rgb: np.ndarray,  # (H, W, 3) BGR
        perception_images: list[np.ndarray],  # list of 4 BGR images
        scene_img: np.ndarray,
        abstract_texts: list[str] = [],
        task: str = "Block Stack Experiment",
    ):
        """Update the image to display.

        Args:
            main_rgb: np.ndarray for principle rgb image in shape (H,W,3)
            perception_images: list of np.ndarray of images from perception pipeline in shape (H,W,3).
            scene_img: np.ndarray image for the initial scene
            abstract_texts: list of printable text strings from abstract state

        """
        self._show_image(self.ax_main, main_rgb, title="RGB")

        grid_titles = ["Depth", "SAM3 BBoxes", "SAM3 Masks", "3D Scene"]
        for ax, img, title in zip(self.ax_grid, perception_images, grid_titles, strict=False):
            self._show_image(ax, img, title=title)

        # self._show_image(self.ax_vlm, scene_img, title="VLM Reconstruction")

        text_titles = ["Current State", "Goal", "Plan", "Current Action"]
        for ax, text, title in zip(self.ax_text, abstract_texts, text_titles, strict=False):
            self._show_text_box(ax, title, text, title_color="#000000")

        self.fig.suptitle(f"Task: {task}", color="black", fontsize=22)

        # Render to numpy array
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype="uint8")
        w, h = self.fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Init writer on first frame

        return frame_bgr

    def close(self):
        if self._writer:
            self._writer.release()
        plt.close(self.fig)

    def run(self) -> None:
        """Run the data logging loop."""
        poll_period_s = 1.0 / self._fps
        next_poll_t = time.perf_counter()
        while not self._stop_event.is_set():
            self.update_visualization()

            next_poll_t += poll_period_s
            sleep_s = max(0.0, next_poll_t - time.perf_counter())
            if sleep_s > 0:
                time.sleep(sleep_s)

        self.stop()

    def update_visualization(self) -> None:
        """Update the logging visualization."""
        obs = self._env.get_observation(self._obs_spec)
        rgb = obs["rgb"].cpu().numpy().squeeze().transpose((1, 2, 0)).astype("uint8")
        depth = depth_to_colormap_np(obs["depth"].cpu().numpy().squeeze())
        h, w, c = rgb.shape
        black_arr = np.zeros((h, w, c), dtype="uint8")
        bboxes, masks, o3d, vlm_img = black_arr, black_arr, black_arr, black_arr
        task = "Block Stack Experiment"
        if self._perception is not None:
            bboxes = self._perception.bbox_frame if self._perception.bbox_frame is not None else black_arr
            masks = self._perception.mask_frame if self._perception.mask_frame is not None else black_arr
            o3d = (
                cv2.resize(self._perception.open3d_scene, (w, h))
                if (self._perception.open3d_scene is not None if self._perception.open3d_scene is not None else None)
                else black_arr
            )
            vlm_img = self._perception.vlm_frame if self._perception.vlm_frame is not None else black_arr
            task = (
                self._perception.task_instruction
                if self._perception.task_instruction is not None
                else "Block Stack Experiment"
            )

        ag_plan = (str(self._agent.plan) if self._agent is not None else None) or " "
        init_state = (str(self._abs_model.init_state) if self._abs_model is not None else None) or " "
        selected_skill = (str(self._agent.selected_skill) if self._agent is not None else None) or " "
        goal = (str(self._abs_model.goal) if self._abs_model is not None else None) or " "
        frame = self.update(
            rgb, [depth, bboxes, masks, o3d], vlm_img, [init_state, goal, ag_plan, selected_skill], task
        )

        fh, fw, _ = frame.shape
        if self._write_video:
            if self._writer is None:
                fpath = Path(f"{self._log_dir}/{self._start_time}/exp_{self._exp_id}")
                fpath.mkdir(exist_ok=True, parents=True)
                self._writer = cv2.VideoWriter(
                    f"{self._log_dir}/{self._start_time}/exp_{self._exp_id}/output_{datetime.now().strftime('%H%M%S')}.mp4",
                    self._fourcc,
                    self._fps,
                    (fw, fh),
                )
            self._writer.write(frame)
            if self._visualize:
                self._ensure_window()
                cv2.imshow(self._data_window_name, frame)
                cv2.waitKey(1)

    def run_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="SkilletDataLoggingThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for the worker thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _ensure_window(self) -> None:
        if self._window_active:
            return
        cv2.namedWindow(self._data_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._data_window_name, self._width, self._height)
        self._window_active = True

    def log(self, log_dir: str | None = None, save_log: bool = False, **kwargs) -> None:
        """Save data."""
        self._log_dir = log_dir if log_dir is not None else self._log_dir
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy().squeeze()
            elif isinstance(v, np.ndarray):
                v = v.squeeze()

            if not hasattr(self, f"_{k}"):
                if isinstance(v, np.ndarray):
                    setattr(self, f"_{k}", v[None, ...])
                elif isinstance(v, Scene):
                    setattr(self, f"_{k}", [copy.deepcopy(v)])
                elif isinstance(v, (AbstractAction, ActionInstance, dict)):
                    setattr(self, f"_{k}", [v])
                else:
                    setattr(self, f"_{k}", [v])

            else:
                if isinstance(v, np.ndarray):
                    new_v = np.concatenate((getattr(self, f"_{k}"), v[None, ...]))
                    setattr(self, f"_{k}", new_v)
                elif isinstance(v, Scene):
                    getattr(self, f"_{k}").append(copy.deepcopy(v))
                elif isinstance(v, (AbstractAction, ActionInstance, dict)):
                    getattr(self, f"_{k}").append(v)
                else:
                    getattr(self, f"_{k}").append(v)
        if save_log:
            print("[INFO][LOGGER] Saving datafile")
            fpath = Path(f"{self._log_dir}/{self._start_time}/exp_{self._exp_id}")
            fpath.mkdir(exist_ok=True, parents=True)
            for k in kwargs:
                v = getattr(self, f"_{k}")
                if isinstance(v, np.ndarray):
                    np.save(f"{fpath}/_{k}.npy", v)
                if isinstance(v, list):
                    with open(f"{fpath}/_{k}.pkl", "wb") as f:
                        pickle.dump(v, f)
            if hasattr(self, "_states") and hasattr(self, "_actions") and hasattr(self, "_executions"):
                self._pddl_trace.write_trace_file(
                    f"{fpath}/_pddl_trace.pddl", self._states, self._actions, self._executions
                )
