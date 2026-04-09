"""data_collector.py

Controls logging of data and messages within Skillet.
"""

from datetime import datetime
import time
from skillet.scene.base import Scene
import h5py
import numpy as np
import cv2

from skillet.envs import SkilletEnv
from skillet.perception.perception import SkilletPerception
from skillet.scene.abstract.abstract_model import AbstractModel
from pathlib import Path


class SkilletDataLogger:
    def __init__(self, log_dir: str, env: SkilletEnv, perception: SkilletPerception, abs_model: AbstractModel):
        self._log_dir = log_dir
        self._env = env
        self._perception = perception
        self._abs_model = abs_model

        self._num_points = 0
        self._exp_id = -1
        self._start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._write_video = False

        if self._write_video:
            self._cap = cv2.VideoCapture(0)
            self.fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # .mp4 output
            self._writer = None

        self.reset_logging()
        Path(self._log_dir).mkdir(exist_ok=True, parents=True)

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
        twist_obs = self._env.get_observation(self._env.unwrapped.obs_spec_twist_tcp)
        rgbd_obs = self._env.get_observation(self._env.unwrapped.obs_spec_rgbd)
        scene_obs: Scene = self._perception.scene
        scene_dict = scene_obs.serialize_scene_poses()
        # abstract_state = self._abs_model.get_abstract_state()
        if self._time_stamps is None:
            self._time_stamps = np.array([time_stamp])
            self._rgbd_obs = rgbd_obs["rgb"].cpu().numpy().squeeze()[None, ...]
            self._depth_obs = rgbd_obs["depth"].cpu().numpy().squeeze()[None, ...]
            self._camera_pose = rgbd_obs["camera_pose"].cpu().numpy().squeeze()[None, ...]
            self._intrinsic_k = rgbd_obs["intrinsic_k"].cpu().numpy().squeeze()[None, ...]
            self._tcp_pose = twist_obs["tcp_pose_b"].cpu().numpy().squeeze()[None, ...]
            self._perception_frame = self._perception.perception_frame[None, ...]
            # self._abs_state = np.array([abstract_state])
            self._obj_ids = scene_dict["ids"][None, ...]
            self._obj_poses = scene_dict["poses"][None, ...]
            self._obj_names = scene_dict["names"][None, ...]
            self._world_bounds = scene_dict["bounds"][None, ...]
        else:
            self._time_stamps = np.concatenate((self._time_stamps, np.array([time_stamp])), axis=0)
            self._rgbd_obs = np.concatenate(
                (self._rgbd_obs, rgbd_obs["rgb"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._depth_obs = np.concatenate(
                (self._depth_obs, rgbd_obs["depth"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._camera_pose = np.concatenate(
                (self._camera_pose, rgbd_obs["camera_pose"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._intrinsic_k = np.concatenate(
                (self._intrinsic_k, rgbd_obs["intrinsic_k"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._tcp_pose = np.concatenate(
                (self._tcp_pose, twist_obs["tcp_pose_b"].cpu().numpy().squeeze()[None, ...]), axis=0
            )
            self._perception_frame = np.concatenate(
                (self._perception_frame, self._perception.perception_frame[None, ...]), axis=0
            )
            # self._abs_state = np.concatenate((self._abs_state, np.array([abstract_state])), axis=-1)
            self._obj_ids = np.concatenate((self._obj_ids, scene_dict["ids"][None, ...]), axis=0)
            self._obj_poses = np.concatenate((self._obj_poses, scene_dict["poses"][None, ...]), axis=0)
            self._obj_names = np.concatenate((self._obj_names, scene_dict["names"][None, ...]), axis=0)
            self._world_bounds = np.concatenate((self._world_bounds, scene_dict["bounds"][None, ...]), axis=0)

        if self._write_video:
            if self._writer is None:
                h, w = self._perception_frame.shape[1:3]
                self._writer = cv2.VideoWriter("output.mp4", self._fourcc, self.fps, (w, h))
            self._writer.write(self._perception.perception_frame)
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
