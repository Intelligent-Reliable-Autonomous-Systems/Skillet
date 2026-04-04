"""data_collector.py

Controls logging of data and messages within Skillet.
"""

from datetime import datetime
import time
from skillet.scene.base import Scene
import h5py
import numpy as np

from skillet.envs import SkilletEnv
from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.scene.abstract.abstract_model import AbstractModel
from pathlib import Path


class SkilletDataLogger:
    def __init__(self, log_dir: str, env: SkilletEnv, reconstructor: ReconstructorBase, abs_model: AbstractModel):
        self._log_dir = log_dir
        self._env = env
        self._reconstructor = reconstructor
        self._abs_model = abs_model

        self._num_points = 0
        self._exp_id = -1
        self._start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        self._intrinsic_k: np.ndarray = None
        self._world_bounds: np.ndarray = None

    def add_datapoint(self) -> None:
        """Add a datapoint to the logger by querying the environment for relevant observations."""
        print("[INFO][LOGGER] Logging datapoint.")
        time_stamp = time.perf_counter()
        twist_obs = self._env.get_observation(self._env.unwrapped.obs_spec_twist_tcp)
        rgbd_obs = self._env.get_observation(self._env.unwrapped.obs_spec_rgbd)
        scene_obs: Scene = self._reconstructor.get_observation()
        scene_dict = scene_obs.serialize_scene_poses()
        # abstract_state = self._abs_model.get_abstract_state()
        if self._time_stamps is None:
            self._time_stamps = np.array([time_stamp])
            self._rgbd_obs = rgbd_obs["rgb"].cpu().numpy().squeeze()[None, ...]
            self._depth_obs = rgbd_obs["depth"].cpu().numpy().squeeze()[None, ...]
            self._camera_pose = rgbd_obs["camera_pose"].cpu().numpy().squeeze()[None, ...]
            self._intrinsic_k = rgbd_obs["intrinsic_k"].cpu().numpy().squeeze()[None, ...]
            self._tcp_pose = twist_obs["tcp_pose_b"].cpu().numpy().squeeze()[None, ...]
            # self._abs_state = np.array([abstract_state])
            self._obj_ids = scene_dict["ids"][None, ...]
            self._obj_poses = scene_dict["poses"][None, ...]
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
            # self._abs_state = np.concatenate((self._abs_state, np.array([abstract_state])), axis=-1)
            self._obj_ids = np.concatenate((self._obj_ids, scene_dict["ids"][None, ...]), axis=0)
            self._obj_poses = np.concatenate((self._obj_poses, scene_dict["poses"][None, ...]), axis=0)
            self._world_bounds = np.concatenate((self._world_bounds, scene_dict["bounds"][None, ...]), axis=0)

        self._num_points += 1

    def save_log(self) -> None:
        """Save the log to a file."""
        print("[INFO][LOGGER] Saving datafile")
        fpath = Path(f"{self._log_dir}/{self._start_time}/exp_{self._exp_id}")
        fpath.mkdir(exist_ok=True, parents=True)
        with h5py.File(f"{fpath}/data.h5", "w") as f:
            ep = f.create_group("episode")
            ep.create_dataset("time_stamps", data=self._time_stamps, compression="gzip")
            ep.create_dataset("rgb", data=self._rgbd_obs, compression="gzip")
            ep.create_dataset("depth", data=self._depth_obs, compression="gzip")
            ep.create_dataset("camera_pose", data=self._camera_pose, compression="gzip")
            ep.create_dataset("intrinsic_k", data=self._intrinsic_k, compression="gzip")
            ep.create_dataset("tcp_pose", data=self._tcp_pose, compression="gzip")
            # ep.create_dataset("abs_state", data=self._abs_state, compression="gzip")
            ep.create_dataset("obj_ids", data=self._obj_ids, compression="gzip")
            ep.create_dataset("obj_poses", data=self._obj_poses, compression="gzip")
            ep.create_dataset("world_bounds", data=self._world_bounds, compression="gzip")
