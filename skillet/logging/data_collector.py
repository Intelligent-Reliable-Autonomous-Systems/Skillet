"""data_collector.py

Controls logging of data and messages within Skillet.
"""

import copy
import time

import h5py
import numpy as np

from skillet.envs import SkilletEnv
from skillet.perception.localization.reconstructor_base import ReconstructorBase
from skillet.scene.abstract.abstract_model import AbstractModel


class SkilletDataCollector:
    def __init__(self, log_dir: str, env: SkilletEnv, reconstructor: ReconstructorBase, abs_model: AbstractModel):
        self._log_dir = log_dir
        self._env = env
        self._reconstructor = reconstructor
        self._abs_model = abs_model

        self._num_points = 0
        self._exp_id = 0
        self._start_time = time.perf_counter()

        self._rgbd_obs: np.ndarray = None
        self._depth_obs: np.ndarray = None
        self._camera_pos: np.ndarray = None
        self._twist_obs: np.ndarray = None
        self._time_stamps: np.ndarray = None
        self._abs_state: np.ndarray = None
        self._scenes: np.ndarray = None

    def setup_logging(self) -> None:
        """Set up logging."""
        pass

    def reset_logging(self, log_dir: str | None = None) -> None:
        """Reset the data collector by dumping everything to the log file and resetting buffers."""
        self._log_dir = log_dir if log_dir is not None else self._log_dir
        self._num_points = 0
        self._exp_id += 1

    def add_datapoint(self) -> None:
        """Add a datapoint to the logger by querying the environment for relevant observations."""
        time_stamp = time.perf_counter
        twist_obs = self._env.get_observation(self._env.obs_spec_twist_tcp)
        rgbd_obs = self._env.get_observation(self._env.obs_spec_rgbd)
        scene_obs = self._reconstructor.get_observation()
        abstract_state = self._abs_model.get_abstract_state()
        if self._time_stamps is None:
            self._time_stamps = np.array([time_stamp])
            self._rgbd_obs = rgbd_obs["rgb"].cpu().numpy().squeeze()
            self._depth_obs = rgbd_obs["depth"].cpu().numpy().squeeze()
            self._camera_pos = rgbd_obs["camera_pos"].cpu().numpy().squeeze()
            self._twist_obs = twist_obs["tcp_pose"].cpu().numpy().squeeze()
            self._abs_state = np.array([abstract_state])
            self._scenes = np.array([copy.deepcopy(scene_obs)])
        else:
            self._time_stamps = np.stack((self._time_stamps, np.array([time_stamp])), axis=-1)
            self._rgbd_obs = np.stack((self._rgbd_obs, rgbd_obs["rgb"].cpu().numpy().squeeze()), axis=-1)
            self._depth_obs = np.stack((self._depth_obs, rgbd_obs["depth"].cpu().numpy().squeeze()), axis=-1)
            self._camera_pos = np.stack((self._camera_pos, rgbd_obs["camera_pos"].cpu().numpy().squeeze()), axis=-1)
            self._twist_obs = np.stack((self._twist_obs, twist_obs["tcp_pose"].cpu().numpy().squeeze()), axis=-1)
            self._abs_state = np.stack((self._abs_state, np.array([abstract_state])), axis=-1)
            self._scenes = np.array((self._scenes, [copy.deepcopy(scene_obs)]), axis=-1)

        self._num_points += 1

    def save_log(self) -> None:
        """Save the log to a file."""
        with h5py.File(f"{self._log_dir}/{self._start_time}/exp_{self._exp_id}", "w") as f:
            ep = f.create_group("episode")

            ep.create_dataset("rgb", data=self._rgbd_obs, compression="gzip")
            ep.create_dataset("depth", data=self._depth_obs, compression="gzip")
            ep.create_dataset("camera_pos", data=self._camera_pos, compression="gzip")
            ep.create_dataset("tcp_pose", data=self._twist_obs, compression="gzip")
            ep.create_dataset("time_stamps", data=self._time_stamps, compression="gzip")
            ep.create_dataset("abs_state", data=self._abs_state, compression="gzip")
            ep.create_dataset("scenes", data=self._scenes, compression="gzip")
