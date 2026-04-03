"""data_collector.py

Controls logging of data and messages within Skillet.
"""

import time

from skillet.envs import SkilletEnv
from skillet.perception.localization.reconstructor_base import ReconstructorBase
from skillet.scene.abstract.abstract_model import AbstractModel


class SkilletDataCollector:
    def __init__(self, env: SkilletEnv, reconstructor: ReconstructorBase, abs_model: AbstractModel):
        self._env = env
        self._reconstructor = reconstructor
        self.abs_model = abs_model

    def add_datapoint(self) -> None:
        """Add a datapoint to the logger by querying the environment for relevant observations."""
        time_stamp = time.perf_counter
        twist_obs = self._env.get_observation(self._env.obs_spec_twist_tcp)
        rgbd_obs = self._env.get_observation(self._env.obs_spec_rgbd)
        scene_obs = self._reconstructor.get_observation()
