

import gymnasium as gym
import numpy as np
import torch

from skillet.core import ObservationSpec
from skillet.core.spaces import BatchedSpaceItem, ParameterizedBox, SpaceItem

RGBD_SPEC_BATCHED = ObservationSpec[dict[str, BatchedSpaceItem[torch.Tensor]]](
    space=gym.spaces.Dict({
        "rgb": ParameterizedBox(low=0, high=255, shape=(3, "height", "width"), dtype=np.uint8),
        # Depth is normalized to float32 meters for downstream perception.
        "depth": ParameterizedBox(low=0.0, high=10.0, shape=(1, "height", "width"), dtype=np.float32),
        "intrinsic_k": gym.spaces.Box(low=0.0, high=2000.0, shape=(3, 3), dtype=np.float32),
        "camera_pose": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
        "timestamp": gym.spaces.Box(low=0.0, high=1e10, shape=(), dtype=np.float64),
    }),
    name="rgb-d",
    is_torch=True,
    is_batched=True,
    n_envs=-1
)

