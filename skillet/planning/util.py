import random

import numpy as np
import torch

from skillet.planning.abstract.spatial_grounding import _is_at
from skillet.scene.base import Scene
from skillet.scene.scene_objs import Cube, Location


def find_valid_table_xy(scene: Scene) -> torch.Tensor:
    """Find a valid clear position on the table to place an object.

    Prioritizes finding an open X position first, then finds the Y position
    that is minimally far from other objects while respecting the buffer.

    Args:
        scene: scene object containing cube positions

    """
    occupied_positions = [obj for obj in scene.objects if isinstance(obj, Cube)]
    candidate_locations = [obj for obj in scene.objects if isinstance(obj, Location) and "loc_0" in obj.name]
    random.shuffle(candidate_locations)
    for loc in candidate_locations:
        if np.asarray([_is_at(obj, loc) for obj in occupied_positions]).any():
            continue

        return torch.cat((loc.pose[0:1] + loc.size / 2, loc.pose[1:2], torch.as_tensor([0.0], device=loc.pose.device)))

    raise RuntimeError("Could not find a valid table position. Table may be too crowded.")
