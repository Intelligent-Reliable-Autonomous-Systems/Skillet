"""Default Skillet Scenes."""

import torch
from skillet.scene.cube import Cube
from skillet.scene.base import Scene

TABLE_X0 = -0.0889
TABLE_Y0 = -0.577
TABLE_DX = 0.762
TABLE_DY = 1.2446
WORLD_BOUNDS = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)

april_cube_0 = Cube(size=0.041, face_apriltags=[{"face": "top", "size": 0.036, "id": 1}])
april_cube_1 = Cube(size=0.041, face_apriltags=[{"face": "front", "size": 0.036, "id": 2}])
april_cube_2 = Cube(size=0.041, face_apriltags=[{"face": "front", "size": 0.036, "id": 5}])

cube_0 = Cube(size=0.041, init_pose=torch.as_tensor([0.26, 0.041, 0.016, 1, 0, 0, 0], device="cuda"))
cube_1 = Cube(size=0.041, init_pose=torch.as_tensor([0.44, 0.041, 0.016, 1, 0, 0, 0], device="cuda"))
cube_2 = Cube(size=0.041, init_pose=torch.as_tensor([0.35, 0.041, 0.016, 1, 0, 0, 0], device="cuda"))

THREE_CUBE_APRIL_SCENE = Scene(
    objects=[cube_0, cube_1, cube_2], closed_set=True, bounds=WORLD_BOUNDS, contains_objects=True
)

THREE_CUBE_SCENE = Scene(objects=[cube_0, cube_1, cube_2], closed_set=True, bounds=WORLD_BOUNDS, contains_objects=True)

EMPTY_SCENE = Scene(objects=[], closed_set=False, bounds=WORLD_BOUNDS, contains_objects=False)
