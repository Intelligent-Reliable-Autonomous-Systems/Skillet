"""Default Skillet Scenes."""

import torch
from skillet import DEVICE
from skillet.scene.cube import Cube, Table
from skillet.scene.base import Scene

CUBE_SIZE = 0.044
SM_APRIL_SZ = 0.036
TABLE_X0 = -0.0889
TABLE_Y0 = -0.577
TABLE_DX = 0.762
TABLE_DY = 1.2446
WORLD_BOUNDS = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)

table_0 = Table(height=0.0, name="table0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE))
april_cube_0 = Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 1}])
april_cube_1 = Cube(size=CUBE_SIZE, face_apriltags=[{"face": "front", "size": SM_APRIL_SZ, "id": 2}])
april_cube_2 = Cube(size=CUBE_SIZE, face_apriltags=[{"face": "front", "size": SM_APRIL_SZ, "id": 5}])

cube_0 = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.26, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="red_block"
)
cube_1 = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.44, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="blue_block"
)
cube_2 = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.35, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="green_block"
)


THREE_CUBE_APRIL_SCENE = Scene(
    objects=[table_0, cube_0, cube_1, cube_2], closed_set=True, bounds=WORLD_BOUNDS, contains_objects=True
)

THREE_CUBE_SCENE = Scene(
    objects=[table_0, cube_0, cube_1, cube_2], closed_set=True, bounds=WORLD_BOUNDS, contains_objects=True
)

EMPTY_SCENE = Scene(objects=[table_0], closed_set=False, bounds=WORLD_BOUNDS, contains_objects=False)
