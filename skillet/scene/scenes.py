"""Default Skillet Scenes."""

import torch

from skillet import DEVICE
from skillet.scene.base import Scene
from skillet.scene.scene_objs import Cube, Location, Table, Target

CUBE_SIZE = 0.044
SM_APRIL_SZ = 0.036
TABLE_X0 = 0.17
TABLE_Y0 = -0.48
TABLE_DX = 0.33
TABLE_DY = 0.96
WORLD_BOUNDS = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)

table_0 = Table(height=0.0, name="table0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE))
april_cube_0 = Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 1}])
april_cube_1 = Cube(size=CUBE_SIZE, face_apriltags=[{"face": "front", "size": SM_APRIL_SZ, "id": 2}])
april_cube_5 = Cube(size=CUBE_SIZE, face_apriltags=[{"face": "front", "size": SM_APRIL_SZ, "id": 5}])

red_cube = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.26, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="red_block"
)
blue_cube = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.44, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="blue_block"
)
purple_cube = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.35, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="purple_block"
)


THREE_CUBE_APRIL_SCENE = Scene(
    objects=[table_0, april_cube_0, april_cube_1, april_cube_5],
    closed_set=True,
    bounds=WORLD_BOUNDS,
    contains_objects=True,
)

THREE_CUBE_SCENE = Scene(
    objects=[table_0, red_cube, blue_cube, purple_cube], closed_set=True, bounds=WORLD_BOUNDS, contains_objects=True
)

EMPTY_SCENE = Scene(objects=[table_0], closed_set=False, bounds=WORLD_BOUNDS, contains_objects=False)

cube_0 = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.26, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="red_block"
)
cube_1 = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.44, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="blue_block"
)
cube_2 = Cube(
    size=CUBE_SIZE, init_pose=torch.as_tensor([0.35, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE), name="purple_block"
)


orange_april_cube = Cube(
    size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 8}], name="orange_block"
)
blue_april_cube = Cube(
    size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 9}], name="blue_block"
)
green_april_cube = Cube(
    size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 10}], name="green_block"
)
red_april_cube = Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 11}], name="red_block")
yellow_april_cube = Cube(
    size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 12}], name="yellow_block"
)
purple_april_cube = Cube(
    size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 13}], name="purple_block"
)

SIX_CUBE_APRIL_SCENE = Scene(
    objects=[
        table_0,
        orange_april_cube,
        blue_april_cube,
        green_april_cube,
        red_april_cube,
        yellow_april_cube,
        purple_april_cube,
    ],
    closed_set=True,
    bounds=WORLD_BOUNDS,
    contains_objects=True,
)
SIX_CUBE_APRIL_SCENE.goal = [{"predicate": "on", "args": ["yellow_block", "green_block"]}]

orange_cube = Cube(size=CUBE_SIZE, name="orange_block")
blue_cube = Cube(size=CUBE_SIZE, name="light_blue_block")
green_cube = Cube(size=CUBE_SIZE, name="green_block")
pink_cube = Cube(size=CUBE_SIZE, name="light_pink_block")
yellow_cube = Cube(size=CUBE_SIZE, name="yellow_block")
purple_cube = Cube(size=CUBE_SIZE, name="dark_purple_block")
red_cube = Cube(size=CUBE_SIZE, name="dark_red_block")

loc_00 = Location(init_pose=torch.as_tensor([0.225, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_00")
loc_01 = Location(init_pose=torch.as_tensor([0.275, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_01")
loc_02 = Location(init_pose=torch.as_tensor([0.325, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_02")
loc_03 = Location(init_pose=torch.as_tensor([0.375, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_03")
loc_04 = Location(init_pose=torch.as_tensor([0.425, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_04")
loc_05 = Location(init_pose=torch.as_tensor([0.475, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_05")
loc_06 = Location(init_pose=torch.as_tensor([0.525, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_06")
loc_07 = Location(init_pose=torch.as_tensor([0.575, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_07")

loc_10 = Location(init_pose=torch.as_tensor([0.225, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_10")
loc_11 = Location(init_pose=torch.as_tensor([0.275, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_11")
loc_12 = Location(init_pose=torch.as_tensor([0.325, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_12")
loc_13 = Location(init_pose=torch.as_tensor([0.375, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_13")
loc_14 = Location(init_pose=torch.as_tensor([0.425, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_14")
loc_15 = Location(init_pose=torch.as_tensor([0.475, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_15")
loc_16 = Location(init_pose=torch.as_tensor([0.525, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_16")
loc_17 = Location(init_pose=torch.as_tensor([0.575, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_17")

loc_20 = Location(init_pose=torch.as_tensor([0.225, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_20")
loc_21 = Location(init_pose=torch.as_tensor([0.275, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_21")
loc_22 = Location(init_pose=torch.as_tensor([0.325, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_22")
loc_23 = Location(init_pose=torch.as_tensor([0.375, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_23")
loc_24 = Location(init_pose=torch.as_tensor([0.425, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_24")
loc_25 = Location(init_pose=torch.as_tensor([0.475, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_25")
loc_26 = Location(init_pose=torch.as_tensor([0.525, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_26")
loc_27 = Location(init_pose=torch.as_tensor([0.575, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_27")


SIX_CUBE_SCENE = Scene(
    objects=[table_0, orange_cube, blue_cube, green_cube, pink_cube, yellow_cube, purple_cube, red_cube],
    closed_set=True,
    bounds=WORLD_BOUNDS,
    contains_objects=True,
)

SIX_CUBE_SCENE.goal = [
    {"predicate": "on", "args": ["light_blue_block", "green_block"]},
]

LOC_CUBE_SCENE = Scene(
    objects=[
        table_0,
        orange_cube,
        blue_cube,
        green_cube,
        pink_cube,
        yellow_cube,
        purple_cube,
        red_cube,
        loc_00,
        loc_01,
        loc_02,
        loc_03,
        loc_04,
        loc_05,
        loc_06,
        loc_07,
        loc_10,
        loc_11,
        loc_12,
        loc_13,
        loc_14,
        loc_15,
        loc_16,
        loc_17,
        loc_20,
        loc_21,
        loc_22,
        loc_23,
        loc_24,
        loc_25,
        loc_26,
        loc_27,
    ],
    closed_set=True,
    bounds=WORLD_BOUNDS,
    contains_objects=True,
)

LOC_CUBE_SCENE.goal = [
    {"predicate": "on", "args": ["light_pink_block", "dark_purple_block"]},
]
