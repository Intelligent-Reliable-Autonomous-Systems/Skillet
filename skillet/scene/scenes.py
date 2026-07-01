"""Default Skillet Scenes."""

import torch

from skillet import DEVICE
from skillet.scene.base import Scene
from skillet.scene.scene_objs import Cube, Location, Spill, Sponge, Table

CUBE_SIZE = 0.044
SPONGE_SIZE = 0.060
SPILL_SIZE = 0.060
TARGET_SIZE = 0.090
SM_APRIL_SZ = 0.036
TABLE_X0 = 0.17
TABLE_Y0 = -0.48
TABLE_DX = 0.33
TABLE_DY = 0.96
WORLD_BOUNDS = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)


def three_cube_april_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 1}]),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "front", "size": SM_APRIL_SZ, "id": 2}]),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "front", "size": SM_APRIL_SZ, "id": 5}]),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
    )


def three_cube_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(
                size=CUBE_SIZE,
                init_pose=torch.as_tensor([0.26, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE),
                name="red_block",
            ),
            Cube(
                size=CUBE_SIZE,
                init_pose=torch.as_tensor([0.44, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE),
                name="blue_block",
            ),
            Cube(
                size=CUBE_SIZE,
                init_pose=torch.as_tensor([0.35, 0.042, 0.016, 1, 0, 0, 0], device=DEVICE),
                name="purple_block",
            ),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
    )


def empty_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE))
        ],
        closed_set=False,
        bounds=WORLD_BOUNDS,
        contains_objects=False,
    )


def six_cube_april_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 8}], name="orange_block"),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 9}], name="blue_block"),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 10}], name="green_block"),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 11}], name="red_block"),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 12}], name="yellow_block"),
            Cube(size=CUBE_SIZE, face_apriltags=[{"face": "top", "size": SM_APRIL_SZ, "id": 13}], name="purple_block"),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
        goal=[{"predicate": "on", "args": ["yellow_block", "green_block"]}],
    )


def six_cube_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(size=CUBE_SIZE, name="orange_block", material="plastic", color="orange"),
            Cube(size=CUBE_SIZE, name="blue_block", material="plastic", color="blue"),
            Cube(size=CUBE_SIZE, name="green_block", material="wooden", color="green"),
            Cube(size=CUBE_SIZE, name="pink_block", material="plastic", color="pink"),
            Cube(size=CUBE_SIZE, name="yellow_block", material="wooden", color="yellow"),
            Cube(size=CUBE_SIZE, name="purple_block", material="wooden", color="dark_purple"),
            Cube(size=CUBE_SIZE, name="red_block", material="plastic", color="red", moveable=False),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
        goal=[
            {"predicate": "on", "args": ["blue_block", "green_block"]},
        ],
    )


def seven_cube_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(size=CUBE_SIZE, name="orange_block", material="plastic", color="orange"),
            Cube(size=CUBE_SIZE, name="blue_block", material="plastic", color="blue"),
            Cube(size=CUBE_SIZE, name="green_block", material="wooden", color="green"),
            Cube(size=CUBE_SIZE, name="pink_block", material="plastic", color="pink"),
            Cube(size=CUBE_SIZE, name="yellow_block", material="wooden", color="yellow"),
            Cube(size=CUBE_SIZE, name="purple_block", material="wooden", color="dark_purple"),
            Cube(size=CUBE_SIZE, name="red_block", material="plastic", color="red", moveable=False),
            Location(init_pose=torch.as_tensor([0.225, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_00"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_01"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_02"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_03"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_04"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_05"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_06"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_07"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_10"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_11"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_12"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_13"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_14"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_15"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_16"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_17"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_20"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_21"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_22"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_23"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_24"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_25"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_26"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_27"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_30"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_31"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_32"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_33"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_34"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_35"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_36"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_37"),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
        goal=[
            {"predicate": "on", "args": ["pink_block", "purple_block"]},
        ],
    )


def four_cube_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(size=CUBE_SIZE, name="green_block", material="wooden", color="green"),
            Cube(size=CUBE_SIZE, name="yellow_block", material="wooden", color="yellow"),
            Cube(size=CUBE_SIZE, name="red_block", material="plastic", color="red"),
            Cube(size=CUBE_SIZE, name="blue_block", material="plastic", color="blue"),
            Location(size=0.07, init_pose=torch.as_tensor([0.20, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_00"),
            Location(size=0.07, init_pose=torch.as_tensor([0.27, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_01"),
            Location(size=0.07, init_pose=torch.as_tensor([0.34, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_02"),
            Location(size=0.07, init_pose=torch.as_tensor([0.41, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_03"),
            Location(size=0.07, init_pose=torch.as_tensor([0.48, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_04"),
            # Location(init_pose=torch.as_tensor([0.475, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_05"),
            # Location(init_pose=torch.as_tensor([0.525, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_06"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_07"),
            Location(size=0.07, init_pose=torch.as_tensor([0.20, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_10"),
            Location(size=0.07, init_pose=torch.as_tensor([0.27, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_11"),
            Location(size=0.07, init_pose=torch.as_tensor([0.34, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_12"),
            Location(size=0.07, init_pose=torch.as_tensor([0.41, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_13"),
            Location(size=0.07, init_pose=torch.as_tensor([0.48, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_14"),
            # Location(init_pose=torch.as_tensor([0.475, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_15"),
            # Location(init_pose=torch.as_tensor([0.525, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_16"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_17"),
            Location(size=0.07, init_pose=torch.as_tensor([0.20, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_20"),
            Location(size=0.07, init_pose=torch.as_tensor([0.27, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_21"),
            Location(size=0.07, init_pose=torch.as_tensor([0.34, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_22"),
            Location(size=0.07, init_pose=torch.as_tensor([0.41, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_23"),
            Location(size=0.07, init_pose=torch.as_tensor([0.48, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_24"),
            # Location(init_pose=torch.as_tensor([0.475, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_25"),
            # Location(init_pose=torch.as_tensor([0.525, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_26"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_27"),
            Location(size=0.07, init_pose=torch.as_tensor([0.20, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_30"),
            Location(size=0.07, init_pose=torch.as_tensor([0.27, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_31"),
            Location(size=0.07, init_pose=torch.as_tensor([0.34, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_32"),
            Location(size=0.07, init_pose=torch.as_tensor([0.41, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_33"),
            Location(size=0.07, init_pose=torch.as_tensor([0.48, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_34"),
            # Location(init_pose=torch.as_tensor([0.475, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_35"),
            # Location(init_pose=torch.as_tensor([0.525, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_36"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_37"),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
        goal=[
            {"predicate": "on", "args": ["green_block", "blue_block"]},
        ],
    )


def five_cube_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(size=CUBE_SIZE, name="green_block", material="wooden", color="green"),
            Cube(size=CUBE_SIZE, name="pink_block", material="plastic", color="pink"),
            Cube(size=CUBE_SIZE, name="yellow_block", material="wooden", color="yellow"),
            Cube(size=CUBE_SIZE, name="blue_block", material="plastic", color="blue"),
            Cube(size=CUBE_SIZE, name="red_block", material="wooden", color="red"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_00"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_01"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_02"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_03"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_04"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_05"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_06"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_07"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_10"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_11"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_12"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_13"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_14"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_15"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_16"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_17"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_20"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_21"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_22"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_23"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_24"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_25"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_26"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_27"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_30"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_31"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_32"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_33"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_34"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_35"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_36"),
            # Location(init_pose=torch.as_tensor([0.575, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_37"),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
        goal=[
            {"predicate": "on", "args": ["pink_block", "green_block"]},
            {"predicate": "on", "args": ["yellow_block", "pink_block"]},
        ],
    )


def sponge_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Sponge(size=0.085, name="yellow_sponge"),
            Sponge(size=0.085, name="blue_sponge"),
            Spill(size=0.085, name="blue_water_spill"),
            # Location(init_pose=torch.as_tensor([0.25, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_00"),
            # Location(init_pose=torch.as_tensor([0.35, -0.25, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_01"),
            # Location(init_pose=torch.as_tensor([0.45, 0.25, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_02"),
            # Location(init_pose=torch.as_tensor([0.30, 0.4, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_03"),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
        goal=[
            {"predicate": "on", "args": ["blue_sponge", "table_0"]},
        ],
    )


def one_cube_scene_loader() -> None:
    return Scene(
        objects=[
            Table(height=0.0, name="table_0", init_pose=torch.as_tensor([0.35, 0.0, 0.0, 1, 0, 0, 0], device=DEVICE)),
            Cube(size=CUBE_SIZE, name="red_block", material="plastic", color="red"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_00"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_01"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_02"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_03"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_04"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_05"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_06"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_07"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_10"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_11"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_12"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_13"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_14"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_15"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_16"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_17"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_20"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_21"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_22"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_23"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_24"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_25"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_26"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_27"),
            Location(init_pose=torch.as_tensor([0.225, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_30"),
            Location(init_pose=torch.as_tensor([0.275, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_31"),
            Location(init_pose=torch.as_tensor([0.325, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_32"),
            Location(init_pose=torch.as_tensor([0.375, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_33"),
            Location(init_pose=torch.as_tensor([0.425, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_34"),
            Location(init_pose=torch.as_tensor([0.475, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_35"),
            Location(init_pose=torch.as_tensor([0.525, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_36"),
            Location(init_pose=torch.as_tensor([0.575, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_37"),
        ],
        closed_set=True,
        bounds=WORLD_BOUNDS,
        contains_objects=True,
    )


# Location(size=0.1, init_pose=torch.as_tensor([0.225, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_00"),
# Location(size=0.1, init_pose=torch.as_tensor([0.325, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_01"),
# Location(size=0.1, init_pose=torch.as_tensor([0.425, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_02"),
# Location(size=0.1, init_pose=torch.as_tensor([0.525, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_03"),
# Location(size=0.1, init_pose=torch.as_tensor([0.625, 0.0, -0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_04"),
# Location(size=0.1, init_pose=torch.as_tensor([0.225, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_10"),
# Location(size=0.1, init_pose=torch.as_tensor([0.325, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_11"),
# Location(size=0.1, init_pose=torch.as_tensor([0.425, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_12"),
# Location(size=0.1, init_pose=torch.as_tensor([0.525, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_13"),
# Location(size=0.1, init_pose=torch.as_tensor([0.625, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_14"),
# Location(size=0.1, init_pose=torch.as_tensor([0.225, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_20"),
# Location(size=0.1, init_pose=torch.as_tensor([0.325, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_21"),
# Location(size=0.1, init_pose=torch.as_tensor([0.425, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_22"),
# Location(size=0.1, init_pose=torch.as_tensor([0.525, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_23"),
# Location(size=0.1, init_pose=torch.as_tensor([0.625, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]), name="loc_24"),
# Location(size=0.1, init_pose=torch.as_tensor([0.225, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_30"),
# Location(size=0.1, init_pose=torch.as_tensor([0.325, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_31"),
# Location(size=0.1, init_pose=torch.as_tensor([0.425, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_32"),
# Location(size=0.1, init_pose=torch.as_tensor([0.525, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_33"),
# Location(size=0.1, init_pose=torch.as_tensor([0.625, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]), name="loc_34"),
