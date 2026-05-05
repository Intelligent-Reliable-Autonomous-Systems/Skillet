"""Tests for inspection-task scene objects: InspectableCube, Platform, DiscardLocation."""

from __future__ import annotations

import torch

from skillet.scene.base import Scene
from skillet.scene.objects import DiscardLocation, InspectableCube, Platform
from skillet.scene.scene_objs import Cube

_POSE = torch.tensor([0.3, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0])


def test_inspectable_cube_is_cube() -> None:
    """InspectableCube is a subclass of Cube."""
    cube = InspectableCube(size=0.044, init_pose=_POSE)
    assert isinstance(cube, Cube)


def test_inspectable_cube_defective_none_by_default() -> None:
    """defective is None when not provided at construction."""
    cube = InspectableCube(size=0.044, init_pose=_POSE)
    assert cube.defective is None


def test_inspectable_cube_defective_label() -> None:
    """defective reflects the value passed at construction."""
    assert InspectableCube(size=0.044, defective=False, init_pose=_POSE).defective is False
    assert InspectableCube(size=0.044, defective=True, init_pose=_POSE).defective is True


def test_inspectable_cube_defective_setter() -> None:
    """defective can be updated after construction."""
    cube = InspectableCube(size=0.044, init_pose=_POSE)
    cube.defective = True
    assert cube.defective is True


def test_platform_is_pose_known_without_pose() -> None:
    """is_pose_known returns False when no init_pose is given."""
    assert not Platform(width=0.2, depth=0.2, height=0.05).is_pose_known()


def test_platform_is_pose_known_with_pose() -> None:
    """is_pose_known returns True when init_pose is provided."""
    pose = torch.tensor([0.5, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0])
    assert Platform(width=0.2, depth=0.2, height=0.05, init_pose=pose).is_pose_known()


def test_platform_aabb() -> None:
    """AABB has shape (6,) and correct min/max values."""
    pose = torch.tensor([0.5, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0])
    aabb = Platform(width=0.2, depth=0.2, height=0.05, init_pose=pose).aabb
    assert aabb.shape == (6,)
    assert torch.allclose(aabb[:3], torch.tensor([0.4, -0.1, 0.0]))
    assert torch.allclose(aabb[3:], torch.tensor([0.6, 0.1, 0.05]))


def test_discard_location_is_pose_known_without_pose() -> None:
    """is_pose_known returns False when no init_pose is given."""
    assert not DiscardLocation(width=0.15, depth=0.15).is_pose_known()


def test_discard_location_is_pose_known_with_pose() -> None:
    """is_pose_known returns True when init_pose is provided."""
    pose = torch.tensor([0.1, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert DiscardLocation(width=0.15, depth=0.15, init_pose=pose).is_pose_known()


def test_discard_location_aabb_shape() -> None:
    """AABB has shape (6,)."""
    pose = torch.tensor([0.1, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert DiscardLocation(width=0.15, depth=0.15, init_pose=pose).aabb.shape == (6,)


def test_scene_serialize_mixed_objects() -> None:
    """Scene with all four object types serializes without error."""
    objects = [
        Cube(size=0.044, init_pose=torch.tensor([0.26, 0.04, 0.022, 1.0, 0.0, 0.0, 0.0]), name="blue_block"),
        InspectableCube(
            size=0.044, defective=True, name="red_block",
            init_pose=torch.tensor([0.35, 0.04, 0.022, 1.0, 0.0, 0.0, 0.0]),
        ),
        Platform(
            width=0.2, depth=0.2, height=0.05, name="platform",
            init_pose=torch.tensor([0.5, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]),
        ),
        DiscardLocation(
            width=0.15, depth=0.15, name="discard",
            init_pose=torch.tensor([0.1, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ),
    ]
    scene = Scene(objects=objects, closed_set=True, bounds=(0.1, -0.5, 0.0, 0.6, 0.5, 1.0))
    result = scene.serialize_scene_poses()

    assert result["poses"].shape == (4, 7)
    assert result["ids"].shape == (4,)
    assert result["names"].tolist() == ["blue_block", "red_block", "platform", "discard"]
