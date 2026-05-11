"""Tests for the MuJoCo inspection pick-and-place scene factory and env (Steps 2.1–2.2)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import (
    make_inspection_scene,
    make_inspection_scene_xml,
)
from skillet.scene.objects import InspectableCube, Platform, DiscardLocation
from skillet.scene.scene_objs import Table
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import (
    CUBE_SIZE,
    DEFAULT_TABLE_HEIGHT,
    DEFAULT_PLATFORM_SIZE_MULT,
    DEFAULT_PLATFORM_HEIGHT_MULT,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mixed_scene():
    """Scene with 3 blocks: clean, defective, clean."""
    return make_inspection_scene([False, True, False])


@pytest.fixture()
def all_defective_scene():
    """Scene with 2 defective blocks."""
    return make_inspection_scene([True, True])


@pytest.fixture()
def single_block_scene():
    """Minimal scene: one clean block."""
    return make_inspection_scene([False])


# ---------------------------------------------------------------------------
# Geometry counts
# ---------------------------------------------------------------------------


def test_ngeom_three_blocks(mixed_scene) -> None:
    """floor + table + 4 legs + platform + discard_region + 3 block geoms = 11."""
    assert mixed_scene.model.ngeom == 11


def test_ngeom_two_blocks(all_defective_scene) -> None:
    """floor + table + 4 legs + platform + discard_region + 2 block geoms = 10."""
    assert all_defective_scene.model.ngeom == 10


def test_ngeom_one_block(single_block_scene) -> None:
    """floor + table + 4 legs + platform + discard_region + 1 block geom = 9."""
    assert single_block_scene.model.ngeom == 9


# ---------------------------------------------------------------------------
# Free joints (one per block)
# ---------------------------------------------------------------------------


def test_njnt_three_blocks(mixed_scene) -> None:
    assert mixed_scene.model.njnt == 3


def test_njnt_two_blocks(all_defective_scene) -> None:
    assert all_defective_scene.model.njnt == 2


# ---------------------------------------------------------------------------
# Material / texture assignment matches defect flags
# ---------------------------------------------------------------------------


def _geom_matname(model: mujoco.MjModel, geom_name: str) -> str:
    """Return the material name for a named geom."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mat_id = model.geom_matid[geom_id]
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)


def test_defect_material_applied(mixed_scene) -> None:
    """Block 1 (defective) must use defect_mat."""
    assert _geom_matname(mixed_scene.model, "block_1_geom") == "defect_mat"


def test_clean_material_applied(mixed_scene) -> None:
    """Blocks 0 and 2 (clean) must use clean_mat."""
    assert _geom_matname(mixed_scene.model, "block_0_geom") == "clean_mat"
    assert _geom_matname(mixed_scene.model, "block_2_geom") == "clean_mat"


def test_all_defective_materials(all_defective_scene) -> None:
    """All blocks should use defect_mat when all are defective."""
    for i in range(2):
        assert _geom_matname(all_defective_scene.model, f"block_{i}_geom") == "defect_mat"


# ---------------------------------------------------------------------------
# Physics step runs without error
# ---------------------------------------------------------------------------


def test_physics_step_no_error(mixed_scene) -> None:
    """mj_step must not raise for a freshly compiled model."""
    data = mujoco.MjData(mixed_scene.model)
    mujoco.mj_step(mixed_scene.model, data)


def test_physics_step_all_defective(all_defective_scene) -> None:
    data = mujoco.MjData(all_defective_scene.model)
    mujoco.mj_step(all_defective_scene.model, data)


# ---------------------------------------------------------------------------
# XML + asset pair
# ---------------------------------------------------------------------------


def _make_default_xml():
    import torch

    table = Table(height=DEFAULT_TABLE_HEIGHT, name="table")
    platform_size = CUBE_SIZE * DEFAULT_PLATFORM_SIZE_MULT
    platform_height = CUBE_SIZE * DEFAULT_PLATFORM_HEIGHT_MULT
    platform = Platform(
        width=platform_size,
        depth=platform_size,
        height=platform_height,
        name="platform",
        init_pose=torch.tensor([0.45, 0.28, DEFAULT_TABLE_HEIGHT + platform_height / 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    discard = DiscardLocation(
        width=platform_size,
        depth=platform_size,
        name="discard",
        init_pose=torch.tensor([0.45, -0.28, DEFAULT_TABLE_HEIGHT + 0.001, 1.0, 0.0, 0.0, 0.0]),
    )
    blocks = [
        InspectableCube(
            size=CUBE_SIZE,
            defective=False,
            name="block_0",
            init_pose=torch.tensor([0.35, 0.0, DEFAULT_TABLE_HEIGHT + CUBE_SIZE / 2.0, 1.0, 0.0, 0.0, 0.0]),
        )
    ]
    return make_inspection_scene_xml(table, blocks, platform, discard)


def test_xml_returns_string_and_assets() -> None:
    xml, assets = _make_default_xml()
    assert isinstance(xml, str)
    assert "defect_texture.png" in assets
    assert "clean_texture.png" in assets


def test_xml_assets_are_bytes() -> None:
    _, assets = _make_default_xml()
    for v in assets.values():
        assert isinstance(v, bytes)
        assert len(v) > 0


def test_xml_contains_worldbody() -> None:
    xml, _ = _make_default_xml()
    assert "<worldbody>" in xml
    assert '<geom name="floor"' in xml


# ---------------------------------------------------------------------------
# InspectionSceneSpec fields
# ---------------------------------------------------------------------------


def test_spec_block_names(mixed_scene) -> None:
    assert mixed_scene.block_names == ["block_0", "block_1", "block_2"]


def test_spec_defective_flags(mixed_scene) -> None:
    assert mixed_scene.defective_flags == [False, True, False]


def test_spec_carries_scene_objects(mixed_scene) -> None:
    assert isinstance(mixed_scene.table, Table)
    assert isinstance(mixed_scene.platform, Platform)
    assert isinstance(mixed_scene.discard, DiscardLocation)
    assert all(isinstance(b, InspectableCube) for b in mixed_scene.blocks)


# ---------------------------------------------------------------------------
# Platform size mult changes platform geometry
# ---------------------------------------------------------------------------


def test_platform_size_mult() -> None:
    """Doubling platform_size_mult should widen the platform geom."""
    spec_default = make_inspection_scene([False], platform_size_mult=DEFAULT_PLATFORM_SIZE_MULT)
    spec_big = make_inspection_scene([False], platform_size_mult=DEFAULT_PLATFORM_SIZE_MULT * 2)

    plat_id_default = mujoco.mj_name2id(spec_default.model, mujoco.mjtObj.mjOBJ_GEOM, "platform")
    plat_id_big = mujoco.mj_name2id(spec_big.model, mujoco.mjtObj.mjOBJ_GEOM, "platform")

    # geom_size stores half-extents; x half-extent should be larger in big spec
<<<<<<< HEAD
    assert spec_big.model.geom_size[plat_id_big][0] > spec_default.model.geom_size[plat_id_default][0]
=======
    assert (
        spec_big.model.geom_size[plat_id_big][0]
        > spec_default.model.geom_size[plat_id_default][0]
    )


# ---------------------------------------------------------------------------
# Step 2.1 — include_robot
# ---------------------------------------------------------------------------

@pytest.fixture()
def robot_scene():
    """Scene with 3 blocks and the Gen3 arm included."""
    return make_inspection_scene([False, True, False], include_robot=True)


def test_robot_scene_has_base_link(robot_scene) -> None:
    """base_link body must be present when include_robot=True."""
    body_id = mujoco.mj_name2id(robot_scene.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert body_id >= 0


def test_robot_scene_has_block_body(robot_scene) -> None:
    """block_0 body must be present alongside the robot."""
    body_id = mujoco.mj_name2id(robot_scene.model, mujoco.mjtObj.mjOBJ_BODY, "block_0")
    assert body_id >= 0


def test_robot_scene_physics_100_steps(robot_scene) -> None:
    """100 mj_step calls must not produce NaN in qpos."""
    data = mujoco.MjData(robot_scene.model)
    if robot_scene.model.nkey > 0:
        mujoco.mj_resetDataKeyframe(robot_scene.model, data, robot_scene.model.key("home").id)
    for _ in range(100):
        mujoco.mj_step(robot_scene.model, data)
    assert not np.isnan(data.qpos).any()


# ---------------------------------------------------------------------------
# Step 2.2 — InspectionMjEnv
# ---------------------------------------------------------------------------

from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.env import InspectionMjEnv  # noqa: E402


@pytest.fixture()
def inspection_env(robot_scene):
    """InspectionMjEnv wrapping the 3-block robot scene."""
    return InspectionMjEnv(robot_scene)


def test_env_obs_shapes(inspection_env) -> None:
    """reset() must return IKEE_Obs tensors with correct shapes (batch=1)."""
    obs, _ = inspection_env.reset()
    assert obs["ee_pose_b"].shape == (1, 7)
    assert obs["tcp_pose_b"].shape == (1, 7)
    assert obs["jacobians"].shape == (1, 6, 7)
    assert obs["joint_pos"].shape == (1, 8)
    assert obs["joint_vel"].shape == (1, 8)
    assert obs["tcp_offset"].shape == (1, 7)
    assert obs["gripper"].shape == (1, 1)
    assert obs["gripper_lim"].shape == (1, 2)
    assert obs["joint_lims"].shape == (1, 2, 8)


def test_env_obs_no_nan_after_200_steps(inspection_env) -> None:
    """200 random-action steps must not produce NaN in ee_pose_b."""
    obs, _ = inspection_env.reset()
    for _ in range(200):
        action = inspection_env.action_space.sample()
        obs, _, _, _, _ = inspection_env.step(action)
    assert not torch.isnan(obs["ee_pose_b"]).any()


def test_env_coerce_obs_spec(inspection_env) -> None:
    """coerce_obs_spec('ik_ee') must return the ik_ee spec without error."""
    spec = inspection_env.coerce_obs_spec("ik_ee")
    assert spec.name == "ik_ee"


def test_env_action_spec_name(inspection_env) -> None:
    """action_spec must have name 'joints_vel' for skill_lib compatibility."""
    assert inspection_env.action_spec.name == "joints_vel"


def test_env_get_observation_matches_step(inspection_env) -> None:
    """get_observation() must return the same tensors as the last step()."""
    obs, _ = inspection_env.reset()
    action = inspection_env.action_space.sample()
    obs_step, _, _, _, _ = inspection_env.step(action)
    obs_get = inspection_env.get_observation()
    for k in obs_step:
        assert torch.allclose(obs_step[k], obs_get[k]), f"Mismatch for key {k!r}"
>>>>>>> 65634d3 (Create  Mujoco Gym env)
