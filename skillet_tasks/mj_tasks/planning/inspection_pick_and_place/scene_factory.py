"""Factory for the inspection pick-and-place MuJoCo scene.

Scene-graph objects (``Table``, ``InspectableCube``, ``Platform``,
``DiscardLocation``) are the single source of truth.  The MuJoCo XML is
derived from their properties so physics and planner representations remain
consistent.  The robot is composed in separately when constructing the full
task environment.

Workspace convention (robot base at origin, +x forward, +y left):
  - Table surface:   z = table.height (default 0.5 m).
  - Blocks:          centred at x ≈ 0.35, spread along y, resting on table.
  - Platform:        forward-left  (high x, positive y), sitting on table.
  - Discard region:  forward-right (high x, negative y), flush with table.
"""

from __future__ import annotations

from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import torch

from skillet.scene.objects import DiscardLocation, InspectableCube, Platform
from skillet.scene.scene_objs import Cube, Table
from skillet_tasks.assets.mujoco.inspection_scene import load_texture
from skillet_tasks.assets.utils import update_assets
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_spec import (
    InspectionSceneSpec,
)

_GEN3_XML: Path = (
    Path(__file__).parents[4] / "skillet_tasks" / "assets" / "mujoco" / "kinova_gen3" / "xmls" / "gen3_gripper.xml"
)

CUBE_SIZE: float = 0.044

DEFAULT_TABLE_HEIGHT: float = 0.5
# Robot base position in world frame.  The Gen3 is attached at the rear edge of
# the table at table-surface height; see _build_full_spec for the attachment site.
ROBOT_BASE_WORLD_POS: tuple[float, float, float] = (0.05, 0.0, DEFAULT_TABLE_HEIGHT)
_TABLE_THICKNESS: float = 0.05
_TABLE_HALF_X: float = 0.30
_TABLE_HALF_Y: float = 0.50

DEFAULT_PLATFORM_SIZE_MULT: float = 3.0
DEFAULT_PLATFORM_HEIGHT_MULT: float = 2.0

# Workspace placement (metres, robot frame).
_PLATFORM_XY: tuple[float, float] = (0.45, 0.28)  # forward-left
_DISCARD_XY: tuple[float, float] = (0.45, -0.28)  # forward-right
_BLOCK_X: float = 0.35
_BLOCK_SPACING_MULT: float = 3.0


def make_inspection_scene(
    block_defective: list[bool],
    cube_size: float = CUBE_SIZE,
    table_height: float = DEFAULT_TABLE_HEIGHT,
    platform_size_mult: float = DEFAULT_PLATFORM_SIZE_MULT,
    platform_height_mult: float = DEFAULT_PLATFORM_HEIGHT_MULT,
    include_robot: bool = False,
) -> InspectionSceneSpec:
    """Build and compile a MuJoCo inspection scene.

    Creates scene-graph objects first, then derives the MuJoCo XML from their
    properties.  The returned ``InspectionSceneSpec`` carries both so callers
    have a single consistent representation.

    Args:
        block_defective: Ground-truth defect flag per block.  Length sets
            the number of blocks spawned.
        cube_size: Side length of each block in metres.
        table_height: Height of the table surface above the world origin in metres.
        platform_size_mult: Platform footprint as a multiple of ``cube_size``.
        platform_height_mult: Platform height as a multiple of ``cube_size``.
        include_robot: When ``True``, compose the Gen3 + 2F-85 MJCF into the
            scene so the returned model contains both the workspace and the arm.
            Defaults to ``False`` for backwards compatibility.

    Returns:
        An ``InspectionSceneSpec`` with the compiled model and scene-graph objects.

    """
    platform_size = cube_size * platform_size_mult
    platform_height = cube_size * platform_height_mult

    table = Table(height=table_height, name="table")

    px, py = _PLATFORM_XY
    platform = Platform(
        width=platform_size,
        depth=platform_size,
        height=platform_height,
        name="platform",
        init_pose=torch.tensor([px, py, table_height + platform_height / 2.0, 1.0, 0.0, 0.0, 0.0]),
    )

    dx, dy = _DISCARD_XY
    discard = DiscardLocation(
        width=platform_size,
        depth=platform_size,
        name="discard",
        init_pose=torch.tensor([dx, dy, table_height + 0.001, 1.0, 0.0, 0.0, 0.0]),
    )

    xy_positions = _block_positions(len(block_defective), cube_size)
    blocks = [
        InspectableCube(
            size=cube_size,
            defective=defective,
            name=f"block_{i}",
            init_pose=torch.tensor(
                [
                    xy_positions[i][0],
                    xy_positions[i][1],
                    table_height + cube_size / 2.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        for i, defective in enumerate(block_defective)
    ]

    if include_robot:
        model = _compile_full_scene(table, blocks, platform, discard)
    else:
        xml, assets = make_inspection_scene_xml(table, blocks, platform, discard)
        model = mujoco.MjModel.from_xml_string(xml, assets)
    return InspectionSceneSpec(model=model, table=table, blocks=blocks, platform=platform, discard=discard)


def make_inspection_scene_xml(
    table: Table,
    blocks: list[InspectableCube],
    platform: Platform,
    discard: DiscardLocation,
) -> tuple[str, dict[str, bytes]]:
    """Derive MuJoCo XML from scene-graph objects.

    All geometry values are read from the scene objects, making them the
    authoritative source.  Callers who need to compose this scene with other
    specs (e.g. the Gen3 robot) can use this function directly.

    Args:
        table: Table scene object.  ``table.height`` sets the surface z.
        blocks: List of inspectable blocks.  Pose and size are read from each.
        platform: Platform scene object.
        discard: Discard-location scene object.

    Returns:
        A ``(xml_string, assets_dict)`` pair.

    """
    table_centre_z = table.height - _TABLE_THICKNESS / 2.0
    table_x = _TABLE_HALF_X / 2.0 + 0.17  # shifted forward to cover workspace
    leg_fragments = _table_leg_xml(table_x, table.height)

    for b in blocks:
        if b.defective is None:
            raise ValueError(f"block {b.name!r} has no defect label — cannot generate XML")
    block_fragments = "\n".join(
        _block_xml(i, b, "defect_mat" if b.defective else "clean_mat") for i, b in enumerate(blocks)
    )

    px, py, pz = (v.item() for v in platform.pose[:3])
    p_hxy = platform.width / 2.0
    p_hz = platform.height / 2.0

    dx, dy, dz = (v.item() for v in discard.pose[:3])
    d_hxy = discard.width / 2.0
    d_hz = discard.slab_thickness / 2.0

    xml = f"""<mujoco model="inspection_scene">
  <asset>
    <texture name="defect_tex" type="2d" file="defect_texture.png"/>
    <material name="defect_mat" texture="defect_tex" specular="0" shininess="0"/>
    <texture name="clean_tex" type="2d" file="clean_texture.png"/>
    <material name="clean_mat" texture="clean_tex" specular="0" shininess="0"/>
    <material name="table_mat" rgba="0.82 0.71 0.55 1"/>
    <material name="platform_mat" rgba="0.25 0.65 0.25 1"/>
    <material name="discard_mat" rgba="0.65 0.25 0.25 0.8"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <geom name="table"
          type="box"
          size="{_TABLE_HALF_X:.6f} {_TABLE_HALF_Y:.6f} {_TABLE_THICKNESS / 2.0:.6f}"
          pos="{table_x:.6f} 0 {table_centre_z:.6f}"
          material="table_mat"/>
{leg_fragments}
{block_fragments}
    <geom name="platform"
          type="box"
          size="{p_hxy:.6f} {p_hxy:.6f} {p_hz:.6f}"
          pos="{px:.6f} {py:.6f} {pz:.6f}"
          material="platform_mat"/>
    <geom name="discard_region"
          type="box"
          size="{d_hxy:.6f} {d_hxy:.6f} {d_hz:.6f}"
          pos="{dx:.6f} {dy:.6f} {dz:.6f}"
          material="discard_mat"/>
  </worldbody>
</mujoco>"""

    assets: dict[str, bytes] = {
        "defect_texture.png": load_texture("defect_texture.png"),
        "clean_texture.png": load_texture("clean_texture.png"),
    }
    return xml, assets


def make_full_scene_xml(
    table: Table,
    blocks: list[InspectableCube],
    platform: Platform,
    discard: DiscardLocation,
) -> tuple[str, dict[str, bytes]]:
    """Return the composed inspection + Gen3 scene as ``(xml_string, assets_dict)``.

    Uses ``mujoco.MjSpec.attach()`` to embed the Gen3 into the scene worldbody
    at the world origin (robot base frame = world frame).  Body names, joint
    names, and actuator names from ``gen3_gripper.xml`` are preserved unchanged.

    The resulting model has:
      - ``nq = 7 * len(blocks) + 15`` (7 freejoint DOF per block + 15 robot DOF)
      - ``nu = 8`` (7 arm position actuators + 1 ``fingers_actuator`` tendon)

    Args:
        table: Table scene object.
        blocks: List of inspectable blocks.
        platform: Platform scene object.
        discard: Discard-location scene object.

    Returns:
        A ``(xml_string, assets_dict)`` pair.  The XML is serialised via
        ``MjSpec.to_xml()`` after composition; use ``_compile_full_scene``
        when you need a compiled ``MjModel`` directly.

    """
    spec = _build_full_spec(table, blocks, platform, discard)
    return spec.to_xml(), dict(spec.assets)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_full_spec(
    table: Table,
    blocks: list[InspectableCube],
    platform: Platform,
    discard: DiscardLocation,
) -> mujoco.MjSpec:
    """Build a composed ``MjSpec`` containing the inspection scene + Gen3 robot."""
    scene_xml, scene_assets = make_inspection_scene_xml(table, blocks, platform, discard)

    scene_spec = mujoco.MjSpec.from_string(scene_xml, scene_assets)
    scene_spec.assets = scene_assets

    gen3_spec = mujoco.MjSpec.from_file(str(_GEN3_XML))
    gen3_assets: dict[str, bytes] = {}
    update_assets(gen3_assets, _GEN3_XML.parent / gen3_spec.meshdir, gen3_spec.meshdir)
    gen3_spec.assets = gen3_assets

    # Attach robot at the rear edge of the table, at table surface height
    attach_site = scene_spec.worldbody.add_site()
    attach_site.name = "robot_attach"
    attach_site.pos = [0.05, 0.0, table.height]
    scene_spec.attach(gen3_spec, prefix="", site=attach_site)
    return scene_spec


def _compile_full_scene(
    table: Table,
    blocks: list[InspectableCube],
    platform: Platform,
    discard: DiscardLocation,
) -> mujoco.MjModel:
    """Compile and return a ``MjModel`` containing the inspection scene + Gen3 robot.WW"""
    return _build_full_spec(table, blocks, platform, discard).compile()


def _table_leg_xml(table_x: float, table_height: float) -> str:
    """Return four table-leg geom fragments connecting the table top to the floor."""
    leg_radius = 0.025
    leg_half_h = (table_height - _TABLE_THICKNESS) / 2.0
    leg_z = leg_half_h

    x_back = table_x - _TABLE_HALF_X + leg_radius + 0.02
    x_front = table_x + _TABLE_HALF_X - leg_radius - 0.02
    y_side = _TABLE_HALF_Y - leg_radius - 0.02

    corners = [
        ("bl", x_back, -y_side),
        ("br", x_back, y_side),
        ("fl", x_front, -y_side),
        ("fr", x_front, y_side),
    ]
    frags = [
        f'    <geom name="table_leg_{tag}" type="cylinder"'
        f' size="{leg_radius:.4f} {leg_half_h:.4f}"'
        f' pos="{x:.6f} {y:.6f} {leg_z:.6f}"'
        f' material="table_mat"/>'
        for tag, x, y in corners
    ]
    return "\n".join(frags)


def _block_positions(n: int, cube_size: float) -> list[tuple[float, float]]:
    """Return evenly-spaced (x, y) block positions centred in front of the robot."""
    spacing = _BLOCK_SPACING_MULT * cube_size
    y_start = -(n - 1) * spacing / 2.0
    return [(_BLOCK_X, y_start + i * spacing) for i in range(n)]


def _block_xml(idx: int, block: Cube, material: str) -> str:
    """Return an MJCF XML fragment for a single free-body block."""
    half = block.size / 2.0
    x, y, z = (v.item() for v in block.pose[:3])
    return (
        f'    <body name="block_{idx}" pos="{x:.6f} {y:.6f} {z:.6f}">\n'
        f'      <freejoint name="block_{idx}_joint"/>\n'
        f'      <geom name="block_{idx}_geom" type="box"'
        f' size="{half:.6f} {half:.6f} {half:.6f}"'
        f' material="{material}" mass="0.06" friction="1.0 0.005 0.0001"/>\n'
        f"    </body>"
    )
