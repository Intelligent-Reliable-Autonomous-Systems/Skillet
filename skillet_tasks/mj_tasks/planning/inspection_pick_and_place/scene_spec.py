"""Compiled scene package for the inspection pick-and-place task."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco  # type: ignore[import-untyped]

from skillet.scene.objects import DiscardLocation, InspectableCube, Platform
from skillet.scene.scene_objs import Table


@dataclass
class InspectionSceneSpec:
    """Compiled MuJoCo model, scene-graph objects, and metadata.

    Satisfies ``MujocoSceneSpec`` structurally — ``model`` is the only field
    required by that protocol; the remaining fields are inspection-task-specific.
    """

    model: mujoco.MjModel
    table: Table
    blocks: list[InspectableCube]
    platform: Platform
    discard: DiscardLocation
    block_names: list[str] = field(init=False)
    defective_flags: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        """Populate derived fields from the block list."""
        self.block_names = [b.name for b in self.blocks]
        self.defective_flags = [b.defective is True for b in self.blocks]
