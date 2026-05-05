"""Protocol for compiled MuJoCo scene packages.

Any object that carries a compiled ``mujoco.MjModel`` satisfies
``MujocoSceneSpec`` structurally.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import mujoco  # type: ignore[import-untyped]


@runtime_checkable
class MujocoSceneSpec(Protocol):
    """Minimal contract for a compiled MuJoCo scene.

    Concrete scene specs (e.g. ``InspectionSceneSpec``) carry additional
    task-specific fields; code that only drives the simulation loop should
    type-hint against this protocol.
    """

    model: mujoco.MjModel
