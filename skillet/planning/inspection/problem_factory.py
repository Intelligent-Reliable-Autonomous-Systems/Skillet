"""Generate PDDL problem instances from inspection scene objects."""

from __future__ import annotations

from skillet.scene.objects import DiscardLocation, InspectableCube, Platform
from skillet.scene.scene_objs import Table


def make_inspection_problem(
    table: Table,
    blocks: list[InspectableCube],
    platform: Platform,
    discard: DiscardLocation,
) -> str:
    """Return a PDDL problem string for an inspection pick-and-place task.

    Defect verdicts are encoded in the initial state so the planner can
    reason about routing before execution.  The ``inspected`` and
    ``gripper-above`` predicates are absent from the initial state: every
    block must be approached and classified at runtime before it can be
    routed.

    Args:
        table: Table scene object; its ``name`` becomes the PDDL table object.
        blocks: Ordered list of inspectable blocks.  Each block's ``name``
            becomes a PDDL block object; ``defective`` sets the initial label.
        platform: Platform scene object; destination for non-defective blocks.
        discard: Discard location; destination for defective blocks.

    Returns:
        A PDDL problem string compatible with ``inspection.domain.pddl``.

    Raises:
        ValueError: If any block has ``defective is None`` (label not set).

    """
    for b in blocks:
        if b.defective is None:
            raise ValueError(
                f"block {b.name!r} has no defect label — cannot generate PDDL problem"
            )

    block_names = " ".join(b.name for b in blocks)

    init_facts: list[str] = ["(gripper-empty)"]
    for b in blocks:
        init_facts.append(f"(on {b.name} {table.name})")
        label = "defective" if b.defective else "non-defective"
        init_facts.append(f"({label} {b.name})")

    goal_facts: list[str] = []
    for b in blocks:
        dest = discard.name if b.defective else platform.name
        goal_facts.append(f"(on {b.name} {dest})")

    init_str = "\n        ".join(init_facts)
    goal_str = "\n            ".join(goal_facts)

    return f"""\
(define (problem inspection-task)
    (:domain inspection)
    (:objects
        {block_names} - block
        {table.name} - table0
        {platform.name} - platform0
        {discard.name} - discard0
    )
    (:init
        {init_str}
    )
    (:goal
        (and
            {goal_str}
        )
    )
)
"""
