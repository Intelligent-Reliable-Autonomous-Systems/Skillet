<<<<<<< HEAD
# examples/view_inspection_scene.py
import mujoco
import mujoco.viewer
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import make_inspection_scene

spec = make_inspection_scene([False, True, False])  # clean, defective, clean
data = mujoco.MjData(spec.model)
mujoco.mj_resetData(spec.model, data)

with mujoco.viewer.launch_passive(spec.model, data) as v:
    while v.is_running():
        mujoco.mj_step(spec.model, data)
        v.sync()
=======
"""View the inspection pick-and-place scene in the MuJoCo interactive viewer.

Usage::

    python examples/view_inspection_scene.py
    python examples/view_inspection_scene.py --defective 0 1 0
    python examples/view_inspection_scene.py --defective 1 1 --no-robot

Controls:
    Space       pause / resume simulation
    Ctrl+Q      quit
    Right-click drag to rotate camera, scroll to zoom.
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer

from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import (
    make_inspection_scene,
)

parser = argparse.ArgumentParser(description="View the inspection MuJoCo scene")
parser.add_argument(
    "--defective",
    type=int,
    nargs="+",
    default=[0, 1, 0],
    metavar="0_OR_1",
    help="Ground-truth defect flag per block (0=clean, 1=defective). Default: 0 1 0",
)
parser.add_argument(
    "--no-robot",
    action="store_true",
    help="Show workspace only, without the Gen3 arm.",
)
args = parser.parse_args()


def main() -> None:
    block_defective = [bool(v) for v in args.defective]
    include_robot = not args.no_robot

    spec = make_inspection_scene(block_defective, include_robot=include_robot)
    model = spec.model
    data = mujoco.MjData(model)

    if include_robot and model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    else:
        mujoco.mj_resetData(model, data)

    print(f"Scene: {len(block_defective)} block(s), robot={'yes' if include_robot else 'no'}")
    print(f"  nq={model.nq}  nu={model.nu}  nbody={model.nbody}")
    print("Opening viewer — Space to pause, Ctrl+Q to quit.")

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
>>>>>>> 2602e04 (Add robot to inspection scene)
