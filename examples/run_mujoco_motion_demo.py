"""Inspection pick-and-place demo with physical arm motion in MuJoCo.

Builds the Gen3 scene, runs PDDL planning, and executes each action via the
skill motion interface.  Pass ``--viewer`` to watch the arm move in real time;
without it the simulation runs headless.

Usage::

    python examples/run_mujoco_motion_demo.py
    python examples/run_mujoco_motion_demo.py --defective 0 1 0 --viewer
    python examples/run_mujoco_motion_demo.py --defective 0 1 0 --viewer --delay 0.01
    python examples/run_mujoco_motion_demo.py --log-dir /tmp/run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.env import InspectionMjEnv
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.orchestrator import run_demo
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import make_inspection_scene

parser = argparse.ArgumentParser(description="Inspection pick-and-place with arm motion")
parser.add_argument(
    "--defective",
    type=int,
    nargs="+",
    default=[0, 1, 0],
    metavar="0_OR_1",
    help="Ground-truth defect flag per block (0=clean, 1=defective). Default: 0 1 0",
)
parser.add_argument(
    "--log-dir",
    type=str,
    default="/tmp/run",
    help="Directory where events.jsonl is written.",
)
parser.add_argument(
    "--viewer",
    action="store_true",
    help="Open the MuJoCo passive viewer. Arm moves while viewer is open.",
)
parser.add_argument(
    "--delay",
    type=float,
    default=0.0,
    metavar="SECONDS",
    help="Sleep this many seconds after each simulation step (default 0). "
         "Use e.g. --delay 0.01 to slow motion for observation.",
)
args = parser.parse_args()


def main() -> None:
    block_defective = [bool(v) for v in args.defective]
    log_dir = Path(args.log_dir)

    print(f"Scene : {len(block_defective)} blocks, defective={block_defective}")
    print(f"Log   : {log_dir}")

    spec = make_inspection_scene(block_defective, include_robot=True)
    env = InspectionMjEnv(spec)
    env.reset()

    handle = None
    if args.viewer:
        import mujoco.viewer
        handle = mujoco.viewer.launch_passive(env.mj_model, env.mj_data)
        delay_s = args.delay

        def _step_callback() -> None:
            handle.sync()
            if delay_s > 0.0:
                time.sleep(delay_s)

        env.set_step_callback(_step_callback)
        print("Viewer launched — close the window after the demo completes.")

    screenshot_dir = log_dir / "screenshots"
    input("Press Enter to start simulation...")

    try:
        metrics = run_demo(block_defective, log_dir=log_dir, env=env, screenshot_dir=screenshot_dir)
    finally:
        if handle is not None:
            env.set_step_callback(None)  # stop new syncs before teardown
            time.sleep(0.1)             # let any in-flight render call finish
            try:
                handle.close()
            except Exception:
                pass                    # viewer teardown can segfault on displays without GPU

    print(f"\nResults:")
    print(f"  Defect accuracy : {metrics.defect_accuracy:.1%}  ({metrics.n_correct_verdict}/{metrics.n_blocks})")
    print(f"  Routing accuracy: {metrics.routing_accuracy:.1%}  ({metrics.n_correct_route}/{metrics.n_blocks})")
    print(f"  Events log      : {log_dir / 'events.jsonl'}")
    print(f"  Wrist screenshots: {screenshot_dir}")


if __name__ == "__main__":
    main()
