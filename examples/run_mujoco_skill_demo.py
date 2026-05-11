"""Inspection pick-and-place demo

The MuJoCo model is loaded and validated (geometry, textures, free joints).
Arm motion control is not yet implemented; this script uses the
simplified skill execute() interface with MockDefectClassifier.

Usage::

    python examples/run_mujoco_skill_demo.py
    python examples/run_mujoco_skill_demo.py --defective 0 1 0
    python examples/run_mujoco_skill_demo.py --log-dir /tmp/run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.orchestrator import run_demo

parser = argparse.ArgumentParser(description="Inspection pick-and-place")
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
args = parser.parse_args()


def main() -> None:
    block_defective = [bool(v) for v in args.defective]
    log_dir = Path(args.log_dir)

    print(f"Scene: {len(block_defective)} blocks, defective={block_defective}")
    print(f"Log dir: {log_dir}")

    metrics = run_demo(block_defective, log_dir=log_dir)

    print(f"\nResults:")
    print(f"  Defect detection accuracy : {metrics.defect_accuracy:.1%}  ({metrics.n_correct_verdict}/{metrics.n_blocks})")
    print(f"  Routing accuracy          : {metrics.routing_accuracy:.1%}  ({metrics.n_correct_route}/{metrics.n_blocks})")
    print(f"  Events log                : {log_dir / 'events.jsonl'}")


if __name__ == "__main__":
    main()