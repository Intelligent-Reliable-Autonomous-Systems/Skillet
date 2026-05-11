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
    python examples/view_inspection_scene.py --skill          # run InspectSkill on block_0
    python examples/view_inspection_scene.py --skill --block block_1

Controls:
    Space       pause / resume simulation
    Ctrl+Q      quit
    Right-click drag to rotate camera, scroll to zoom.
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import (
    make_inspection_scene,
    CUBE_SIZE,
    ROBOT_BASE_WORLD_POS,
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
parser.add_argument(
    "--skill",
    action="store_true",
    help="Run InspectSkill on the target block while the viewer is open.",
)
parser.add_argument(
    "--block",
    default="block_0",
    help="Name of the block to inspect when --skill is used (default: block_0).",
)
args = parser.parse_args()


def _run_passive(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Physics-only viewer loop (no skill)."""
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


def _run_with_skill(spec, block_name: str) -> None:  # type: ignore[no-untyped-def]
    """Run InspectSkill on *block_name* while streaming each step to the viewer."""
    # Import here so the plain viewer path has no extra deps.
    from skillet.core.skill import SkillStatusCodes
    from skillet.scene.base import Scene
    from skillet.skill.high_level.inspect import InspectSkill, _xyz_to_reach_params
    from skillet.skill.skill_lib import make_reach_xyzrpy_skill
    from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.env import InspectionMjEnv

    env = InspectionMjEnv(spec)
    env.reset()

    scene = Scene(objects=[spec.table, *spec.blocks, spec.platform, spec.discard])
    try:
        (block_id,) = scene.resolve_names_to_ids([block_name])
    except (ValueError, KeyError):
        raise SystemExit(f"Block {block_name!r} not found in scene. Available: "
                         + ", ".join(o.name for o in spec.blocks))

    reach_skill = make_reach_xyzrpy_skill(env, skill_length=300)
    skill = InspectSkill(
        scene,
        env=env,
        reach_skill=reach_skill,
        block_half_extents=np.full(3, CUBE_SIZE / 2.0),
        robot_base_world_pos=np.array(ROBOT_BASE_WORLD_POS),
    )
    skill.set_target(block_id)

    if not skill.preconditions(scene):
        raise SystemExit("InspectSkill preconditions not met — is the block pose known and x > 0?")

    viewpoint = skill._compute_viewpoint(scene)
    params = _xyz_to_reach_params(viewpoint)
    obs = env.get_observation()
    reach_skill.initiate(obs, params)

    print(f"Running InspectSkill → target TCP (base frame): {viewpoint.round(3)}")
    print("Viewer open — Ctrl+Q to quit at any time.")

    with mujoco.viewer.launch_passive(env._model, env._data) as v:
        # Step the skill loop, syncing the viewer after each env.step().
        for step in range(300):
            if not v.is_running():
                break
            obs = env.get_observation()
            action = reach_skill.get_action(obs)
            env.step(action, None)
            v.sync()
            status = int(reach_skill.status[0].item())
            if status == SkillStatusCodes.SUCCESS:
                tcp = env.get_observation()["tcp_pose_b"][0, :3].cpu().numpy()
                dist = float(np.linalg.norm(tcp - viewpoint))
                print(f"SUCCESS at step {step} — TCP {tcp.round(4)}, dist={dist:.4f} m")
                break
            if status == SkillStatusCodes.FAILED:
                print(f"FAILED at step {step}.")
                break
        else:
            print("TIMEOUT (300 steps).")

        # Keep viewer open so the final pose can be inspected.
        while v.is_running():
            mujoco.mj_step(env._model, env._data)
            v.sync()


def main() -> None:
    block_defective = [bool(v) for v in args.defective]
    include_robot = not args.no_robot

    if args.skill and not include_robot:
        raise SystemExit("--skill requires a robot (remove --no-robot).")

    spec = make_inspection_scene(block_defective, include_robot=include_robot)

    print(f"Scene: {len(block_defective)} block(s), robot={'yes' if include_robot else 'no'}")
    print(f"  nq={spec.model.nq}  nu={spec.model.nu}  nbody={spec.model.nbody}")

    if args.skill:
        _run_with_skill(spec, args.block)
    else:
        model = spec.model
        data = mujoco.MjData(model)
        if include_robot and model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        else:
            mujoco.mj_resetData(model, data)
        print("Opening viewer — Space to pause, Ctrl+Q to quit.")
        _run_passive(model, data)


if __name__ == "__main__":
    main()
>>>>>>> 2602e04 (Add robot to inspection scene)
