"""Run a tabletop S2R task."""

import argparse
import pathlib
from typing import TYPE_CHECKING

import torch

from skillet.agents import S2RAgent
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import SkilletEnv
from skillet.perception import SkilletPerception
from skillet.policy import FixedSequencePolicy, PidRlPolicy, RandomPolicy
from skillet.scene import EMPTY_SCENE, SIX_CUBE_APRIL_SCENE, Open3DVisualizer
from skillet.skill import ReachXYZRPYSkill
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED
from skillet_tasks.kortex_tasks.factory import create_kortex_env

if TYPE_CHECKING:
    from skillet.envs.specs import RGBD_Gripper_Obs

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Tick rate of the perception")
parser.add_argument("--task", type=str, default="Kortex-Gen3Lite-v0", help="Kortex Environment")
parser.add_argument("--build_scene", type=argparse.BooleanOptionalAction, default=False, help="If to build the scene")
parser.add_argument("--reconstruction", type=str, choices=["sam", "april"], default="sam")
parser.add_argument(
    "--perception", type=argparse.BooleanOptionalAction, default=False, help="If to run the perception pipeline"
)
parser.add_argument("--o3d", type=argparse.BooleanOptionalAction, default=False, help="If to visualize with open3d")
parser.add_argument(
    "--goal",
    type=str,
    default="Place the red block on the purple block and the green block on the red block and the yellow block on the blue block.",
    help="Natural language goal for the block scene.",
)
args_cli = parser.parse_args()


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    import pickle

    scene = SIX_CUBE_APRIL_SCENE if args_cli.reconstruction == "april" else EMPTY_SCENE
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "device": "cpu",
        "num_envs": args_cli.num_envs,
    }

    env = create_kortex_env(args_cli.task, env_cfg)
    env = SkilletEnv(env)
    env = BatchToSingleWrapper(env)
    env.reset()
    rgbd_grip_spec: ObservationSpec[RGBD_Gripper_Obs] = env.coerce_obs_spec("rgbd-gripper")

    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=rgbd_grip_spec,
        reconstructor=args_cli.reconstruction,
        poll_rate_hz=args_cli.poll_rate_hz,
        device="cuda",
        vis_perception=False,
    )
    target_pose_func = None
    if args_cli.o3d:
        visualizer = Open3DVisualizer(scene, env)
        perception.set_visualizer(visualizer, segment_point_cloud=True)
        visualizer.run_thread()
        target_pose_func = visualizer.set_target_pos
    if args_cli.perception:
        perception.run_thread()
    else:
        with pathlib.Path("data/test/vlm_out_multi.pkl").open("rb") as f:
            scene = pickle.load(f)
            b = scene.get_objects_from_name(["red_block"])[0]
            scene.tcp_pose = b.pose.clone()
            scene.gripper_pos = 0.8

    # Low-level policies
    # arm_policy = JointPosPidPosePolicy(env.batched_env.obs_spec_joints, env.batched_env.action_spec_joints)
    arm_policy = PidRlPolicy(
        env.batched_env.obs_spec_joints,
        env.batched_env.action_spec_joints,
        agent_fpath="data/rl/gen3lite_reach",
        poll_rate_hz=60,
    )
    # Skills
    skill_length = 1e9

    reach_pose_skill = ReachXYZRPYSkill(name="reach_xyzrpy_skill", policy=arm_policy, length=skill_length)
    skills = [reach_pose_skill]

    # Parameters policy
    fixed_param_policy = FixedSequencePolicy(
        env.batched_env.obs_spec_policy,
        reach_pose_skill.params_spec,  # TODO: fix this device mismatch
        torch.as_tensor(
            [
                [0.4, -0.1, 0.3, 0.0, 1.57, 0.0],
            ],
            device=env.batched_env.device,
        ),
    )

    # High-level policy
    options_spec = (
        SELECT_OPTIONS_SPEC_BATCHED.bind(n_options=len(skills))
        .with_n_envs(args_cli.num_envs)
        .replace(device=env.batched_env.device)
    )
    policy_over_options = RandomPolicy(env.batched_env.obs_spec_policy, options_spec)

    s2r_agent = S2RAgent(
        scene,
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    # simulate environment

    if args_cli.build_scene:
        input("Press Enter to start the scene building...\n")
        perception.task_instruction = args_cli.goal
        perception.build_scene = args_cli.build_scene

    # input("Press Enter to start the skill execution...\n")

    while True:
        with torch.inference_mode():
            env.reset()
            s2r_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")
            break


if __name__ == "__main__":
    main()
