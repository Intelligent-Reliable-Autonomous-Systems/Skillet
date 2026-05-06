"""Run a tabletop block stacking task."""

import argparse
import pathlib
from typing import TYPE_CHECKING

import torch

from skillet.agents import PlanningAgent
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import SkilletEnv
from skillet.logging import SkilletDataLogger
from skillet.perception.perception import SkilletPerception
from skillet.planning import AbstractModel
from skillet.policy import TcpCartPolicy, TwistPidPosePolicy
from skillet.scene import EMPTY_SCENE, SIX_CUBE_APRIL_SCENE, SIX_CUBE_SCENE, Open3DVisualizer
from skillet.skill import PickBlock2Skill, PickSkill, PlaceBlock2Skill, PlaceSkill
from skillet_tasks.kortex_tasks.factory import create_kortex_env

if TYPE_CHECKING:
    from skillet.envs.specs import RGBD_Gripper_Obs

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cpu", help="Device to use")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Tick rate of the perception")
parser.add_argument("--task", type=str, default="Kortex-Gen3-v0", help="Kortex Environment")
parser.add_argument("--build_scene", type=argparse.BooleanOptionalAction, default=False, help="If to build the scene")
parser.add_argument("--reconstruction", type=str, choices=["sam3", "april", "vlm", "sam+vlm"], default="sam3")
parser.add_argument(
    "--perception", type=argparse.BooleanOptionalAction, default=True, help="If to run the perception pipeline"
)
parser.add_argument("--o3d", type=argparse.BooleanOptionalAction, default=True, help="If to visualize with open3d")
parser.add_argument(
    "--goal",
    type=str,
    default="Place the red block on the yellow block.",
    help="Natural language goal for the block scene.",
)
args_cli = parser.parse_args()


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    import pickle

    if args_cli.reconstruction == "april":
        scene = SIX_CUBE_APRIL_SCENE
    elif args_cli.reconstruction == "sam+vlm" or args_cli.reconstruction == "vlm":
        scene = EMPTY_SCENE
    elif args_cli.reconstruction == "sam3":
        scene = SIX_CUBE_SCENE
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "device": "cuda",
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
        vis_perception=True,
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
    # arm_policy = TwistPidPosePolicy(env.batched_env.obs_spec_twist_tcp, env.batched_env.action_spec_twist_tcp)
    arm_policy = TcpCartPolicy(env.batched_env.obs_spec_tcp_cart, env.batched_env.action_spec_tcp_cart)
    # Skills
    skill_length = 1e9
    place_skill = PlaceSkill(reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, length=skill_length)
    pick_skill = PickSkill(reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, length=skill_length)

    pick_block_skill = PickBlock2Skill(scene, pick_skill, vis_target_pos=target_pose_func)
    place_block_skill = PlaceBlock2Skill(scene, place_skill, vis_target_pos=target_pose_func)

    ACTION_MAP = {"place_block": place_block_skill, "pick_block": pick_block_skill}
    block_domain = "skillet/planning/abstract/assets/blocks.domain.pddl"
    block_task = None  # "skillet/scene/abstract/assets/3-block-table.problem.pddl"

    abs_model = AbstractModel(block_domain, block_task, scene)
    planning_agent = PlanningAgent(scene, abstract_model=abs_model, action_to_skill_map=ACTION_MAP)

    # simulate environment
    logger = SkilletDataLogger(
        "data/test/", env, scene, perception, abs_model, planning_agent, obs_spec=rgbd_grip_spec, visualize=False
    )
    if args_cli.build_scene:
        input("Press Enter to start the scene building...\n")
        perception.task_instruction = args_cli.goal
        perception.build_scene = args_cli.build_scene

    input("Press Enter to start the skill execution...\n")
    logger.write_video = True
    logger.run_thread()

    while True:
        with torch.inference_mode():
            env.reset()
            planning_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")
            logger.save_video()
            break


if __name__ == "__main__":
    main()
