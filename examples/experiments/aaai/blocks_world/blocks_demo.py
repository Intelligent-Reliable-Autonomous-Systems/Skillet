"""Run a tabletop block stacking task."""

import argparse
import json
import pathlib
import time
from typing import TYPE_CHECKING

from skillet.agents import PlanningAgent
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import SkilletEnv
from skillet.envs.realsense import RealsenseEnv
from skillet.envs.specs import NULL_ACTION_SPEC
from skillet.logging import SkilletDataLogger
from skillet.perception.perception import SkilletPerception
from skillet.planning import AbstractModel
from skillet.scene import (
    Open3DVisualizer,
    load_scene,
)
from skillet.skill.high_level import (
    PickSkill,
    PlaceSkill,
)
from skillet.skill.object_level import (
    PickBlock4Skill,
    PlaceBlock4Skill,
)
from skillet.skill.policy import TcpCartPolicy
from skillet_tasks.kortex_tasks.factory import create_kortex_env

if TYPE_CHECKING:
    from skillet.envs.specs import RGBD_Gripper_Obs

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cpu", help="Device to use")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Tick rate of the perception")
parser.add_argument("--task", type=str, default="Kortex-Gen3-v0", help="Kortex Environment")

parser.add_argument("--o3d", action=argparse.BooleanOptionalAction, default=True, help="If to visualize with open3d")
parser.add_argument(
    "--vlm", type=argparse.BooleanOptionalAction, default=False, help="If to use the VLM for scene building"
)
parser.add_argument(
    "--domain_path",
    type=str,
    default="skillet_tasks/blocksworld-pick-place/eval/domains/default/simple-blocksworld-pick-place.domain.pddl",
    help="Path to .domain.pddl file",
)
parser.add_argument("--model_dir", type=str, default="default", help="Name of model used")
args_cli = parser.parse_args()


def main() -> None:

    scene = load_scene("5cube_3dots")
    block_domain = args_cli.domain_path
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "device": "cuda",
        "num_envs": args_cli.num_envs,
        "base_apriltag_id": 1,
        "base_apriltag_pose": [0.14, -0.01, 0.0, 0.0, 0.0, 0.7071068, -0.7071068],
        "base_apriltag_fam": "tag36h11",
        "base_apriltag_size": 0.1,
    }

    env = create_kortex_env(args_cli.task, env_cfg)
    env = SkilletEnv(env)
    env = BatchToSingleWrapper(env)
    env.reset()
    rgbd_grip_spec: ObservationSpec[RGBD_Gripper_Obs] = env.coerce_obs_spec("rgbd-gripper")
    # rgbd_grip_spec: ObservationSpec[RGBD_Gripper_Obs] = env.coerce_obs_spec("rgbd-gripper")
    # env = RealsenseEnv(
    #     apriltag_id=1,
    #     apriltag_pose=[0.14, -0.01, 0.0, 0.0, 0.0, 0.7071068, -0.7071068],
    #     apriltag_fam="tag36h11",
    #     apriltag_size_m=0.1,
    # )
    # rgbd_grip_spec: ObservationSpec[RGBD_Gripper_Obs] = env.obs_spec

    abs_model = AbstractModel(block_domain, scene=scene)

    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=rgbd_grip_spec,
        abstract_model=abs_model,
        reconstructor="sam3",
        poll_rate_hz=args_cli.poll_rate_hz,
        device="cuda",
        vis_perception=args_cli.o3d,
    )
    target_pose_func = None
    if args_cli.o3d:
        visualizer = Open3DVisualizer(scene, env)
        perception.set_visualizer(visualizer, segment_point_cloud=True)
        visualizer.run_thread()
        target_pose_func = visualizer.set_target_pos
    perception.run_thread()

    # env.reset()
    # while True:
    #     obs = env.get_observation(env.coerce_obs_spec("ik_ee"))
    #     print(obs)
    #     env.step(NULL_ACTION_SPEC.unbatched().cast([]), NULL_ACTION_SPEC.unbatched())
    # Low-level policies
    skill_length = 1e9
    arm_policy = TcpCartPolicy(env.batched_env.obs_spec_tcp_cart, env.batched_env.action_spec_tcp_cart)
    place_skill = PlaceSkill(reach_policy=arm_policy, lift_height=0.21, gripper_close=0.6, length=skill_length)
    pick_skill = PickSkill(reach_policy=arm_policy, lift_height=0.21, gripper_close=0.6, length=skill_length)
    pick_block_skill = PickBlock4Skill(scene, pick_skill, vis_target_pos=target_pose_func)
    place_block_skill = PlaceBlock4Skill(scene, place_skill, vis_target_pos=target_pose_func)
    ACTION_MAP = {"place_block": place_block_skill, "pick_block": pick_block_skill}

    tamp_agent = PlanningAgent(scene, abstract_model=abs_model, action_to_skill_map=ACTION_MAP)
    while True:
        goal = input("Enter the goal ('exit' to quit): ")
        if goal == "exit":
            break
        # scene.goal = goal
        # print(scene.goal)

        input("Press Enter to start the planning and evaluation experiment...\n")

        env.reset()
        tamp_agent.execute(env, task_pddl=goal)
        print("[INFO][Main] finished experiment, exiting...")


if __name__ == "__main__":
    main()
