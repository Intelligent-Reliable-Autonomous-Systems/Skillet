"""Run a tabletop block stacking task."""

import argparse
import json
import time
from typing import TYPE_CHECKING

from skillet.agents import PlanningAgent
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import SkilletEnv
from skillet.logging import SkilletDataLogger
from skillet.perception.perception import SkilletPerception
from skillet.planning import AbstractModel
from skillet.policy import TcpCartPolicy
from skillet.scene import (
    FIVE_CUBE_SCENE,
    Open3DVisualizer,
)
from skillet.skill import (
    DragBlock5Skill,
    DragSkill,
    PickBlock4Skill,
    PickSkill,
    PlaceBlock4Skill,
    PlaceSkill,
)
from skillet_tasks.kortex_tasks.factory import create_kortex_env

if TYPE_CHECKING:
    from skillet.envs.specs import RGBD_Gripper_Obs

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cpu", help="Device to use")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Tick rate of the perception")
parser.add_argument("--task", type=str, default="Kortex-Gen3-v0", help="Kortex Environment")

parser.add_argument("--o3d", type=argparse.BooleanOptionalAction, default=False, help="If to visualize with open3d")
parser.add_argument("--eval_dir", type=str, default="_robot_data/blocks_eval_tasks/task_4", help="Evaluation directory")
parser.add_argument(
    "--vlm", type=argparse.BooleanOptionalAction, default=False, help="If to use the VLM for scene building"
)
parser.add_argument(
    "--domain_path",
    default="skillet_tasks/blocks-world/simple-blocks-a3.domain.pddl",
    type=str,
    help="Path to .domain.pddl file",
)
args_cli = parser.parse_args()


def main() -> None:
    scene = FIVE_CUBE_SCENE
    block_domain = args_cli.domain_path
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

    abs_model = AbstractModel(block_domain, None, scene)

    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=rgbd_grip_spec,
        abstract_model=abs_model,
        reconstructor="sam3",
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
    perception.run_thread()

    # Low-level policies
    skill_length = 1e9
    arm_policy = TcpCartPolicy(env.batched_env.obs_spec_tcp_cart, env.batched_env.action_spec_tcp_cart)
    place_skill = PlaceSkill(reach_policy=arm_policy, lift_height=0.21, gripper_close=0.6, length=skill_length)
    pick_skill = PickSkill(reach_policy=arm_policy, lift_height=0.21, gripper_close=0.6, length=skill_length)
    drag_skill = DragSkill(reach_policy=arm_policy, lift_height=0.21, gripper_close=0.6, length=skill_length)
    pick_block_skill = PickBlock4Skill(scene, pick_skill, vis_target_pos=target_pose_func)
    place_block_skill = PlaceBlock4Skill(scene, place_skill, vis_target_pos=target_pose_func)
    drag_block_skill = DragBlock5Skill(scene, drag_skill, vis_target_pos=target_pose_func)
    ACTION_MAP = {"place_block": place_block_skill, "pick_block": pick_block_skill, "drag_block": drag_block_skill}

    with open(f"{args_cli.eval_dir}/g_pddl.txt") as f:
        pddl_goal = json.load(f)
    print(pddl_goal)
    scene.goal = pddl_goal
    print("[INFO] Warming up Perception...")
    time.sleep(5)

    if args_cli.vlm:
        input("Press Enter to start the scene building...\n")
        with open(f"{args_cli.eval_dir}/g_nl.txt") as f:
            goal_txt = f.read()

        perception.task_instruction = goal_txt
        perception.build_scene = True
        time.sleep(4)
        with open(f"{args_cli.eval_dir}/g_vlm.txt", "w") as f:
            f.write(str(scene.goal))
        print(f"[INFO][VLM Goal]:\n{scene.goal}")

    input("Press Enter to start the plan execution...\n")
    tamp_agent = PlanningAgent(scene, abstract_model=abs_model, action_to_skill_map=ACTION_MAP)

    # simulate environment
    logger = SkilletDataLogger(
        args_cli.eval_dir, env, scene, perception, abs_model, tamp_agent, obs_spec=rgbd_grip_spec, visualize=False
    )
    logger.write_video = True
    logger.run_thread()
    env.reset()
    tamp_agent.execute(env, logger=logger)
    logger.save_video()


if __name__ == "__main__":
    main()
