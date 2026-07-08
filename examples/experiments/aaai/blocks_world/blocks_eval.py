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

parser.add_argument("--o3d", type=argparse.BooleanOptionalAction, default=True, help="If to visualize with open3d")
parser.add_argument("--task_file", type=str, required=True, help="Path to evaluation task file description")
parser.add_argument(
    "--vlm", type=argparse.BooleanOptionalAction, default=False, help="If to use the VLM for scene building"
)
parser.add_argument(
    "--domain_path",
    type=str,
    help="Path to .domain.pddl file",
)
parser.add_argument("--model_dir", type=str, default="default", help="Name of model used")
args_cli = parser.parse_args()


def main() -> None:
    with pathlib.Path(args_cli.task_file).open("r") as f:
        task_data = json.load(f)

    scene = load_scene(task_data["scene_name"])
    block_domain = args_cli.domain_path
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "device": "cuda",
        "num_envs": args_cli.num_envs,
        "base_apriltag_id": 1,
        "base_apriltag_pose": [0.14, -0.01, 0.0, 0.0, 0.0, 0.7071068, 0.7071068],
        "base_apriltag_fam": "tag36h11",
        "base_apriltag_size": 0.1,
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
        vis_perception=args_cli.o3d,
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
    pick_block_skill = PickBlock4Skill(scene, pick_skill, vis_target_pos=target_pose_func)
    place_block_skill = PlaceBlock4Skill(scene, place_skill, vis_target_pos=target_pose_func)
    ACTION_MAP = {"place_block": place_block_skill, "pick_block": pick_block_skill}

    pathlib.Path(f"{task_data['log_dir']}/{args_cli.model_dir}").mkdir(exist_ok=True, parents=True)
    scene.goal = task_data["pddl_goal"]
    print(scene.goal)
    print("[INFO] Warming up Perception...")
    time.sleep(3)

    if args_cli.vlm:
        input("Press Enter to start the scene building...\n")

        perception.task_instruction = task_data["nl_goal"]
        perception.build_scene = True
        time.sleep(4)
        with pathlib.Path(f"{task_data['log_dir']}/{args_cli.model_dir}/g_vlm.txt").open("w") as f:
            f.write(str(scene.goal))
        print(f"[INFO][VLM Goal]:\n{scene.goal}")

    tamp_agent = PlanningAgent(scene, abstract_model=abs_model, action_to_skill_map=ACTION_MAP)

    logger = SkilletDataLogger(
        f"{task_data['log_dir']}/{args_cli.model_dir}",
        env,
        scene,
        perception,
        abs_model,
        tamp_agent,
        obs_spec=rgbd_grip_spec,
        visualize=False,
    )
    input("Press Enter to start the planning and evaluation experiment...\n")

    logger.write_video = True
    logger.run_thread()

    env.reset()
    tamp_agent.execute(env, logger=logger)
    logger.save_video()
    print("[INFO][Main] finished experiment, exiting...")


if __name__ == "__main__":
    main()
