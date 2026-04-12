"""Run a tabletop block stacking task."""

import argparse
from typing import TYPE_CHECKING

import torch

from skillet.agents import PlanningAgent
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import RealsenseEnv, SkilletEnv
from skillet.logging import SkilletDataLogger
from skillet.perception import SkilletPerception
from skillet.policy import TwistPidPosePolicy
from skillet.scene import EMPTY_SCENE, Open3DVisualizer
from skillet.scene.abstract.abstract_model import AbstractModel
from skillet.skill import PickBlock2Skill, PickSkill, PlaceBlock2Skill, PlaceSkill
from skillet_tasks.kortex_tasks.factory import create_kortex_env

if TYPE_CHECKING:
    from skillet.envs.specs import RGBD_Obs

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--realsense_env", action=argparse.BooleanOptionalAction, default=False, help="Use RealSense camera environment."
)
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Seconds between service requests.")
parser.add_argument("--task", type=str, default="Kortex-Gen3Lite-v0", help="Kortex Environment")
parser.add_argument("--build_scene", type=argparse.BooleanOptionalAction, default=True, help="If to build the scene")
parser.add_argument("--perception", type=argparse.BooleanOptionalAction, default=True, help="If to build the scene")
parser.add_argument(
    "--goal",
    type=str,
    default="Move the red block onto the purple block",
    help="Natural language goal for the block scene.",
)
args_cli = parser.parse_args()


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    import pickle

    scene = EMPTY_SCENE
    if args_cli.realsense_env:
        env = RealsenseEnv(apriltag_size_m=0.1, apriltag_id=3)
    else:
        env_cfg = {
            "robot_ip": args_cli.robot_ip,
            "device": args_cli.device,
            "num_envs": args_cli.num_envs,
        }

        env = create_kortex_env(args_cli.task, env_cfg)
        env = SkilletEnv(env)
        env = BatchToSingleWrapper(env)
        env.reset()
    rgbd_grip_spec: ObservationSpec[RGBD_Obs] = env.coerce_obs_spec("rgbd-gripper")

    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=rgbd_grip_spec,
        reconstructor="sam",
        poll_rate_hz=args_cli.poll_rate_hz,
        device=args_cli.device,
    )
    visualizer = Open3DVisualizer(scene, env)
    perception.set_visualizer(visualizer, segment_point_cloud=True)
    if args_cli.perception:
        perception.run_thread()
        visualizer.run_thread()
    else:
        with open("data/test/vlm_scene_2.pkl", "rb") as f:
            scene = pickle.load(f)

    import time

    if args_cli.realsense_env:
        while True:
            perception.run()
            time.sleep(0.2)

    # Low-level policies
    arm_policy = TwistPidPosePolicy(env.batched_env.obs_spec_twist_tcp, env.batched_env.action_spec_twist_tcp)
    # Skills
    skill_length = 1e9
    place_skill = PlaceSkill(reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, length=skill_length)
    pick_skill = PickSkill(reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, length=skill_length)

    pick_block_skill = PickBlock2Skill(scene, pick_skill, vis_target_pos=visualizer.set_target_pos)
    place_block_skill = PlaceBlock2Skill(scene, place_skill, vis_target_pos=visualizer.set_target_pos)

    ACTION_MAP = {"place_block": place_block_skill, "pick_block": pick_block_skill}
    block_domain = "skillet/scene/abstract/assets/blocks.domain.pddl"
    block_task = None  # "skillet/scene/abstract/assets/3-block-table.problem.pddl"

    abs_model = AbstractModel(block_domain, block_task, scene)
    planning_agent = PlanningAgent(scene, abstract_model=abs_model, action_to_skill_map=ACTION_MAP)

    # simulate environment
    logger = SkilletDataLogger("data/test/", env, perception, abs_model)

    if not args_cli.realsense_env:
        input("Press Enter to start the scene building...")
        perception.build_scene = args_cli.build_scene

    if not args_cli.realsense_env:
        input("Press Enter to start the skill execution...")

    while True:
        with torch.inference_mode():
            env.reset()
            planning_agent.execute(env, data_logger=logger)
            print("[INFO][Main] finished run of skill executor, resetting")


if __name__ == "__main__":
    main()
