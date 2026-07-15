"""Run a tabletop block stacking task."""

import argparse
import time
from typing import TYPE_CHECKING

from conditional_repair.baselines.online.random_agent import RandomAgent
from conditional_repair.dataset import RepairDataset

from skillet.agents import ActiveLearningAgent
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
from skillet.skill.high_level import PickSkill, PlaceSkill, WipeSkill
from skillet.skill.object_level import (
    PickObj2Skill,
    PlaceObj2Skill,
    WipeSurface3Skill,
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
parser.add_argument(
    "--log_dir",
    default="_robot_data/exp/",
    type=str,
)
parser.add_argument(
    "--exp_config",
    type=str,
    default="skillet_tasks/spongeworld-clean/repair-sponge/repair-effects-random.json",
    help="Path to experiment JSON file",
)
parser.add_argument("--scene_name", type=str, default="2sponge_1plate", help="What scene to load")

args_cli = parser.parse_args()


def main() -> None:
    scene = load_scene(args_cli.scene_name)
    domain = "skillet_tasks/spongeworld-clean/simple-sponge.domain.pddl"
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

    abs_model = AbstractModel(domain, None, scene, domain="sponge")

    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=rgbd_grip_spec,
        abstract_model=abs_model,
        reconstructor="sam3",
        poll_rate_hz=args_cli.poll_rate_hz,
        device="cuda",
        vis_perception=args_cli.o3d,
        domain="sponge",
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
    place_skill = PlaceSkill(
        reach_policy=arm_policy, lift_height=0.21, gripper_close=0.6, length=skill_length, default_quat=[[0, 1, 0, 0]]
    )
    pick_skill = PickSkill(
        reach_policy=arm_policy, lift_height=0.21, gripper_close=0.6, length=skill_length, default_quat=[[0, 1, 0, 0]]
    )
    wipe_skill = WipeSkill(reach_policy=arm_policy, lift_height=0.21, gripper_close=0.9, length=skill_length)
    pick_obj_skill = PickObj2Skill(scene, pick_skill, vis_target_pos=target_pose_func, xyz_offset=(0, 0, 0.04))
    place_obj_skill = PlaceObj2Skill(scene, place_skill, vis_target_pos=target_pose_func)
    wipe_obj_skill = WipeSurface3Skill(scene, wipe_skill, vis_target_pos=target_pose_func)
    ACTION_MAP = {
        "place": place_obj_skill,
        "pick": pick_obj_skill,
        "wipe": wipe_obj_skill,
    }
    print("[INFO] Warming up Perception...")
    time.sleep(3)

    learning_agent = RandomAgent(RepairDataset(args_cli.exp_config))

    tamp_agent = ActiveLearningAgent(
        scene,
        abstract_model=abs_model,
        action_to_skill_map=ACTION_MAP,
        learning_agent=learning_agent,
    )

    logger = SkilletDataLogger(
        args_cli.log_dir, env, scene, perception, abs_model, tamp_agent, obs_spec=rgbd_grip_spec, visualize=False
    )
    input("Press Enter to start the active learning experiment...")

    logger.write_video = True
    logger.run_thread()

    env.reset()
    tamp_agent.execute(env, logger=logger)
    logger.save_video()
    print("[INFO][Main] finished experiment, exiting...")


if __name__ == "__main__":
    main()
