"""Run a tabletop block stacking task."""

import argparse
import time
from typing import TYPE_CHECKING

from skillet.agents import RandomTampAgent
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import SkilletEnv
from skillet.logging import SkilletDataLogger
from skillet.perception.perception import SkilletPerception
from skillet.planning import AbstractModel
from skillet.scene import (
    sponge_scene_loader,
    Open3DVisualizer,
)
from skillet.skill.high_level import (
    PickSkill,
    PlaceSkill,
    SqueezeSkill,
    WipeSkill,
)
from skillet.skill.object_level import (
    PickBlock2Skill,
    PlaceBlock3Skill,
    SqueezeSpongeSkill,
    WipeTableSkill,
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
parser.add_argument("--o3d", type=argparse.BooleanOptionalAction, default=False, help="If to visualize with open3d")

args_cli = parser.parse_args()


def main() -> None:
    scene = sponge_scene_loader()
    block_domain = "skillet_tasks/pddl_tasks/sponge-world/simple-sponge.domain.pddl"
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "device": "cuda",
        "num_envs": args_cli.num_envs,
        "base_apriltag_id": 0,
        "base_apriltag_pose": [0.13, -0.02, 0.0, 0.0, 0.0, 0.7071068, 0.7071068],
        "base_apriltag_fam": "tag16h5",
        "base_apriltag_size": 0.09,
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
        vis_perception=True,
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
    place_skill = PlaceSkill(reach_policy=arm_policy, lift_height=0.25, gripper_close=0.6, length=skill_length)
    pick_skill = PickSkill(reach_policy=arm_policy, lift_height=0.25, gripper_close=0.6, length=skill_length)
    squeeze_skill = SqueezeSkill(
        reach_policy=arm_policy, lift_height=0.25, gripper_close=0.6, timeout=5, length=skill_length
    )
    wipe_skill = WipeSkill(reach_policy=arm_policy, lift_height=0.25, gripper_close=0.6, length=skill_length)
    pick_obj_skill = PickBlock2Skill(scene, pick_skill, vis_target_pos=target_pose_func)
    place_obj_skill = PlaceBlock3Skill(scene, place_skill, vis_target_pos=target_pose_func)
    wipe_table_skill = WipeTableSkill(scene, wipe_skill, vis_target_pos=target_pose_func)
    squeeze_sponge_skill = SqueezeSpongeSkill(scene, squeeze_skill, vis_target_pos=target_pose_func)
    ACTION_MAP = {
        "place_moveable": place_obj_skill,
        "pick_movable": pick_obj_skill,
        "squeeze_movable": squeeze_sponge_skill,
        "wipe_movable": wipe_table_skill,
    }

    tamp_agent = RandomTampAgent(scene, abstract_model=abs_model, action_to_skill_map=ACTION_MAP)

    logger = SkilletDataLogger(
        "_robot_data/exp/", env, scene, perception, abs_model, tamp_agent, obs_spec=rgbd_grip_spec, visualize=False
    )

    print("[INFO] Warming up Perception...")
    time.sleep(5)
    logger.write_video = True
    logger.run_thread()

    env.reset()
    tamp_agent.execute(env, logger=logger, num_actions=100)
    logger.save_video()
    print("[INFO][Main] finished experiment, exiting...")


if __name__ == "__main__":
    main()
