"""Run a tabletop block stacking task."""

import argparse
import time
from typing import TYPE_CHECKING

import gymnasium as gym

import skillet_tasks.mj_tasks  # noqa: F401
from skillet.agents import RandomTampAgent
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import SkilletEnv
from skillet.envs.util import parse_mj_env_cfg
from skillet.logging import SkilletDataLogger
from skillet.perception.perception import SkilletPerception
from skillet.planning import AbstractModel
from skillet.scene import (
    ONE_CUBE_SCENE,
    Open3DVisualizer,
)
from skillet.skill.high_level import (
    DragSkill,
    PickSkill,
    PlaceSkill,
)
from skillet.skill.object_level import (
    DragBlock5Skill,
    PickBlock4Skill,
    PlaceBlock4Skill,
)
from skillet.skill.policy import TcpCartPolicy

if TYPE_CHECKING:
    from skillet.envs.specs import RGBD_Gripper_Obs

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Tick rate of the perception")
parser.add_argument("--task", type=str, default="Mj-Gen3-v0", help="Kortex Environment")
parser.add_argument("--o3d", type=argparse.BooleanOptionalAction, default=True, help="If to visualize with open3d")

args_cli = parser.parse_args()


def main() -> None:
    scene = ONE_CUBE_SCENE
    block_domain = "skillet_tasks/pddl_tasks/blocks-world/simple-blocks-a3.domain.pddl"
    env_cfg = parse_mj_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
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
        max_depth_m=5,
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
