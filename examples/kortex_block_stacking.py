"""Run a tabletop block stacking task."""

import argparse
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch

from skillet.agents.policy_over_options import PolicyOverOptionsAgent, SelectedSkill
from skillet.core import ActionSpec, ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import RealsenseEnv, SkilletEnv
from skillet.logging import SkilletDataLogger
from skillet.perception import SkilletPerception
from skillet.policy import FixedSequencePolicy, TwistPidPosePolicy
from skillet.scene import EMPTY_SCENE, Open3DVisualizer
from skillet.skill import PickBlockSkill, PickSkill, PlaceBlockSkill, PlaceSkill, RotateBlockSkill, RotateYawSkill
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

args_cli = parser.parse_args()


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
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
    rgbd_spec: ObservationSpec[RGBD_Obs] = env.coerce_obs_spec("rgb-d")

    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=rgbd_spec,
        reconstructor="sam",
        poll_rate_hz=args_cli.poll_rate_hz,
        device=args_cli.device,
    )
    visualizer = Open3DVisualizer(scene, env)
    perception.set_visualizer(visualizer, segment_point_cloud=True)
    perception.run_thread()
    visualizer.run_thread()

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
    rotate_y_skill = RotateYawSkill(
        reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, lift_delta=0.04, length=skill_length
    )
    pick_block_skill = PickBlockSkill(perception.scene, pick_skill, vis_target_pos=visualizer.set_target_pos)
    place_block_skill = PlaceBlockSkill(perception.scene, place_skill, vis_target_pos=visualizer.set_target_pos)
    rotate_block_skill = RotateBlockSkill(perception.scene, rotate_y_skill, vis_target_pos=visualizer.set_target_pos)
    skills = [pick_block_skill, place_block_skill, rotate_block_skill]

    # High-level policy
    options_spec = ActionSpec[SelectedSkill](
        space=gym.spaces.Discrete(len(skills)),
        name="options",
        is_torch=True,
        is_batched=False,
    )
    policy_over_options = FixedSequencePolicy[Any, SelectedSkill](
        rgbd_spec,
        options_spec,
        torch.as_tensor(
            [0, 1, 0, 1],
            device=rgbd_spec.device,
            dtype=torch.int32,
        ),
    )
    fixed_param_policy = FixedSequencePolicy(
        rgbd_spec,
        pick_block_skill.params_spec,
        torch.as_tensor(
            [1, 2, 0, 1],
            device=rgbd_spec.device,
            dtype=torch.int32,
        ),
    )

    policy_over_options_agent = PolicyOverOptionsAgent(
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    # simulate environment
    abs_model = None  # AbstractModel(Path("skillet/scene/abstract/assets/3-block-table-restack.problem.pddl"))
    logger = SkilletDataLogger("data/test/", env, perception, abs_model)

    if not args_cli.realsense_env:
        input("Press Enter to start the skill execution...")

    while True:
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env, data_logger=logger)
            print("[INFO][Main] finished run of skill executor, resetting")


if __name__ == "__main__":
    main()
