"""Run a tabletop block stacking task."""

import argparse
from typing import TYPE_CHECKING

import torch

from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import RealsenseEnv, SkilletEnv
from skillet.policy import FixedSequencePolicy, RandomPolicy, TwistPidPosePolicy
from skillet.skill import ReachPoseSkill
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED, XYZ_QUAT_Params
from skillet_tasks.kortex_tasks.factory import create_kortex_env

if TYPE_CHECKING:
    from skillet.core import BatchedSkill
    from skillet.envs.specs import BxM_Action, IKEE_Obs

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
        env.reset()

    # Low-level policies
    arm_policy = TwistPidPosePolicy(env.obs_spec_twist_tcp, env.action_spec_twist_tcp)
    # Skills
    skill_length = 1e9
    # Skills
    reach_pose_skill = ReachPoseSkill(name="reach_pose_skill", policy=arm_policy, length=skill_length)
    skills: list[BatchedSkill[IKEE_Obs, BxM_Action, XYZ_QUAT_Params]] = [reach_pose_skill]

    # Parameters policy
    fixed_param_policy = FixedSequencePolicy(
        env.obs_spec_policy,
        reach_pose_skill.params_spec,
        torch.as_tensor(
            [
                [0.3, 0.2, 0.3, 0.0, 0.0, -1.0, 0.0],
                [0.3, 0.2, 0.05, 0.0, 0.0, -1.0, 0.0],
            ],
            device=env.device,
        ),
    )

    # High-level policy
    options_spec = (
        SELECT_OPTIONS_SPEC_BATCHED.bind(n_options=len(skills))
        .with_n_envs(args_cli.num_envs)
        .replace(device=env.device)
    )
    policy_over_options = RandomPolicy(env.obs_spec, options_spec)

    policy_over_options_agent = PolicyOverOptionsAgent(
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    if not args_cli.realsense_env:
        input("Press Enter to start the skill execution...")

    while True:
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")


if __name__ == "__main__":
    main()
