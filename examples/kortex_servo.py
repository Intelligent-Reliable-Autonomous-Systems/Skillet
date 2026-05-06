"""Run a tabletop S2R task."""

import argparse
import pathlib
from typing import TYPE_CHECKING

import torch

from skillet.agents import PolicyOverOptionsAgent
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import SkilletEnv
from skillet.policy import FixedSequencePolicy, RandomPolicy, TwistPidPosePolicy, TcpCartPolicy
from skillet.skill import ReachPoseSkill
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED
from skillet_tasks.kortex_tasks.factory import create_kortex_env

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Tick rate of the perception")
parser.add_argument("--task", type=str, default="Kortex-Gen3-v0", help="Kortex Environment")
parser.add_argument("--build_scene", type=argparse.BooleanOptionalAction, default=False, help="If to build the scene")
parser.add_argument("--reconstruction", type=str, choices=["sam", "april"], default="april")
parser.add_argument(
    "--perception", type=argparse.BooleanOptionalAction, default=False, help="If to run the perception pipeline"
)
parser.add_argument("--o3d", type=argparse.BooleanOptionalAction, default=False, help="If to visualize with open3d")
parser.add_argument(
    "--goal",
    type=str,
    default="Place the red block on the purple block and the green block on the red block and the yellow block on the blue block.",
    help="Natural language goal for the block scene.",
)
args_cli = parser.parse_args()


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""

    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "device": "cpu",
        "num_envs": args_cli.num_envs,
    }

    env = create_kortex_env(args_cli.task, env_cfg)
    env = SkilletEnv(env)
    env = BatchToSingleWrapper(env)
    env.reset()

    # arm_policy = TwistPidPosePolicy(env.batched_env.obs_spec_twist_tcp, env.batched_env.action_spec_twist_tcp)
    arm_policy = TcpCartPolicy(env.batched_env.obs_spec_tcp_cart, env.batched_env.action_spec_tcp_cart)

    # Skills
    skill_length = 1e9

    reach_pose_skill = ReachPoseSkill(name="reach_xyzrpy_skill", policy=arm_policy, length=skill_length)
    skills = [reach_pose_skill]

    # Parameters policy
    fixed_param_policy = FixedSequencePolicy(
        env.batched_env.obs_spec_twist_tcp,
        reach_pose_skill.params_spec,
        torch.as_tensor(
            [
                [0.4, 0.1, 0.2, 0.0, 0.7071, 0.7071, 0.0],
                [0.5, 0.1, 0.2, 0.0, 0.7071, 0.7071, 0.0],
                [0.5, -0.1, 0.2, 0.0, 0.7071, 0.7071, 0.0],
            ],
            device=env.batched_env.device,
        ),
    )

    # High-level policy
    options_spec = (
        SELECT_OPTIONS_SPEC_BATCHED.bind(n_options=len(skills))
        .with_n_envs(args_cli.num_envs)
        .replace(device=env.batched_env.device)
    )
    policy_over_options = RandomPolicy(env.batched_env.obs_spec_twist_tcp, options_spec)

    s2r_agent = PolicyOverOptionsAgent(
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    input("Press Enter to start the skill execution...\n")

    while True:
        with torch.inference_mode():
            env.reset()
            s2r_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")
            break


if __name__ == "__main__":
    main()
