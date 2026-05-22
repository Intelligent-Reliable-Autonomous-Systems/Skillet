"""ros2_ik.py.

A demonstration of the ROS2 pick skill.

Written by Will Solow and Jeff Jewett, 2026

"""

import argparse
from typing import TYPE_CHECKING

import torch

import skillet_tasks.ros2_tasks.ros2_tasks  # noqa: F401
from skillet.agents.policy_over_options import PolicyOverOptionsBatchedAgent
from skillet.envs.skillet_env import SkilletEnv
from skillet.skill.high_level import PickSkill
from skillet.skill.policy import FixedSequencePolicy, PoseAbsIkEePolicy, RandomPolicy, TwistPidPosePolicy
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED, XYZ_YAW_Params
from skillet_tasks.ros2_tasks.ros2_tasks.factory import create_ros2_env

if TYPE_CHECKING:
    from skillet.core import BatchedSkill
    from skillet.envs.specs import BxM_Action, IKEE_Obs

# add argparse arguments
parser = argparse.ArgumentParser(description="Main ROS2 executor file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Ros2-Gen3Lite-v0", help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")

# parse the arguments
args_cli = parser.parse_args()


def main() -> None:
    """Run the ROS2 pick example."""
    # create environment
    env_cfg = {
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
    }
    env = create_ros2_env(args_cli.task, env_cfg)

    env = SkilletEnv(env)
    env.reset()

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Low-level policies
    ik_ee_pose_policy = PoseAbsIkEePolicy(env.obs_spec_ikee, env.action_spec)
    ik_ee_pose_policy = TwistPidPosePolicy(env.obs_spec_twist_tcp, env.action_spec_twist_tcp)
    # Skills
    skill_length = 1e9
    pick_skill = PickSkill(reach_policy=ik_ee_pose_policy, gripper_policy=None, lift_height=0.3, length=skill_length)
    skills: list[BatchedSkill[IKEE_Obs, BxM_Action, XYZ_YAW_Params]] = [pick_skill]

    # Parameters policy
    fixed_param_policy = FixedSequencePolicy(
        env.obs_spec,
        pick_skill.params_spec,
        torch.as_tensor(
            [
                [0.3, -0.2, 0.03, 0.0],
                [0.3, 0.2, 0.05, 0.0],
                [0.2, 0.4, 0.05, 0.0],
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

    policy_over_options_agent = PolicyOverOptionsBatchedAgent(
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")


if __name__ == "__main__":
    main()
