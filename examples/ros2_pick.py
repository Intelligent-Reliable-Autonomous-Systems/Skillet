"""ros2_ik.py.

A demonstration of the ROS2 pick skill.

Written by Will Solow and Jeff Jewett, 2026

"""

import argparse
import os
from typing import TYPE_CHECKING

import torch

import skillet_tasks.ros2_tasks  # noqa: F401
from skillet.agents.policy_over_options import PolicyOverOptionsBatchedAgent
from skillet.envs.skillet_env import SkilletEnv
from skillet.policy.dummy import FixedSequencePolicy, RandomPolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy
from skillet.policy.twist import TwistPIDPosePolicy
from skillet.skill.high_level.pick import PickSkill
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED, XYZ_YAW_Params
from skillet_tasks.ros2_tasks.factory import create_ros2_env

if TYPE_CHECKING:
    from skillet.core import BatchedSkill
    from skillet.envs.specs import BxM_Action, IKEE_Obs

# add argparse arguments
parser = argparse.ArgumentParser(description="Main ROS2 executor file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ROS2-Gen3Lite-v0", help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument(
    "--launch_ros", action=argparse.BooleanOptionalAction, default=False, help="Launch ROS from env startup."
)

# parse the arguments
args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


def main() -> None:
    """Run the ROS2 pick example."""
    # create environment
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "launch_ros": args_cli.launch_ros,
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
        "ros2_workspace": args_cli.ros2_ws,
        "use_fake_hardware": True,
    }
    env = create_ros2_env(args_cli.task, env_cfg)

    env = SkilletEnv(env)
    env.reset()

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Low-level policies
    ik_ee_pose_policy = PoseAbsIKEEPolicy(env.obs_spec_ikee, env.action_spec)
    ik_ee_pose_policy = TwistPIDPosePolicy(env.obs_spec_twist_tcp, env.action_spec_twist_tcp)
    # Skills
    skill_length = 1e9
    pick_skill = PickSkill(reach_policy=ik_ee_pose_policy, gripper_policy=None, lift_height=0.23, length=skill_length)
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

    env.close()


if __name__ == "__main__":
    main()
