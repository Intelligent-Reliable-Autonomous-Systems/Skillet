"""ros2_ik.py.

Test file for executor integration ROS2 skills

Written by Will Solow and Jeff Jewett, 2026

"""

import argparse
import os
from typing import TYPE_CHECKING

import gymnasium as gym
import torch

if TYPE_CHECKING:
    from skillet.core import BatchedSkill
    from skillet.envs.specs import BxM_Action, IKEE_Obs

import skillet_tasks.ros2_tasks  # noqa: F401
from skillet.agents.policy_over_options import PolicyOverOptionsBatchedAgent
from skillet.envs.ros2_skillet_env import ROS2SkilletEnv
from skillet.envs.util import parse_ros2_env_cfg, setup_ros
from skillet.policy.dummy import FixedSequencePolicy, RandomPolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy
from skillet.skill import ReachPoseSkill
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED, XYZ_QUAT_Params

# add argparse arguments
parser = argparse.ArgumentParser(description="Main ROS2 executor file.")
parser.add_argument("--num_envs", type=int, default=1, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ROS2-Reach-Gen3-v0", required=True, help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, required=False, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--robot_ip", default="192.168.8.10", type=str, help="IP of the robot.")
parser.add_argument("--launch_ros", action="store_true", help="If to launch robot bringup files.")
parser.add_argument("--use_fake_hardware", default="false", type=str, help="If to use fake hardware (RViz) or not.")


# parse the arguments
args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


def main() -> None:
    """Test the executor within the IsaacLab/IsaacSim framework."""
    # create environment configuration
    env_cfg = parse_ros2_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, ros2_workspace=args_cli.ros2_ws
    )
    env_cfg.robot_ip = args_cli.robot_ip
    env_cfg.use_fake_hardware = args_cli.use_fake_hardware
    env_cfg.launch_ros = args_cli.launch_ros

    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())
    env = ROS2SkilletEnv(env)

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Low-level policies
    ik_ee_pose_policy = PoseAbsIKEEPolicy(env.obs_spec_ikee, env.action_spec)

    # Skills
    skill_length = 100
    reach_pose_skill = ReachPoseSkill(name="reach_pose_skill", policy=ik_ee_pose_policy, length=skill_length)
    skills: list[BatchedSkill[IKEE_Obs, BxM_Action, XYZ_QUAT_Params]] = [reach_pose_skill]

    # Parameters policy
    fixed_param_policy = FixedSequencePolicy(
        env.obs_spec_policy,
        reach_pose_skill.params_spec,
        torch.as_tensor(
            [
                [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
                [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
                [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
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

    while True:
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")


if __name__ == "__main__":
    main()
