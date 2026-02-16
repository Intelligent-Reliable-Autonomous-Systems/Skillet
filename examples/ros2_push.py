"""ros2_ik.py.

Test file for executor integration ROS2 skills

Written by Will Solow and Jeff Jewett, 2026

"""

"""Script to an environment with random action agent."""

import argparse
import os

from skillet.skill.high_level.push import PushSkill

# add argparse arguments
parser = argparse.ArgumentParser(description="Main ROS2 executor file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ROS2-Reach-Kinova-v0", help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)

# parse the arguments
args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


"""Rest everything follows."""

import gymnasium as gym
import torch
from jaxtyping import Float, Int

import kinova_tasks.ros2_tasks  # noqa: F401
from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.ros2_env_wrapper import ROS2EnvWrapper
from skillet.envs.util import parse_ros2_env_cfg, setup_ros
from skillet.policy.dummy import FixedSequencePolicy, RandomPolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy

BxN_Obs = Float[torch.Tensor, "b n"]
"""Environment observation: torch.Tensor[(b, n), float]"""
BxM_Action = Float[torch.Tensor, "b m"]
"""Environment action: torch.Tensor[(b, m), float]"""
B_Int_HighLevel = Int[torch.Tensor, "b"]
"""Selected skills action: torch.Tensor[(b,), int]"""


def main() -> None:
    """Test the executor within the IsaacLab/IsaacSim framework."""
    # create environment configuration
    env_cfg = parse_ros2_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        ros2_workspace=args_cli.ros2_ws,
    )
    env_cfg.robot_ip = "192.168.1.10"
    env_cfg.use_fake_hardware = "true"
    env_cfg.launch_ros = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Set up Skill executor and environment in framework
    env = ROS2EnvWrapper[BxN_Obs, BxM_Action](env)

    action_spec: ActionSpec[BxM_Action] = env.action_spec
    observation_spec: ObservationSpec[BxN_Obs] = env.obs_spec

    # Low-level policies
    _obs_spec_ik = ObservationSpec[Float[torch.Tensor, "b ..."]](
        space=gym.spaces.Dict(),
        name="ik_ee",
        is_torch=True,
        is_batched=True,
        n_envs=-1,
        device=env.device,
    )

    ik_ee_pose_policy = PoseAbsIKEEPolicy[BxN_Obs, BxM_Action](_obs_spec_ik, action_spec)
    # Skills
    skill_length = 200
    grasp_skill = PushSkill[BxN_Obs, BxM_Action, None](
        reach_policy=ik_ee_pose_policy, gripper_policy=None, length=skill_length
    )
    skills = [grasp_skill]

    # Parameters policy
    fixed_param_policy = FixedSequencePolicy[BxN_Obs, BxM_Action](
        observation_spec,
        action_spec,
        torch.as_tensor(
            [
                [0.6, -0.2, 0.03, 0.1, 0.1, 0.0],
            ],
            device=env.device,
        ),
    )

    # High-level policy
    options_spec = ActionSpec[B_Int_HighLevel](
        space=gym.spaces.MultiDiscrete([len(skills)] * args_cli.num_envs),
        name="options",
        is_torch=True,
        is_batched=True,
        # n_envs=args_cli.num_envs,
    )
    policy_over_options = RandomPolicy[BxN_Obs, B_Int_HighLevel](observation_spec, options_spec)

    policy_over_options_agent = PolicyOverOptionsAgent[BxN_Obs, BxM_Action, B_Int_HighLevel, None](
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
