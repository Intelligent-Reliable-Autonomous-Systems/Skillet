"""isaac_ik.py.

Test file for executor integration Skillet skills in IsaacSim

Written by Will Solow and Jeff Jewett, 2026

"""

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillet.core import BatchedSkill
    from skillet.envs.specs import BxM_Action, IKEE_Obs
import gymnasium as gym
import torch

import skillet_tasks.mj_tasks  # noqa: F401
from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.envs import SkilletEnv
from skillet.envs.util import parse_mj_env_cfg
from skillet.skill.low_level import ReachPoseSkill
from skillet.skill.policy.dummy import FixedSequencePolicy, RandomPolicy
from skillet.skill.policy.ik_ee import PoseAbsIkEePolicy
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED, XYZ_QUAT_Params

# Add argparse arguments
parser = argparse.ArgumentParser(description="Main IsaacSim Executor file through IsaacLab.")
parser.add_argument("--num_envs", type=int, default=4, required=False, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Mj-Gen3Lite-Reach-Direct-v0", required=False, help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", required=False, help="Name of device.")

args_cli = parser.parse_args()


def main() -> None:
    env_cfg = parse_mj_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )  # Override hydra task cfg to avoid serialization

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkilletEnv(env)

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Low-level policies
    ik_ee_pose_policy = PoseAbsIkEePolicy(env.obs_spec_ikee, env.action_spec)

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

    policy_over_options_agent = PolicyOverOptionsAgent(
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    while True:
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")

    env.close()


if __name__ == "__main__":
    main()
