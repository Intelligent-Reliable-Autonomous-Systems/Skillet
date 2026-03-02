"""isaac_ik.py.

Test file for executor integration Skillet skills in IsaacSim

Written by Will Solow and Jeff Jewett, 2026

"""

import argparse
from typing import TYPE_CHECKING

import torch
from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from skillet.core import BatchedSkill
    from skillet.envs.specs import BxM_Action, IKEE_Obs
from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.envs.isaac_env_wrapper import IsaacEnvWrapper
from skillet.policy.dummy import FixedSequencePolicy, RandomPolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy
from skillet.skill import ReachPoseSkill
from skillet.skill.specs import SELECT_OPTIONS_SPEC_BATCHED, XYZ_QUAT_Params

# Add argparse arguments
parser = argparse.ArgumentParser(description="Main IsaacSim Executor file through IsaacLab.")
parser.add_argument("--num_envs", type=int, default=4, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Kinova-Reach-IK-v0", required=True, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from kinova_tasks.isaac_tasks.factory import create_isaac_env


def main() -> None:
    cfg = {
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
    }
    env = create_isaac_env(args_cli.task, cfg)
    env = IsaacEnvWrapper(env)

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
    simulation_app.close()
