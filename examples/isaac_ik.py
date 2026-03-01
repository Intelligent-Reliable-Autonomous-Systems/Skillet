"""main_isaac.py.

Test file for executor integration with IsaacSim and ROS2

Written by Will Solow and Jeff Jewett, 2026

"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Launch Isaac Sim Simulator first.
import argparse
from typing import Mapping

from isaaclab.app import AppLauncher
from skillet.core.policy import Unparameterized

# add argparse arguments
parser = argparse.ArgumentParser(description="Main IsaacSim Executor file through IsaacLab.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Kinova-Reach-IK-v0", help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Import isaaclab_tasks after app launcher
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from jaxtyping import Float, Int

import kinova_tasks.isaac_tasks as tasks  # noqa: F401
from kinova_tasks.isaac_tasks.factory import create_isaac_env
from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.isaac_env_wrapper import IsaacEnvWrapper
from skillet.policy.dummy import FixedSequencePolicy, RandomPolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy, XYZ_QUAT_Params
from skillet.skill import ReachPoseSkill

BxN_Obs = Float[torch.Tensor, "b n"]
"""Environment observation: torch.Tensor[(b, n), float]"""
BxM_Action = Float[torch.Tensor, "b m"]
"""Environment action: torch.Tensor[(b, m), float]"""
IKEE_Obs = Mapping[str, torch.Tensor]
B_Int_HighLevel = Int[torch.Tensor, "b"]
"""Selected skills action: torch.Tensor[(b,), int]"""
XYZ_Params = Float[torch.Tensor, "b 3"]
"""XYZ parameters: torch.Tensor[(b, 3), float]"""


def main() -> None:

    cfg = {
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
        "use_fabric": not args_cli.disable_fabric,
    }
    env = create_isaac_env(args_cli.task, cfg)
    # Set up Skill executor and environment in framework
    env = IsaacEnvWrapper[BxN_Obs, BxM_Action](env)


    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")


    action_spec: ActionSpec[BxM_Action] = env.action_spec
    observation_spec: ObservationSpec[BxN_Obs] = env.obs_spec

    # Low-level policies
    ik_ee_pose_policy = PoseAbsIKEEPolicy[BxM_Action](action_spec)
    # Skills
    skill_length = 100
    reach_pose_skill = ReachPoseSkill[BxM_Action](
        name="reach_pose_skill", policy=ik_ee_pose_policy, length=skill_length
    )
    skills = [reach_pose_skill]

    # Parameters policy
    fixed_param_policy = FixedSequencePolicy[BxN_Obs, XYZ_QUAT_Params](
        observation_spec,
        action_spec,
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
    options_spec = ActionSpec[B_Int_HighLevel](
        space=gym.spaces.MultiDiscrete([len(skills)] * args_cli.num_envs),
        name="options",
        is_torch=True,
        is_batched=True,
    )
    policy_over_options = RandomPolicy[BxN_Obs, B_Int_HighLevel](observation_spec, options_spec)

    policy_over_options_agent = PolicyOverOptionsAgent[BxN_Obs, BxM_Action, B_Int_HighLevel, XYZ_QUAT_Params](
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            # skill_executor.execute()
            print("[INFO][Main] finished run of skill executor, resetting")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
