# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Gen3-Lift-Cube-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_lift_env_cfg:TeleOpGen3LiftCubeEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Gen3-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_lift_env_cfg:Gen3LiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
