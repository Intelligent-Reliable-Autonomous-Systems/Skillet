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
    id="Kinova-Lift-Cube-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_lift_env_cfg:TeleOpKinovaLiftCubeEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Kinova-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_lift_env_cfg:KinovaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
