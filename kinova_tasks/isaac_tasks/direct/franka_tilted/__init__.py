# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Franka-Cabinet environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Franka-Tilted-Direct-v0",
    entry_point=f"{__name__}.franka_tilted_env:FrankaTiltedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tilted_env:FrankaTiltedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaTiltedPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Tilted-IK-v0",
    entry_point=f"{__name__}.franka_tilted_env:FrankaTiltedIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tilted_env:FrankaTiltedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaTiltedPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Tilted-OSC-v0",
    entry_point=f"{__name__}.franka_tilted_env:FrankaTiltedOSCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tilted_env:FrankaTiltedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaTiltedPPORunnerCfg",
    },
)
