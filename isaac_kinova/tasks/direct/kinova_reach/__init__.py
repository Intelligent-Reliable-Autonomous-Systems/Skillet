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
print("[INFO][INITIALIZING KINOVA REACH]")
gym.register(
    id="Kinova-Reach-Direct-v0",
    entry_point=f"{__name__}.kinova_reach_env:KinovaReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_reach_env:KinovaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaReachPPORunnerCfg",
    },
)
