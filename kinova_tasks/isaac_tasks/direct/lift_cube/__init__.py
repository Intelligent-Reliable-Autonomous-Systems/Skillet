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
    id="Kinova-Lift-Cube-Direct-v0",
    entry_point=f"{__name__}.kinova_lift_cube_env:KinovaLiftCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_lift_cube_env:KinovaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaLiftCubePPORunnerCfg",
    },
)

gym.register(
    id="Kinova-Lift-Cube-IK-v0",
    entry_point=f"{__name__}.kinova_lift_cube_env:KinovaLiftCubeIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_lift_cube_env:KinovaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaLiftCubePPORunnerCfg",
    },
)

gym.register(
    id="Kinova-Lift-Cube-OSC-v0",
    entry_point=f"{__name__}.kinova_lift_cube_env:KinovaLiftCubeOSCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_lift_cube_env:KinovaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaLiftCubePPORunnerCfg",
    },
)


gym.register(
    id="Franka-Lift-Cube-Direct-v0",
    entry_point=f"{__name__}.franka_lift_cube_env:FrankaLiftCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_lift_cube_env:FrankaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaLiftCubePPORunnerCfg",
    },
)

gym.register(
    id="Franka-Lift-Cube-IK-v0",
    entry_point=f"{__name__}.franka_lift_cube_env:FrankaLiftCubeIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_lift_cube_env:FrankaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaLiftCubePPORunnerCfg",
    },
)

gym.register(
    id="Franka-Lift-Cube-OSC-v0",
    entry_point=f"{__name__}.franka_lift_cube_env:FrankaLiftCubeOSCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_lift_cube_env:FrankaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaLiftCubePPORunnerCfg",
    },
)
