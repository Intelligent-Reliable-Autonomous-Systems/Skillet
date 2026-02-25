# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##
gym.register(
    id="ROS2-Kinova-Reach-v0",
    entry_point=f"{__name__}.kinova_reach_ros2:KinovaROS2ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_reach_ros2:KinovaROS2ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaReachPPORunnerCfg",
    },
)

gym.register(
    id="ROS2-Kinova-Reach-RL-v0",
    entry_point=f"{__name__}.kinova_reach_ros2_rl:KinovaROS2ReachRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_reach_ros2:KinovaROS2ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaReachPPORunnerCfg",
    },
)
