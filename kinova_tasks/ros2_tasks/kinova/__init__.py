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
    id="ROS2-Kinova-v0",
    entry_point=f"{__name__}.kinova_ros2:KinovaROS2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_ros2:KinovaROS2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaReachPPORunnerCfg",
    },
)

gym.register(
    id="ROS2-Kinova-IK-Rel-v0",
    entry_point=f"{__name__}.kinova_ik_rel_ros2:KinovaROS2IKRelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_ros2:KinovaROS2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaReachPPORunnerCfg",
    },
)


gym.register(
    id="ROS2-Kinova-Reach-RL-v0",
    entry_point=f"{__name__}.kinova_ros2:KinovaROS2ReachRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_ros2:KinovaROS2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaReachPPORunnerCfg",
    },
)
