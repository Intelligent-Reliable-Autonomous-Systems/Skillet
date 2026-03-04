# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="ROS2-Kinova-Twist-Rel-v0",
    entry_point=f"{__name__}.kinova_rel_twist_ros2:KinovaROS2TwistRelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_ik_rel_ros2:TeleOpKinovaROS2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaReachPPORunnerCfg",
    },
)
