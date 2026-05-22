# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from ....ros2web_tasks.gen3 import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##
gym.register(
    id="Ros2Web-Gen3-v0",
    entry_point=f"{__name__}.gen3_ros2:Gen3Ros2WebEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_ros2:Gen3Ros2WebEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
)


gym.register(
    id="Ros2Web-Gen3-Reach-RL-v0",
    entry_point=f"{__name__}.gen3_ros2:Gen3ROS2ReachRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_ros2:Gen3Ros2WebEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
)
