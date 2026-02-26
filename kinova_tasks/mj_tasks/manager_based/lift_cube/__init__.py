"""Kinova Gen3 manipulation tasks for mjlab."""

import gymnasium as gym
from mjlab.tasks.registry import register_mjlab_task

from .agents.rsl_rl_ppo_cfg import kinova_ppo_runner_cfg
from .lift_cube_ik_env_cfg import kinova_lift_ik_env_cfg
from .lift_cube_joint_env_cfg import kinova_lift_cube_joint_env_cfg

register_mjlab_task(
    task_id="MJ-Lift-Cube-Kinova-v0",
    env_cfg=kinova_lift_cube_joint_env_cfg(),
    play_env_cfg=kinova_lift_cube_joint_env_cfg(play=True),
    rl_cfg=kinova_ppo_runner_cfg(),
)

# Cartesian-space lift task
register_mjlab_task(
    task_id="MJ-Lift-Cube-Kinova-IK-v0",
    env_cfg=kinova_lift_ik_env_cfg(),
    play_env_cfg=kinova_lift_ik_env_cfg(play=True),
    rl_cfg=kinova_ppo_runner_cfg(),
)

gym.register(
    id="MJ-Lift-Cube-Kinova-v0",
    entry_point="skillet.envs.mujoco:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": kinova_lift_cube_joint_env_cfg(),
        "rsl_rl_cfg_entry_point": kinova_ppo_runner_cfg(),
    },
    disable_env_checker=True,
)
