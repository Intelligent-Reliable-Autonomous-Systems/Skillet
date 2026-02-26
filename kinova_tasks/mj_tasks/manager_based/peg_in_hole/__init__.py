"""Kinova Gen3 manipulation tasks for mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from .agents.rsl_rl_ppo_cfg import kinova_ppo_runner_cfg
from .peg_in_hole_env_cg import kinova_peg_in_hole_env_cfg

# Peg-in-hole task (cartesian)
register_mjlab_task(
    task_id="Mjlab-Peg-In-Hole-Kinova",
    env_cfg=kinova_peg_in_hole_env_cfg(),
    play_env_cfg=kinova_peg_in_hole_env_cfg(play=True),
    rl_cfg=kinova_ppo_runner_cfg(),
)
