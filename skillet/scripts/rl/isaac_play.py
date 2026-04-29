"""isaac_play.py.

Script for visualizing a policy with IsaacSim.

Written by Will Solow and Jeff Jewett, 2026
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher

from skillet.envs import SkillEnvWrapper, SkilletEnv
from skillet.envs.compatibility.rsl_rl import RslRlVecEnvWrapper
from skillet.envs.util import get_checkpoint_path
from skillet.envs.util.dict import print_dict
from skillet.envs.util.hydra import hydra_task_config
from skillet.rl.cfg import RslRlBaseRunnerCfg
from skillet.rl.exporter import export_policy_as_jit
from skillet.rl.rsl_rl import cli_args
from skillet.rl.rsl_rl.runners import OnPolicyRunner

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--skill", action="store_true", help="If to use a a skill-based RL environment")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401

import skillet_tasks.isaac_tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""

    # Override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Specify directory for logging experiments
    log_root_path = os.path.join("_logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # env_cfg = OmegaConf.structured(type(env_cfg), flags={"allow_objects": True})
    # env_yaml_cfg = OmegaConf.load(f"{log_dir}/params/env.yaml")
    # env_cfg = OmegaConf.merge(env_cfg, env_yaml_cfg)

    # agent_cfg = OmegaConf.structured(type(agent_cfg))
    # agent_yaml_cfg = OmegaConf.load(f"{log_dir}/params/agent.yaml")
    # agent_cfg = OmegaConf.merge(agent_cfg, agent_yaml_cfg)
    # Set the log directory for the environment
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap around environment for RSL-RL
    env = SkillEnvWrapper(env) if args_cli.skill else SkilletEnv(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if hasattr(runner.alg.policy, "actor_obs_normalizer"):
        normalizer = runner.alg.policy.actor_obs_normalizer
    elif hasattr(runner.alg.policy, "student_obs_normalizer"):
        normalizer = runner.alg.policy.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, normalizer=normalizer, path=export_model_dir, filename="agent")

    # Visualize with IsaacSim
    env.reset()
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
