"""ros2_play.py.

Script for visualizing a policy with ROS2.

Written by Will Solow and Jeff Jewett, 2026
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

import skillet_tasks.ros2_tasks  # noqa: F401
from skillet.envs import SkillEnvWrapper, SkilletEnv
from skillet.envs.compatibility.rsl_rl import RslRlVecEnvWrapper
from skillet.envs.util import get_checkpoint_path, setup_ros
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
parser.add_argument("--num_envs", type=int, default=1, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--ros2_ws", type=str, default=None, required=True, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--device", type=str, default="cuda", choices={"cuda", "cpu"}, help="Device to run on: cuda/cpu")
parser.add_argument("--robot_ip", default="192.168.8.10", type=str, help="IP of the robot.")
parser.add_argument("--launch_ros", action="store_true", help="If to launch robot bringup files.")
parser.add_argument("--skill", action="store_true", help="If to use a a skill-based RL environment")


cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")

if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    np.set_printoptions(suppress=True, precision=4)

    # Override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.num_envs

    # Set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.device = args_cli.device if args_cli.device is not None else env_cfg.device

    # Specify directory for logging experiments
    log_root_path = os.path.join("_logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # Set the log directory for the environment
    env_cfg.log_dir = log_dir
    env_cfg.robot_ip = args_cli.robot_ip
    env_cfg.launch_ros = args_cli.launch_ros

    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
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
    export_policy_as_jit(runner.alg.policy, normalizer=normalizer, path=export_model_dir, filename="policy.pt")

    # Visualize with ROS2
    env.reset()
    obs = env.get_observations()
    timestep = 0
    while True:
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
