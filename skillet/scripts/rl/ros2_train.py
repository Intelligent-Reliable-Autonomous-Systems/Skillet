"""ros2_train.py.

Script for training agents in ROS2.

Written by Will Solow and Jeff Jewett, 2026
"""

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import torch

import skillet_tasks.ros2web_tasks  # noqa: F401
from skillet.envs import SkillEnvWrapper, SkilletEnv
from skillet.envs.compatibility.rsl_rl import RslRlVecEnvWrapper
from skillet.envs.ros2.websocket.ros_bridge import setup_ros
from skillet.envs.util import get_checkpoint_path
from skillet.envs.util.dict import print_dict
from skillet.envs.util.hydra import hydra_task_config
from skillet.envs.util.parse_cfg import dump_yaml
from skillet.rl.cfg import RslRlBaseRunnerCfg
from skillet.rl.rsl_rl import cli_args
from skillet.rl.rsl_rl.runners import OnPolicyRunner

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--device", type=str, default="cuda", choices={"cuda", "cpu"}, help="Device to run on: cuda/cpu")
parser.add_argument(
    "--ros2_ws", type=str, default=None, required=True, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--robot_ip", default="192.168.8.10", type=str, help="IP of the robot.")
parser.add_argument("--launch_ros", action="store_true", help="If to launch robot bringup files.")
parser.add_argument("--skill", action="store_true", help="If to use a a skill-based RL environment")

cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")


# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    # Override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    # Set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.device = args_cli.device if args_cli.device is not None else env_cfg.device

    # Specify directory for logging experiments
    log_root_path = os.path.join("_logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[INFO] Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Set the log directory for the environment
    env_cfg.log_dir = log_dir
    env_cfg.robot_ip = args_cli.robot_ip
    env_cfg.launch_ros = args_cli.launch_ros

    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    # Save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for RSL-RL
    env = SkillEnvWrapper(env) if args_cli.skill else SkilletEnv(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner from RSL-RL
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # Load checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
