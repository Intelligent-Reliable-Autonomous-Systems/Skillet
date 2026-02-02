"""main_ros2.py.

Test file for executor integration with IsaacSim and ROS2

Written by Will Solow and Jeff Jewett, 2026

"""

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Main ROS2 executor file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, required=True, help="Absolute path to ROS2 workspace containing bringup files"
)

# parse the arguments
args_cli = parser.parse_args()


"""Rest everything follows."""
import time

import gymnasium as gym
import torch
from jaxtyping import Float, Int
from roslibpy import Ros

import ros2  # noqa: F401
from ros2.envs.utils import parse_ros2_env_cfg
from skillet.agents.policy_over_options import PolicyOverOptionsAgent
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.ros2_env_wrapper import ROS2EnvWrapper
from skillet.policy.dummy import RandomPolicy, ZeroPolicy
from skillet.skill.fixed_length import FixedLengthSkill


def setup_ros() -> Ros:
    """Open the ROS2 interface."""
    print("[INFO][Setup ROS] Waiting to connect to ROSBridge")
    print(
        "[INFO][Setup ROS] Ensure that rosbridge node is running: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`"
    )
    # Wait until it starts
    ros = Ros(host="localhost", port=9090)
    start = time.time()
    while True:
        try:
            ros.run(timeout=1)
            if ros.is_connected:
                print("[INFO][Setup ROS] Connected to rosbridge")
                break
        except RuntimeError:
            if time.time() - start > 30:
                raise TimeoutError(
                    "RosBridge failed to start. Is the rosbridge node running? ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
                )
            time.sleep(0.2)

    return ros


BxN_Obs = Float[torch.Tensor, "b n"]
"""Environment observation: torch.Tensor[(b, n), float]"""
BxM_Action = Float[torch.Tensor, "b m"]
"""Environment action: torch.Tensor[(b, m), float]"""
B_Int_HighLevel = Int[torch.Tensor, "b"]
"""Selected skills action: torch.Tensor[(b,), int]"""


def main() -> None:
    """Test the executor within the IsaacLab/IsaacSim framework."""
    # create environment configuration
    env_cfg = parse_ros2_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, ros2_workspace=args_cli.ros2_ws
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Set up Skill executor and environment in framework
    env = ROS2EnvWrapper[BxN_Obs, BxM_Action](env)

    # action_spec = ActionSpec[BxN_Obs](
    #     space=env.action_space,
    #     name="isaac_action",
    #     is_torch=True,
    #     is_batched=True,
    #     # n_envs=args_cli.num_envs,
    # )
    action_spec: ActionSpec[BxM_Action] = env.action_spec
    observation_spec: ObservationSpec[BxN_Obs] = env.obs_spec
    # observation_spec = ObservationSpec[BxN_Obs](
    #     space=env.observation_space,
    #     name="policy",
    #     is_torch=True,
    #     is_batched=True,
    #     # n_envs=args_cli.num_envs,
    # )

    # Low-level policies
    zero_policy = ZeroPolicy[BxN_Obs, BxM_Action](observation_spec, action_spec)
    random_policy = RandomPolicy[BxN_Obs, BxM_Action](observation_spec, action_spec)
    # Skills
    skill_length = 5
    zero_skill = FixedLengthSkill[BxN_Obs, BxM_Action, None](name="zero_skill", policy=zero_policy, length=skill_length)
    random_skill = FixedLengthSkill[BxN_Obs, BxM_Action, None](
        name="random_skill", policy=random_policy, length=skill_length
    )
    skills = [zero_skill, random_skill]

    # High-level policy
    options_spec = ActionSpec[B_Int_HighLevel](
        space=gym.spaces.MultiDiscrete([len(skills)] * args_cli.num_envs),
        name="options",
        is_torch=True,
        is_batched=True,
        # n_envs=args_cli.num_envs,
    )
    policy_over_options = RandomPolicy[BxN_Obs, B_Int_HighLevel](observation_spec, options_spec)

    policy_over_options_agent = PolicyOverOptionsAgent[BxN_Obs, BxM_Action, B_Int_HighLevel, None](
        skills=[zero_skill, random_skill],
        high_level_policy=policy_over_options,
        params_policy=None,
    )

    # env.step()
    # skill_executor = SkillExecutor(DummyCfg(), env)

    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            # skill_executor.execute()
            print("[INFO][Main] finished run of skill executor, resetting")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()


'''def main() -> None:
    """Test the executor within the IsaacLab/IsaacSim framework."""
    np.set_printoptions(precision=3)
    # create environment configuration
    env_cfg = parse_ros2_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, ros2_workspace=args_cli.ros2_ws
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, ros=setup_ros())

    print("[INFO][Main] Testing Executor environment")
    print(f"[INFO][Main] Gym observation space: {env.observation_space}")
    print(f"[INFO][Main] Gym action space: {env.action_space}")

    # Set up Skill executor and environment in framework
    ROS2EnvWrapper = import_ros2_wrapper()
    env = ROS2EnvWrapper(env)
    skill_executor = SkillExecutor(DummyCfg(), env)

    # simulate environment
    skill_executor.execute()

    # close the environment. Note that the spun up ROS2 nodes will not close automatically
    env.close()


if __name__ == "__main__":
    main()
'''
