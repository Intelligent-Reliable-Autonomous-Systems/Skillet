"""Run a tabletop block stacking task."""

import argparse
import os
from typing import Any

import gymnasium as gym
import torch

from skillet.agents.policy_over_options import PolicyOverOptionsAgent, SelectedSkill
from skillet.core import ActionSpec, ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs.ros2_skillet_env import ROS2SkilletEnv
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action, N_Obs
from skillet.envs.util import setup_ros
from skillet.perception.perception import Perception
from skillet.perception.realsense import RealsenseEnv
from skillet.perception.sam3.sam3 import SAMConcept
from skillet.policy.dummy import FixedSequencePolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy
from skillet.scene.base import Scene
from skillet.scene.cube import Cube
from skillet.scene.visualize import Open3DVisualizer
from skillet.skill.high_level.pick import PickSkill
from skillet.skill.high_level.pick_block import PickBlockSkill
from skillet_tasks.ros2_tasks.gen3.gen3_ros2 import Gen3ROS2Env, Gen3ROS2EnvCfg

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--ros2_ws", type=str, default=None, help="Absolute path to ROS2 workspace containing bringup files"
)
parser.add_argument("--segmentation", action=argparse.BooleanOptionalAction, default=True, help="Use segmentation.")
parser.add_argument("--realsense_env", action="store_true", help="Use RealSense camera environment.")
parser.add_argument("--viz", type=str, default="rgb,depth,pointcloud", help="Visualization modes to display, as comma-separated string.")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--use_fake_hardware", type=str, default="true", help="'true' or 'false'.")
parser.add_argument("--launch_ros", action="store_true", help="Launch ROS from env startup.")
parser.add_argument("--period_s", type=float, default=1.0, help="Seconds between service requests.")
parser.add_argument("--max_depth_m", type=float, default=None, help="Optional far-plane clipping depth in meters.")

args_cli = parser.parse_args()
if args_cli.ros2_ws is None:
    args_cli.ros2_ws = os.getenv("ROS2_WS", None)
    if args_cli.ros2_ws is None:
        raise ValueError("ROS2 workspace path must be provided via --ros2_ws argument or ROS2_WS environment variable.")

TABLE_X0 = -0.0889
TABLE_Y0 = -0.577
TABLE_DX = 0.762
TABLE_DY = 1.2446

def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    cube_0 = Cube(size=0.04, face_apriltags=[{"face": "top", "size": 0.036, "id": 0}])
    cube_1 = Cube(size=0.04, face_apriltags=[{"face": "front", "size": 0.036, "id": 3}])

    world_bounds = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1) # min_x, min_y, min_z, max_x, max_y, max_z
    scene = Scene(objects=[cube_0, cube_1], closed_set=True, bounds=world_bounds)

    # PROMPTS = {
    #     "wooden_block": "a light brown wooden block",
    #     "purple_block": "a solid purple block without any writing or markings",
    #     "yellow_block": "a solid yellow block without any writing or markings",
    #     "green_block": "a solid green block without any writing or markings",
    # }
    sam3_prompts = [
        SAMConcept(name="block_8", prompt="wooden block with number 8 on it", exemplar_images=["/home/iras/skillet/data/images/wooden_block_8.png"]),
        SAMConcept(name="block_7", prompt="wooden block with number 7 on it", exemplar_images=["/home/iras/skillet/data/images/wooden_block_7.png"]),
        SAMConcept(name="mouse", prompt="a computer mouse", exemplar_images=["/home/iras/skillet/data/images/computer_mouse.jpeg"]),
    ]

    if args_cli.realsense_env:
        env = RealsenseEnv(apriltag_size_m=0.036, apriltag_id=0)
    else:
        env_cfg = Gen3ROS2EnvCfg(
            robot_ip=args_cli.robot_ip,
            use_fake_hardware=args_cli.use_fake_hardware,
            launch_ros=args_cli.launch_ros,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            ros2_workspace=args_cli.ros2_ws,
            episode_length_s=30.0,
        )

        env = Gen3ROS2Env(cfg=env_cfg, ros=setup_ros())
        env = ROS2SkilletEnv(env)
    env = BatchToSingleWrapper[N_Obs, M_Action](env)
    env.reset()
    rgbd_spec = env.coerce_obs_spec("rgb-d")
    ikee_spec: ObservationSpec[IKEE_Obs] = env.coerce_obs_spec("ik_ee").batched()
    low_action_spec: ActionSpec[BxM_Action] = env.coerce_action_spec("joints").batched()

    poll_rate_hz = 1.0 / max(args_cli.period_s, 1e-6)
    perception = Perception(
        env=env,
        obs_spec=rgbd_spec,
        scene=scene,
        segmentation=args_cli.segmentation,
        poll_rate=poll_rate_hz,
        device=args_cli.device,
        max_depth_m=args_cli.max_depth_m,
        prompts=sam3_prompts,
    )

    vis = Open3DVisualizer(scene)
    if "pointcloud" in args_cli.viz:
        perception.set_visualizer(vis, segment_point_cloud=True)
    perception.start_cv2_visualization(
        display_rgb="rgb" in args_cli.viz, display_depth="depth" in args_cli.viz,
        segment_rgb="rgb" in args_cli.viz, segment_depth="depth" in args_cli.viz,
    )
    perception.run_thread()
    # perception.run()
    vis.run_thread()
    # vis.run()

    # Low-level policies
    ik_ee_pose_policy = PoseAbsIKEEPolicy(ikee_spec, low_action_spec)
    # Skills
    skill_length = 200
    pick_skill = PickSkill(
        reach_policy=ik_ee_pose_policy, gripper_policy=None, lift_height=0.23, length=skill_length
    )
    pick_block_skill = PickBlockSkill(scene, pick_skill)
    skills = [pick_block_skill]


    # High-level policy
    options_spec = ActionSpec[SelectedSkill](
        space=gym.spaces.Discrete(len(skills)),
        name="options",
        is_torch=True,
        is_batched=False,
    )
    policy_over_options = FixedSequencePolicy[Any, SelectedSkill](
        rgbd_spec,
        options_spec,
        torch.as_tensor(
            [0],
            device=rgbd_spec.device,
            dtype=torch.int32,
        ),
    )
    fixed_param_policy = FixedSequencePolicy(
        rgbd_spec,
        pick_block_skill.params_spec,
        torch.as_tensor(
            [1, 0],
            device=rgbd_spec.device,
            dtype=torch.int32,
        ),
    )

    policy_over_options_agent = PolicyOverOptionsAgent(
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    # simulate environment
    while True:
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            # skill_executor.execute()
            print("[INFO][Main] finished run of skill executor, resetting")


    perception.stop()
    env.close()


if __name__ == "__main__":
    main()
