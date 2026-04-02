"""Run a tabletop block stacking task."""

import argparse
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch

from skillet.agents.policy_over_options import PolicyOverOptionsAgent, SelectedSkill
from skillet.core import ActionSpec, ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs.skillet_env import SkilletEnv
from skillet.perception.localization import ApriltagStateReconstructor
from skillet.perception.realsense import RealsenseEnv
from skillet.policy.dummy import FixedSequencePolicy
from skillet.policy.ik_ee import PoseAbsIKEEPolicy
from skillet.policy.moveit import MoveItTcpQuatPolicy
from skillet.policy.twist import TwistPIDPosePolicy
from skillet.scene import SkilletVisualizer
from skillet.scene.base import Scene
from skillet.scene.cube import Cube
from skillet.skill.high_level.pick import PickSkill
from skillet.skill.high_level.place import PlaceSkill
from skillet.skill.high_level.rotate_yaw import RotateYawSkill
from skillet.skill.object_level.pick_block import PickBlockSkill
from skillet.skill.object_level.place_block import PlaceBlockSkill
from skillet.skill.object_level.rotate_block import RotateBlockSkill
from skillet_tasks.kortex_tasks.factory import create_kortex_env

if TYPE_CHECKING:
    from skillet.envs.specs import BxM_Action, IKEE_Obs, RGBD_Obs

parser = argparse.ArgumentParser(description="Visualize latest RGB-D frame from ROS2 service.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--use_moveit", action=argparse.BooleanOptionalAction, default=True, help="Use MoveIt for motion planning."
)
parser.add_argument("--use_twist", action=argparse.BooleanOptionalAction, default=False, help="Use cartesian servoing.")
parser.add_argument("--realsense_env", action="store_true", help="Use RealSense camera environment.")
parser.add_argument(
    "--viz", type=str, default="rgb,depth,pointcloud", help="Visualization modes to display, as comma-separated string."
)
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--period_s", type=float, default=1.0, help="Seconds between service requests.")
parser.add_argument("--max_depth_m", type=float, default=None, help="Optional far-plane clipping depth in meters.")
parser.add_argument("--task", type=str, default="Kortex-Gen3Lite-v0", help="Kortex Environment")

args_cli = parser.parse_args()

TABLE_X0 = -0.0889
TABLE_Y0 = -0.577
TABLE_DX = 0.762
TABLE_DY = 1.2446


def main() -> None:
    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    cube_0 = Cube(size=0.041, face_apriltags=[{"face": "top", "size": 0.036, "id": 1}])
    cube_1 = Cube(size=0.041, face_apriltags=[{"face": "front", "size": 0.036, "id": 2}])
    cube_2 = Cube(size=0.041, face_apriltags=[{"face": "front", "size": 0.036, "id": 5}])

    world_bounds = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)
    scene = Scene(objects=[cube_0, cube_1, cube_2], closed_set=True, bounds=world_bounds)

    if args_cli.realsense_env:
        env = RealsenseEnv(apriltag_size_m=0.1, apriltag_id=3)
    else:
        env_cfg = {
            "robot_ip": args_cli.robot_ip,
            "device": args_cli.device,
            "num_envs": args_cli.num_envs,
        }

        env = create_kortex_env(args_cli.task, env_cfg)
        env = SkilletEnv(env)
        ikee_spec: ObservationSpec[IKEE_Obs] = env.coerce_obs_spec("ik_ee").batched()
        low_action_spec: ActionSpec[BxM_Action] = env.coerce_action_spec("joints").batched()
        env = BatchToSingleWrapper(env)
        env.reset()
    rgbd_spec: ObservationSpec[RGBD_Obs] = env.coerce_obs_spec("rgb-d")

    poll_rate_hz = 1.0 / max(args_cli.period_s, 1e-6)
    visualizer = SkilletVisualizer(
        env=env,
        obs_spec=rgbd_spec,
        reconstructor=ApriltagStateReconstructor(scene),
        poll_rate=poll_rate_hz,
        device=args_cli.device,
        max_depth_m=args_cli.max_depth_m,
    )

    if "pointcloud" in args_cli.viz:
        visualizer.set_open3d_visualizer(segment_point_cloud=True)
    visualizer.start_cv2_visualization(
        display_rgb="rgb" in args_cli.viz,
        display_depth="depth" in args_cli.viz,
        segment_rgb="rgb" in args_cli.viz,
        segment_depth="depth" in args_cli.viz,
    )
    visualizer.run_thread()

    # Low-level policies
    if args_cli.use_moveit:
        arm_policy = MoveItTcpQuatPolicy(env.batched_env.obs_spec_ikee, env.batched_env.action_spec_moveit_tcp_quat)
    elif args_cli.use_twist:
        arm_policy = TwistPIDPosePolicy(env.batched_env.obs_spec_twist_tcp, env.batched_env.action_spec_twist_tcp)
    else:
        arm_policy = PoseAbsIKEEPolicy(ikee_spec, low_action_spec)
    # Skills
    skill_length = 1e9
    place_skill = PlaceSkill(reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, length=skill_length)
    pick_skill = PickSkill(reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, length=skill_length)
    rotate_y_skill = RotateYawSkill(
        reach_policy=arm_policy, gripper_policy=None, lift_height=0.23, lift_delta=0.04, length=skill_length
    )
    pick_block_skill = PickBlockSkill(scene, pick_skill, vis_target_pos=vis.set_target_pos)
    place_block_skill = PlaceBlockSkill(scene, place_skill, vis_target_pos=vis.set_target_pos)
    rotate_block_skill = RotateBlockSkill(scene, rotate_y_skill, vis_target_pos=vis.set_target_pos)
    skills = [pick_block_skill, place_block_skill, rotate_block_skill]

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
            [0, 1, 0, 1, 2],
            device=rgbd_spec.device,
            dtype=torch.int32,
        ),
    )
    fixed_param_policy = FixedSequencePolicy(
        rgbd_spec,
        pick_block_skill.params_spec,
        torch.as_tensor(
            [1, 2, 0, 1, 0],
            device=rgbd_spec.device,
            dtype=torch.int32,
        ),
    )

    policy_over_options_agent = PolicyOverOptionsAgent(
        skills=skills,
        high_level_policy=policy_over_options,
        params_policy=fixed_param_policy,
    )

    if not args_cli.realsense_env:
        input("Press Enter to start the skill execution...")

    # simulate environment
    while True:
        with torch.inference_mode():
            env.reset()
            policy_over_options_agent.execute(env)
            print("[INFO][Main] finished run of skill executor, resetting")


if __name__ == "__main__":
    main()
