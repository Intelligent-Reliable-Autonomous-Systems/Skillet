# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from skillet.envs.util import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# Pre-defined configs
##
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.managers import SceneEntityCfg

from kinova_tasks.assets.kinova_gen3_2f85 import KINOVA_GEN3_2F85_CFG

##
# Environment configuration
##


@configclass
class KinovaReachEnvCfg(ReachEnvCfg):
    """Reach task with Kinova Gen3 Arm."""

    name = "gen3_reach"

    def __post_init__(self) -> None:
        """Parse config."""
        # post init of parent
        super().__post_init__()

        # Switch robot to Kinova Gen3
        self.scene.robot = KINOVA_GEN3_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override observations: this filters out the last two joints
        self.obs_asset = SceneEntityCfg("robot")
        self.obs_asset.joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.observations.policy.joint_pos.params = {"asset_cfg": self.obs_asset}
        self.observations.policy.joint_vel.params = {"asset_cfg": self.obs_asset}

        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["end_effector_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["end_effector_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["end_effector_link"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            scale=1.0,
            use_default_offset=True,
            preserve_order=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": -1.0},
            close_command_expr={"finger_joint": 1.0},
        )

        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "end_effector_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/end_effector_link",
                    name="end_effector_link",
                    offset=OffsetCfg(
                        pos=[0.120, 0.0, 0.0],
                    ),
                ),
            ],
        )


@configclass
class TeleOpKinovaReachEnvCfg(KinovaReachEnvCfg):
    """Teleop Reach task with Kinova Gen3 Arm."""

    def __post_init__(self) -> None:
        """Set up the actions and teleop devices."""
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = KINOVA_GEN3_2F85_HIGH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            body_name="end_effector_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.120, 0.0, 0.0], rot=[0.0,1.0,0.0,0.0]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
