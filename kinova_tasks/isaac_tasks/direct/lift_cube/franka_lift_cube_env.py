# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from kinova_tasks.isaac_tasks.direct.cfg import FrankaBaseCfg
from skillet.envs.util import configclass

from .lift_cube_env import LiftCubeEnv


@configclass
class FrankaLiftCubeEnvCfg(FrankaBaseCfg):
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.7, 0.7, 0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    action_scale = 0.5

    # reward scales
    min_height = 0.04
    cube_reach_reward_scale = 1.0
    cube_reach_reward_std = 0.1
    cube_lift_reward_scale = 15.0
    cube_goal_reward_scale = 16.0
    cube_goal_reward_std = 0.3
    cube_goal_fine_grained_scale = 5.0
    cube_goal_fine_grained_std = 0.05
    action_rate_reward_scale = -0.0001
    joint_vel_reward_scale = -0.0001

    # Cube pose initial ranges
    init_pos_x = [0.2, 0.6]
    init_pos_y = [-0.25, 0.25]
    init_pos_z = [0.055, 0.055]
    init_roll = [0.0, 0.0]
    init_pitch = [0.0, 0.0]
    init_yaw = [0.0, 0.0]
    cube_init_ranges = [init_pos_x, init_pos_y, init_pos_z, init_roll, init_pitch, init_yaw]

    # Cube pose initial ranges
    goal_pos_x = [0.4, 0.6]
    goal_pos_y = [-0.25, 0.25]
    goal_pos_z = [0.25, 0.5]
    goal_roll = [0.0, 0.0]
    goal_pitch = [0.0, 0.0]
    goal_yaw = [0.0, 0.0]
    cube_goal_ranges = [goal_pos_x, goal_pos_y, goal_pos_z, goal_roll, goal_pitch, goal_yaw]

    cube_goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cube_goal_pose",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.7, 0.7, 0.7),
            )
        },
    )

    cube_current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/cube_current_pose"
    )

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    cube_current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class FrankaLiftCubeEnv(LiftCubeEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaLiftCubeEnvCfg

    def __init__(self, cfg: FrankaLiftCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


class FrankaLiftCubeIKEnv(LiftCubeEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaLiftCubeEnvCfg

    def __init__(self, cfg: FrankaLiftCubeEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.decimation = 2
        cfg.sim.dt = 0.01
        cfg.robot.spawn.rigid_props.disable_gravity = True
        cfg.robot.actuators["arm"].stiffness = {
            "joint_[1-3]": 4000.0,
            "joint_[5-7]": 1500.0,
        }
        cfg.robot.actuators["arm"].damping = {
            "joint_[1-3]": 1000.0,
            "joint_[4-7]": 500.0,
        }
        cfg.robot.actuators["gripper"].stiffness = 2000.0
        cfg.robot.actuators["gripper"].damping = 200.0
        cfg.skills = ["reach_xyz"]
        super().__init__(cfg, render_mode, **kwargs)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_dof_targets = torch.clamp(actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # Update markers
        self.cube_current_marker.visualize(self.cube_pose_w[:, 0:3], self.cube_pose_w[:, 3:7])
        self.current_marker.visualize(self.robot_tcp_pose_w[:, 0:3], self.robot_tcp_pose_w[:, 3:7])


class FrankaLiftCubeOSCEnv(LiftCubeEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaLiftCubeEnvCfg

    def __init__(self, cfg: FrankaLiftCubeEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.decimation = 2
        cfg.sim.dt = 0.01
        cfg.robot.spawn.rigid_props.disable_gravity = False
        cfg.robot.actuators["arm"].stiffness = 0.0
        cfg.robot.actuators["arm"].damping = 0.0
        cfg.robot.actuators["gripper"].stiffness = 0.0
        cfg.robot.actuators["gripper"].damping = 0.0
        cfg.skills = ["reach_xyz"]
        super().__init__(cfg, render_mode, **kwargs)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_dof_targets = torch.clamp(actions, -self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # Update markers
        self.cube_current_marker.visualize(self.cube_pose_w[:, 0:3], self.cube_pose_w[:, 3:7])
        self.current_marker.visualize(self.robot_tcp_pose_w[:, 0:3], self.robot_tcp_pose_w[:, 3:7])
