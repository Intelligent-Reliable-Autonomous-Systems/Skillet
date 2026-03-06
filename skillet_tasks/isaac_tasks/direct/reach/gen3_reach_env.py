# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    quat_error_magnitude,
)

from skillet_tasks.isaac_tasks.direct.cfg import Gen3BaseCfg
from skillet_tasks.isaac_tasks.direct.reach.reach_env import ReachEnv
from skillet.envs.util import configclass


@configclass
class Gen3ReachEnvCfg(Gen3BaseCfg):
    action_scale = 0.5
    dof_velocity_scale = 0.1
    skills = []

    # reward scales
    ee_dist_reward_scale = -0.2
    ee_dist_reward_fine_grained_scale = 0.1
    ee_dist_reward_fine_grained_std = 0.1
    ee_orientation_reward_scale = -0.02
    action_rate_reward_scale = -0.0001
    joint_vel_reward_scale = -0.0001

    # EE Pose target ranges
    pos_x = [0.35, 0.65]
    pos_y = [-0.2, 0.2]
    pos_z = [0.15, 0.5]
    roll = [0.0, 0.0]
    pitch = [1.57, 1.57]
    yaw = [-3.14, 3.14]
    ee_ranges = [pos_x, pos_y, pos_z, roll, pitch, yaw]

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class Gen3ReachEnv(ReachEnv):
    """Use this environment for computing actions with RL."""

    cfg: Gen3ReachEnvCfg

    def __init__(self, cfg: Gen3ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


class Gen3ReachIKEnv(ReachEnv):
    """Use this environment when computing actions with a Diff IK controller or the skills environments."""

    cfg: Gen3ReachEnvCfg

    def __init__(self, cfg: Gen3ReachEnvCfg, render_mode: str | None = None, **kwargs):
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
        cfg.skills = ["reach_xyz", "orient_rpy"]
        super().__init__(cfg, render_mode, **kwargs)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_dof_targets = torch.clamp(actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # Update markers (world frame)
        self.goal_marker.visualize(self.goal_ee_pos_w, self.goal_ee_quat_w)
        self.current_marker.visualize(self.robot_ee_pos_w, self.robot_ee_quat_w)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot_tcp_pose_b,
                self.goal_ee_xyz_b,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return compute_rewards_osc(
            self.robot_tcp_pose_b[:, 0:3],
            self.robot_tcp_pose_b[:, 3:7],
            self.goal_ee_pose_b[:, 0:3],
            self.goal_ee_pose_b[:, 3:7],
        )


class Gen3ReachOSCEnv(ReachEnv):
    """Use this environment when computing actions with a OSC controlller."""

    cfg: Gen3ReachEnvCfg

    def __init__(self, cfg: Gen3ReachEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.decimation = 2
        cfg.sim.dt = 0.01
        cfg.robot.actuators["arm"].stiffness = 0.0
        cfg.robot.actuators["arm"].damping = 0.0
        cfg.robot.actuators["gripper"].stiffness = 0.0
        cfg.robot.actuators["gripper"].damping = 0.0
        cfg.ee_link_name = "end_effector_link"
        cfg.skills = ["reach_xyz_osc", "orient_rpy_osc"]
        super().__init__(cfg, render_mode, **kwargs)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.robot_dof_targets = torch.clamp(actions, -self.robot_effort_limits, self.robot_effort_limits)

        # Update markers (world frame)
        self.goal_marker.visualize(self.goal_ee_pos_w, self.goal_ee_quat_w)
        self.current_marker.visualize(self.robot_ee_pos_w, self.robot_ee_quat_w)

    def _apply_action(self):
        self._robot.set_joint_effort_target(self.robot_dof_targets, joint_ids=self.cfg.joint_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot_tcp_pose_b,
                self.goal_ee_xyz_b,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return compute_rewards_osc(
            self.robot_tcp_pose_b[:, 0:3],
            self.robot_tcp_pose_b[:, 3:7],
            self.goal_ee_pose_b[:, 0:3],
            self.goal_ee_pose_b[:, 3:7],
        )


@torch.jit.script
def compute_rewards_osc(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    goal_ee_pos: torch.Tensor,
    goal_ee_quat: torch.Tensor,
) -> torch.Tensor:
    reach_rew = 1 - torch.tanh(10 * torch.norm(ee_pos - goal_ee_pos, dim=1))

    orientation_reward = 1 - torch.tanh(10 * quat_error_magnitude(ee_quat, goal_ee_quat))

    return reach_rew + orientation_reward
