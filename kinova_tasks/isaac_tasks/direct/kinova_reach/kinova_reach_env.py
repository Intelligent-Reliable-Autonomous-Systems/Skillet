# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz, sample_uniform
from isaacsim.core.utils.torch.transformations import tf_combine

from kinova_tasks.isaac_tasks.direct.cfg import KinovaBaseCfg
from skillet.envs.util import configclass


@configclass
class KinovaReachEnvCfg(KinovaBaseCfg):
    action_scale = 0.5
    dof_velocity_scale = 0.1

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


class KinovaReachEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: KinovaReachEnvCfg

    def __init__(self, cfg: KinovaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Goal poses and end effector positions
        self.ee_ranges = torch.tensor(self.cfg.ee_ranges, device=self.device)
        self.goal_ee_pose_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.goal_ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)

        self.robot_ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions, device=self.device)

        # Limits and targets
        self.robot_dof_lower_limits = self._robot.data.joint_pos_limits[0, :, 0].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_upper_limits = self._robot.data.joint_pos_limits[0, :, 1].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_effort_limits = self._robot.data.joint_effort_limits[0, :].to(device=self.device)[self.cfg.joint_ids]
        self.robot_dof_lower_limits[self.robot_dof_lower_limits == -float("inf")] = -torch.pi
        self.robot_dof_upper_limits[self.robot_dof_upper_limits == float("inf")] = torch.pi

        self.default_joint_pos = self._robot.data.default_joint_pos[:, self.cfg.joint_ids]
        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.cfg.joint_ids)), device=self.device)

        self.cfg.ee_link_idx = self._robot.find_bodies(self.cfg.ee_link_name)[0][0]

        # Markers
        self.goal_marker = VisualizationMarkers(cfg=self.cfg.goal_pose_visualizer_cfg)
        self.current_marker = VisualizationMarkers(cfg=self.cfg.current_pose_visualizer_cfg)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-5.0, 5.0)
        targets = self.default_joint_pos + self.actions * self.cfg.action_scale
        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # Update markers (world frame)
        self.goal_marker.visualize(self.goal_ee_pos_w, self.goal_ee_quat_w)
        self.current_marker.visualize(self.robot_ee_pos_w, self.robot_ee_quat_w)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.cfg.joint_ids)

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros((self.num_envs), dtype=torch.bool), truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return compute_rewards(
            self.actions,
            self.prev_actions,
            self._robot.data.joint_pos[:, self.cfg.joint_ids],
            self._robot.data.joint_vel[:, self.cfg.joint_ids],
            self.goal_ee_pos_w,
            self.goal_ee_quat_w,
            self.robot_ee_pos_w,
            self.robot_ee_quat_w,
            self.cfg.ee_dist_reward_scale,
            self.cfg.ee_dist_reward_fine_grained_scale,
            self.cfg.ee_dist_reward_fine_grained_std,
            self.cfg.ee_orientation_reward_scale,
            self.cfg.action_rate_reward_scale,
            self.cfg.joint_vel_reward_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            + sample_uniform(
                -0.125,
                0.125,
                (len(env_ids), self._robot.num_joints),
                self.device,
            )
        )[:, self.cfg.joint_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.cfg.joint_ids)

        # target state
        base_pos_w = self._robot.data.root_pos_w[env_ids]
        base_quat_w = self._robot.data.root_quat_w[env_ids]
        self.goal_ee_pose_b[env_ids] = (
            self.ee_ranges[:, 0]
            + (self.ee_ranges[:, 1] - self.ee_ranges[:, 0])
            * torch.rand((self.num_envs, len(self.ee_ranges)), device=self.device)
        )[env_ids]
        goal_ee_pos_b = self.goal_ee_pose_b[env_ids, :3]
        goal_ee_quat_b = quat_from_euler_xyz(
            self.goal_ee_pose_b[env_ids, 3], self.goal_ee_pose_b[env_ids, 4], self.goal_ee_pose_b[env_ids, 5]
        )

        # World frame
        self.goal_ee_quat_w[env_ids], self.goal_ee_pos_w[env_ids] = tf_combine(
            base_quat_w, base_pos_w, goal_ee_quat_b, goal_ee_pos_b
        )

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._robot.data.joint_pos[:, self.cfg.joint_ids]
                - self._robot.data.default_joint_pos[:, self.cfg.joint_ids],
                self._robot.data.joint_vel[:, self.cfg.joint_ids],
                self.goal_ee_pose_b,
                self.prev_actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Docstring for _compute_intermediate_values

        :param self: Description
        :param env_ids: Description
        :type env_ids: torch.Tensor | None

        Compute vlaues in the world frame
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.robot_ee_quat_w[env_ids] = self._robot.data.body_quat_w[env_ids, self.cfg.ee_link_idx]
        self.robot_ee_pos_w[env_ids] = self._robot.data.body_pos_w[env_ids, self.cfg.ee_link_idx]

        self.prev_actions[env_ids] = torch.clone(self.actions[env_ids])


class KinovaReachIKEnv(KinovaReachEnv):
    """Use this environment when computing actions with a Diff IK controller or the skills environments."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: KinovaReachEnvCfg

    def __init__(self, cfg: KinovaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.robot_dof_targets = torch.clamp(actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # Update markers (world frame)
        self.goal_marker.visualize(self.goal_ee_pos_w, self.goal_ee_quat_w)
        self.current_marker.visualize(self.robot_ee_pos_w, self.robot_ee_quat_w)


class KinovaReachOSCEnv(KinovaReachEnv):
    """Use this environment when computing actions with a OSC controlller."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: KinovaReachEnvCfg

    def __init__(self, cfg: KinovaReachEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.robot.actuators["arm"].stiffness = 0.0
        cfg.robot.actuators["arm"].damping = 0.0
        cfg.robot.actuators["gripper"].stiffness = 0.0
        cfg.robot.actuators["gripper"].damping = 0.0
        super().__init__(cfg, render_mode, **kwargs)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.robot_dof_targets = torch.min(actions, self.robot_effort_limits)

        # Update markers (world frame)
        self.goal_marker.visualize(self.goal_ee_pos_w, self.goal_ee_quat_w)
        self.current_marker.visualize(self.robot_ee_pos_w, self.robot_ee_quat_w)

    def _apply_action(self):
        self._robot.set_joint_effort_target(self.robot_dof_targets, joint_ids=self.cfg.joint_ids)


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    goal_ee_pos: torch.Tensor,
    goal_ee_quat: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ee_dist_reward_scale: float,
    ee_dist_reward_fine_grained_scale: float,
    ee_dist_reward_fine_grained_std: float,
    ee_orientation_reward_scale: float,
    action_rate_reward_scale: float,
    joint_vel_reward_scale: float,
) -> torch.Tensor:
    ee_distance = torch.norm(ee_pos - goal_ee_pos, dim=1)
    dist_reward = ee_dist_reward_scale * ee_distance
    dist_fine_grained_rew = ee_dist_reward_fine_grained_scale * (
        1 - torch.tanh(ee_distance / ee_dist_reward_fine_grained_std)
    )

    orientation_reward = ee_orientation_reward_scale * quat_error_magnitude(ee_quat, goal_ee_quat)

    action_rate_reward = action_rate_reward_scale * torch.sum(torch.square(actions - prev_actions), dim=1)
    joint_vel_reward = (joint_vel_reward_scale * torch.sum(torch.square(joint_vel), dim=1)).clamp(-1, 1)

    return (dist_reward + dist_fine_grained_rew + orientation_reward + joint_vel_reward + action_rate_reward).clamp(
        -1, 1
    )
