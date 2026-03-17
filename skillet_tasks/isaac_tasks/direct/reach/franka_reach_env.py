from __future__ import annotations

import torch
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    quat_error_magnitude,
)

from skillet.envs.util import configclass
from skillet_tasks.isaac_tasks.direct.cfg import FrankaBaseCfg
from skillet_tasks.isaac_tasks.direct.reach.reach_env import ReachEnv


@configclass
class FrankaReachEnvCfg(FrankaBaseCfg):
    action_space = 9
    observation_space = 33
    state_space = 0

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


class FrankaReachEnv(ReachEnv):
    """Use this environment for computing actions with RL."""

    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


class FrankaReachIKEnv(FrankaReachEnv):
    """Use this environment when computing actions with a Diff IK controller or the skills environments."""

    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        cfg.skills = ["reach_xyz", "orient_rpy"]
        cfg.observation_space = 13
        cfg.action_space = 9
        cfg.state_space = 0

        cfg.decimation = 2
        cfg.sim.dt = 0.01
        cfg.robot.actuators["panda_shoulder"].stiffness = 400.0
        cfg.robot.actuators["panda_shoulder"].damping = 80.0
        cfg.robot.actuators["panda_forearm"].stiffness = 400.0
        cfg.robot.actuators["panda_forearm"].damping = 80.0
        cfg.ee_link_name = "panda_hand"

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


class FrankaReachOSCEnv(FrankaReachEnv):
    """Use this environment when computing actions with a OSC controlller."""

    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.skills = ["reach_xyz_osc", "orient_rpy_osc"]
        cfg.observation_space = 13
        cfg.action_space = 9
        cfg.state_space = 0

        cfg.decimation = 2
        cfg.sim.dt = 0.01
        cfg.robot.actuators["panda_shoulder"].stiffness = 0.0
        cfg.robot.actuators["panda_shoulder"].damping = 0.0
        cfg.robot.actuators["panda_forearm"].stiffness = 0.0
        cfg.robot.actuators["panda_forearm"].damping = 0.0
        cfg.robot.actuators["panda_hand"].stiffness = 0.0
        cfg.robot.actuators["panda_hand"].damping = 0.0
        cfg.ee_link_name = "panda_hand"
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
