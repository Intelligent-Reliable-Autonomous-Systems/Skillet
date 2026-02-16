# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform, subtract_frame_transforms
from isaacsim.core.utils.torch.transformations import tf_combine

from skillet.envs.isaac import SkillsDirectRLEnvCfg
from skillet.envs.util import configclass


@configclass
class KinovaLiftCubeEnvCfg(SkillsDirectRLEnvCfg):
    # env
    episode_length_s = 2.0  # 500 timesteps
    decimation = 2
    action_space = 8
    observation_space = 25
    state_space = 0

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KINOVA_ASSET_DIR}/robots/kinova/kinova_gen3_robotiq_2f_85_action_graph.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.523599,
                "joint_3": 0.0,
                "joint_4": 1.5708,
                "joint_5": 0.0,
                "joint_6": 0.785398,
                "joint_7": 0.0,
                "finger_joint": 0.0,  # left outer knuckle joint for manipulation
                "right_outer_knuckle_joint": 0.0,
                "left_outer_finger_joint": 0.0,
                "right_outer_finger_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
                "right_inner_finger_knuckle_joint": 0.0,
                "left_inner_finger_knuckle_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-7]"],
                velocity_limit_sim=100.0,
                effort_limit_sim={
                    "joint_[1-2]": 80.0,
                    "joint_[3]": 40.0,
                    "joint_[4-7]": 20.0,
                },
                stiffness={
                    "joint_[1-3]": 4000.0,
                    "joint_[5-7]": 1500.0,
                },
                damping={
                    "joint_[1-3]": 1000.0,
                    "joint_[4-7]": 500.0,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[
                    "finger_joint",
                    "right_outer_knuckle_joint",
                    "left_outer_finger_joint",
                    "right_outer_finger_joint",
                    "left_inner_finger_joint",
                    "right_inner_finger_joint",
                    "right_inner_finger_knuckle_joint",
                    "left_inner_finger_knuckle_joint",
                ],
                effort_limit_sim=10,
                stiffness=2000.0,
                damping=200.0,
            ),
        },
    )

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

    tcp_offset_b = [0.120, 0.0, 0.0]
    tcp_offset_w = [0.0, 1.0, 0.0, 0.0]

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
    # init_pos_x = [0.2, 0.6]
    init_pos_x = [0.5, 0.5]
    # init_pos_y = [-0.3, 0.3]
    init_pos_y = [0.1, 0.1]
    init_pos_z = [0.0, 0.0]
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


class KinovaLiftCubeEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: KinovaLiftCubeEnvCfg

    def __init__(self, cfg: KinovaLiftCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions, device=self.device)

        self.cube_init_ranges = torch.tensor(self.cfg.cube_init_ranges, device=self.device)
        self.cube_goal_ranges = torch.tensor(self.cfg.cube_goal_ranges, device=self.device)
        self.tcp_offset_pos = (
            torch.tensor(self.cfg.tcp_offset_b, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.tcp_offset_quat = (
            torch.tensor(self.cfg.tcp_offset_w, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Cube position
        self.cube_pose_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.cube_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_goal_pose_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.cube_goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_goal_quat_w = torch.zeros((self.num_envs, 4), device=self.device)

        # create auxiliary variables for joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_lower_limits[self.robot_dof_lower_limits == -float("inf")] = -torch.pi
        self.robot_dof_upper_limits[self.robot_dof_upper_limits == float("inf")] = torch.pi

        self.default_joint_pos = self._robot.data.default_joint_pos[:, self.cfg.joint_ids]

        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.cfg.joint_ids)), device=self.device)

        self.cfg.ee_link_idx = self._robot.find_bodies("gripper_base_link")[0][0]

        self.robot_ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.cube_goal_marker = VisualizationMarkers(cfg=self.cfg.cube_goal_pose_visualizer_cfg)
        self.cube_current_marker = VisualizationMarkers(cfg=self.cfg.cube_current_pose_visualizer_cfg)
        self.current_marker = VisualizationMarkers(cfg=self.cfg.current_pose_visualizer_cfg)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

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

        # Update markers
        robot_tcp_rot, robot_tcp_pos = tf_combine(
            self.robot_ee_quat_w, self.robot_ee_pos_w, self.tcp_offset_quat, self.tcp_offset_pos
        )

        self.cube_current_marker.visualize(self.cube_pos_w, self.cube_quat_w)
        self.current_marker.visualize(robot_tcp_pos, robot_tcp_rot)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.cfg.joint_ids)

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate if max length is reached or cube goes below minimum height"""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated_pos = self.cube_pos_w[:, 2] < -0.05
        terminated_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1) > 1000.0

        return torch.logical_or(terminated_vel, terminated_pos), truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return compute_rewards(
            self.actions,
            self.prev_actions,
            self._robot.data.joint_pos[:, self.cfg.joint_ids],
            self._robot.data.joint_vel[:, self.cfg.joint_ids],
            self.robot_ee_pos_w,
            self.robot_ee_quat_w,
            self.tcp_offset_pos,
            self.tcp_offset_quat,
            self.cube_pos_w,
            self.cube_goal_pos_w,
            self.cfg.min_height,
            self.cfg.cube_reach_reward_scale,
            self.cfg.cube_reach_reward_std,
            self.cfg.cube_lift_reward_scale,
            self.cfg.cube_goal_reward_scale,
            self.cfg.cube_goal_reward_std,
            self.cfg.cube_goal_fine_grained_scale,
            self.cfg.cube_goal_fine_grained_std,
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
        joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)[env_ids]
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self._robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self._robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

        # Reset the cube position
        self._reset_cube_pose(env_ids=env_ids)

        self._object.write_root_pose_to_sim(torch.cat((self.cube_pos_w, self.cube_quat_w), dim=1)[env_ids], env_ids)
        self._object.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._robot.data.joint_pos[:, self.cfg.joint_ids]
                - self._robot.data.default_joint_pos[:, self.cfg.joint_ids],
                self._robot.data.joint_vel[:, self.cfg.joint_ids],
                self.cube_pos_b,
                self.cube_goal_pose_b[:, :3],
                self.prev_actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Docstring for _compute_intermediate_values.

        :param self: Description
        :param env_ids: Description
        :type env_ids: torch.Tensor | None

        Compute vlaues in the world frame
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.cube_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]
        self.cube_quat_w[env_ids] = self._object.data.root_quat_w[env_ids]

        self.robot_ee_quat_w[env_ids] = self._robot.data.body_quat_w[env_ids, self.cfg.ee_link_idx]
        self.robot_ee_pos_w[env_ids] = self._robot.data.body_pos_w[env_ids, self.cfg.ee_link_idx]

        self.prev_actions[env_ids] = torch.clone(self.actions[env_ids])

        # Object position in robot frame
        self.cube_pos_b[env_ids], _ = subtract_frame_transforms(
            self._robot.data.root_pos_w[env_ids], self._robot.data.root_quat_w[env_ids], self.cube_pos_w[env_ids]
        )

    def _reset_cube_pose(self, env_ids: torch.Tensor | None):
        """Reset the position of the cube"""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Set cube position and rotation
        base_pos_w = self._robot.data.root_pos_w[env_ids]
        base_quat_w = self._robot.data.root_quat_w[env_ids]

        self.cube_pose_b[env_ids] = self.cube_init_ranges[:, 0] + (
            self.cube_init_ranges[:, 1] - self.cube_init_ranges[:, 0]
        )  # * torch.rand((len(env_ids), len(self.cube_init_ranges)), device=self.device)

        cube_pos_b = self.cube_pose_b[env_ids, :3]
        cube_quat_b = quat_from_euler_xyz(
            self.cube_pose_b[env_ids, 3], self.cube_pose_b[env_ids, 4], self.cube_pose_b[env_ids, 5]
        )
        # In world frame
        self.cube_quat_w[env_ids], self.cube_pos_w[env_ids] = tf_combine(
            base_quat_w, base_pos_w, cube_quat_b, cube_pos_b
        )

        # Set cube goal position and rotation
        self.cube_goal_pose_b[env_ids] = self.cube_goal_ranges[:, 0] + (
            self.cube_goal_ranges[:, 1] - self.cube_goal_ranges[:, 0]
        ) * torch.rand((len(env_ids), len(self.cube_goal_ranges)), device=self.device)
        cube_goal_pos_b = self.cube_goal_pose_b[env_ids, :3]
        cube_goal_quat_b = quat_from_euler_xyz(
            self.cube_goal_pose_b[env_ids, 3], self.cube_goal_pose_b[env_ids, 4], self.cube_goal_pose_b[env_ids, 5]
        )
        # In world frame
        self.cube_goal_quat_w[env_ids], self.cube_goal_pos_w[env_ids] = tf_combine(
            base_quat_w, base_pos_w, cube_goal_quat_b, cube_goal_pos_b
        )


class KinovaLiftCubeSkillEnv(KinovaLiftCubeEnv):
    """Lift cube environment for skillet."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: KinovaLiftCubeEnvCfg

    def __init__(self, cfg: KinovaLiftCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.robot_dof_targets = torch.clamp(actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # Update markers
        robot_tcp_rot, robot_tcp_pos = tf_combine(
            self.robot_ee_quat_w, self.robot_ee_pos_w, self.tcp_offset_quat, self.tcp_offset_pos
        )

        self.cube_current_marker.visualize(self.cube_pos_w, self.cube_quat_w)
        self.current_marker.visualize(robot_tcp_pos, robot_tcp_rot)


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    tcp_offset_pos: torch.Tensor,
    tcp_offset_quat: torch.Tensor,
    cube_pos_w: torch.Tensor,
    cube_goal_pos_w: torch.Tensor,
    min_height: float,
    cube_reach_reward_scale: float,
    cube_reach_reward_std: float,
    cube_lift_reward_scale: float,
    cube_goal_reward_scale: float,
    cube_goal_reward_std: float,
    cube_goal_fine_grained_scale: float,
    cube_goal_fine_grained_std: float,
    action_rate_reward_scale: float,
    joint_vel_reward_scale: float,
):
    _, robot_tcp_pos = tf_combine(ee_quat, ee_pos, tcp_offset_quat, tcp_offset_pos)

    # Distance of the end-effector to the object
    cube_tcp_distance = torch.norm(cube_pos_w - robot_tcp_pos, dim=1)
    cube_tcp_distance_reward = cube_reach_reward_scale * (1 - torch.tanh(cube_tcp_distance / cube_reach_reward_std))

    gripper_reward = 5.0 * (cube_tcp_distance < 0.05) * torch.clamp(actions[:, -1], 0.0, 1.0)

    cube_lifted_reward = cube_lift_reward_scale * torch.where(cube_pos_w[:, 2] > min_height, 1.0, 0.0)

    cube_distance = torch.norm(cube_goal_pos_w - cube_pos_w, dim=1)

    cube_dist_reward = (
        cube_goal_reward_scale
        * (cube_pos_w[:, 2] > min_height)
        * (1 - torch.tanh(cube_distance / cube_goal_reward_std))
    )
    cube_dist_reward_fine_grained = (
        cube_goal_fine_grained_scale
        * (cube_pos_w[:, 2] > min_height)
        * (1 - torch.tanh(cube_distance / cube_goal_fine_grained_std))
    )
    action_rate_reward = (action_rate_reward_scale * torch.sum(torch.square(actions - prev_actions), dim=1)).clamp(
        -1, 1
    )
    joint_vel_reward = (joint_vel_reward_scale * torch.sum(torch.square(joint_vel), dim=1)).clamp(-1, 1)

    return (
        gripper_reward
        + cube_tcp_distance_reward
        + cube_lifted_reward
        + cube_dist_reward
        + cube_dist_reward_fine_grained
        + action_rate_reward
        + joint_vel_reward
    )
