"""Kinova Gen3 environment with a configurable RGBD camera.

Env config + scene setup.
  - Robot:  KINOVA_GEN3_2F85_CFG
  - Camera: RGBDCameraCfg  (spawnable at any world-frame position)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from skillet.envs.isaac import IsaacDirectRlEnv, RGBDCameraCfg, SkillsDirectRlEnvCfg
from skillet.envs.util import configclass
from skillet_tasks.assets.isaac.kinova_gen3_2f85 import KINOVA_GEN3_2F85_CFG


@configclass
class Gen3GenCameraEnvCfg(SkillsDirectRlEnvCfg):
    """Env config for the Kinova Gen3 arm with an RGBD camera.

    Customize the camera spawn pose by overriding ``camera_cfg``::

        cfg = KinovaGenCameraEnvCfg()
        cfg.camera_cfg.pos = (1.0, 0.0, 1.2)   # 1 m in front, 1.2 m high
    """

    # --- env timing ---
    episode_length_s = 5.0
    decimation = 2

    action_space = 8
    observation_space = 25
    state_space = 0

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.0, 0.0, 0.120, 1.0, 0.0, 0.0, 0.0]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

    # --- simulation ---
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

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # --- robot: Kinova Gen3 arm (Kinova_Gen3.usd) ---
    robot = KINOVA_GEN3_2F85_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- camera: mounted on bracelet_link (wrist camera) ---
    # prim_path attaches the camera to the wrist link; it moves with the robot.
    # pos/rot are offsets in the bracelet_link local frame
    camera_cfg: RGBDCameraCfg = RGBDCameraCfg(
        prim_path="/World/envs/env_.*/Robot/Arm/bracelet_link/wrist_mounted_camera",
        pos=(0.0, 0.05, 0.05),
        rot=(0.0, 1.0, 0.0, 0.0),
        width=640,
        height=480,
        convention="ros",
    )

    # --- red cube: spawned on the table, randomised on every reset ---
    red_cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RedCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Workspace randomisation bounds (robot-base frame, table surface z≈0.02)
    cube_pos_x_range: tuple = (0.35, 0.65)
    cube_pos_y_range: tuple = (-0.25, 0.25)

    # --- static side-view camera ---
    # Fixed world-frame camera positioned to the side of the table.
    # pos/rot are in the world frame (convention="world").
    # Adjust pos/rot to reframe the view.
    workspace_camera_cfg: RGBDCameraCfg = RGBDCameraCfg(
        prim_path="/World/envs/env_.*/workspace_camera",
        pos=(0.5611907542841481, 1.14435, 0.72975),  # 1.5 m to the side, 1.0 m high
        rot=(0.0, 0.0, 0.95, -0.51),  # (w, x, z, -y)
        width=640,
        height=480,
        convention="ros",
    )


class Gen3GenCameraEnv(IsaacDirectRlEnv):
    """Kinova Gen3 + RGBD camera environment with a table and two cameras.

    Observations returned from ``_get_observations()`` are a nested dict
    under the ``"policy"`` key::

        {
            "policy": {
                "rgb":          (N, H, W, 3)  float32  [0, 1]  - wrist RGB
                "depth":        (N, H, W, 1)  float32  metres  - wrist depth
                "static_rgb":   (N, H, W, 3)  float32  [0, 1]  - static side-view RGB
                "static_depth": (N, H, W, 1)  float32  metres  - static side-view depth
                "joint_pos":    (N, num_joints) float32
                "joint_vel":    (N, num_joints) float32
            }
        }
    """

    cfg: Gen3GenCameraEnvCfg

    def __init__(self, cfg: Gen3GenCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # joint limit tensors (arm joints only)
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.cfg.joint_ids)), device=self.device)
        self.default_joint_pos = self._robot.data.default_joint_pos[:, self.cfg.joint_ids]

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # Table
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Red cube
        self._red_cube = RigidObject(self.cfg.red_cube_cfg)
        self.scene.rigid_objects["red_cube"] = self._red_cube

        # Wrist camera
        isaac_cam_cfg = self.cfg.camera_cfg.to_isaac_cfg()
        self._camera = Camera(isaac_cam_cfg)
        self.scene.sensors["camera"] = self._camera

        # Static side-view camera
        workspace_isaac_cam_cfg = self.cfg.workspace_camera_cfg.to_isaac_cfg()
        self._workspace_camera = Camera(workspace_isaac_cam_cfg)
        self.scene.sensors["workspace_camera"] = self._workspace_camera

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone + replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    # ------------------------------------------------------------------
    # Step stubs
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor, action_spec: ActionSpec = None):
        self.actions = actions.clone()
        targets = self.default_joint_pos + self.actions
        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.cfg.joint_ids)

    def _get_observations(self) -> dict:
        cam_out = self._camera.data.output
        workspace_cam_out = self._workspace_camera.data.output

        # RGB: uint8 (N, H, W, 3) → float32 [0, 1]
        rgb = cam_out["rgb"].float() / 255.0
        static_rgb = workspace_cam_out["rgb"].float() / 255.0

        # Depth: float32 (N, H, W, 1) in metres
        depth = cam_out["distance_to_image_plane"]
        static_depth = workspace_cam_out["distance_to_image_plane"]

        joint_pos = self._robot.data.joint_pos[:, self.cfg.joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.cfg.joint_ids]

        return {
            "policy": {
                "rgb": rgb,
                "depth": depth,
                "workspace_rgb": static_rgb,
                "workspace_depth": static_depth,
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
            }
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), truncated

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids][:, self.cfg.joint_ids]
        # joint_vel must cover ALL robot joints, not just the arm subset
        joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)[env_ids]
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self._robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self._robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

        # Randomize red cube position on the table workspace.
        # Offsets are in the robot-base frame; add root_pos_w to get world coords
        num = len(env_ids)
        base_pos_w = self._robot.data.root_pos_w[env_ids]  # (num, 3)
        cube_pos = base_pos_w.clone()
        cube_pos[:, 0] += self.cfg.cube_pos_x_range[0] + (
            self.cfg.cube_pos_x_range[1] - self.cfg.cube_pos_x_range[0]
        ) * torch.rand(num, device=self.device)
        cube_pos[:, 1] += self.cfg.cube_pos_y_range[0] + (
            self.cfg.cube_pos_y_range[1] - self.cfg.cube_pos_y_range[0]
        ) * torch.rand(num, device=self.device)
        cube_pos[:, 2] = base_pos_w[:, 2] + 0.02
        cube_quat = torch.zeros((num, 4), device=self.device)
        cube_quat[:, 0] = 1.0
        self._red_cube.write_root_pose_to_sim(torch.cat([cube_pos, cube_quat], dim=-1), env_ids)
        self._red_cube.write_root_velocity_to_sim(torch.zeros((num, 6), device=self.device), env_ids)
