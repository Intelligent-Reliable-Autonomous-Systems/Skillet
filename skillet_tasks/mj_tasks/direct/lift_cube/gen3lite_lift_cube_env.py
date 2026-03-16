from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation

from skillet.envs.util import configclass
from skillet_tasks.assets.mujoco.kinova_gen3lite import get_gen3lite_robot_cfg
from skillet_tasks.mj_tasks.direct.cfg import Gen3LiteBaseCfg

from .lift_cube_env import LiftCubeEnv


@configclass
class Gen3LiteLiftCubeEnvCfg(Gen3LiteBaseCfg):
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


class Gen3LiteLiftCubeEnv(LiftCubeEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Gen3LiteLiftCubeEnvCfg

    def __init__(self, cfg: Gen3LiteLiftCubeEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.skills = []
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()

        def get_cube_spec(cube_size: float = 0.025, mass: float = 0.06) -> mujoco.MjSpec:
            """Create a cube object specification."""
            spec = mujoco.MjSpec()
            body = spec.worldbody.add_body(name="cube")
            body.add_freejoint(name="cube_joint")
            body.add_geom(
                name="cube_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(cube_size,) * 3,
                mass=mass,
                rgba=(0.2, 0.6, 0.9, 1.0),
            )
            spec.worldbody.add_geom(
                name="floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=(2, 2, 0.1),
                pos=(0, 0, 0),
                rgba=(0.8, 0.8, 0.8, 1.0),
            )

            return spec

        self.cfg.scene.entities = {"robot": get_gen3lite_robot_cfg(), "cube": EntityCfg(spec_fn=get_cube_spec)}

        """self.cfg.scene.sensors = (
            ContactSensorCfg(
                name="ee_ground_collision",
                primary=ContactMatch(
                    mode="subtree",
                    pattern=[
                        "left_silicone_pad",
                        "right_silicone_pad",
                    ],  # Set per-robot (e.g., "link_6" for all geoms in end-effector).
                    entity="robot",
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found",),
                reduce="none",
                num_slots=1,
            ),
        )"""

        # Initialize scene and simulation.
        self.scene = Scene(self.cfg.scene, device=self.cfg.sim.device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            model=self.scene.compile(),
            device=self.cfg.sim.device,
        )

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )

        # Get the robot from the scene
        self.robot = self.scene.entities["robot"]
        self.cube = self.scene.entities["cube"]
