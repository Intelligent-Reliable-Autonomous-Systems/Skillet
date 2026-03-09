from __future__ import annotations

from mjlab.scene import Scene
from mjlab.sim.sim import Simulation

from skillet.envs.util import configclass
from skillet_tasks.assets.mujoco.kinova_gen3 import get_gen3_robot_cfg
from skillet_tasks.mj_tasks.direct.cfg import Gen3BaseCfg
from skillet_tasks.mj_tasks.direct.reach.reach_env import ReachEnv


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


class Gen3ReachEnv(ReachEnv):
    """Use this environment for computing actions with RL."""

    cfg: Gen3ReachEnvCfg

    def __init__(self, cfg: Gen3ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()

        self.cfg.scene.entities = {"robot": get_gen3_robot_cfg()}

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
