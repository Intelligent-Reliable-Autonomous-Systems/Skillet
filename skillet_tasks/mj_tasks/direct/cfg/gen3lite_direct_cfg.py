from skillet.envs.mujoco import SkillsDirectRlEnvCfg
from mjlab.scene import SceneCfg

from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from dataclasses import MISSING
from mjlab.viewer import ViewerConfig


class BaseSceneCfg(SceneCfg):
    """Configuration for Lift Environment."""

    terrain = TerrainEntityCfg(terrain_type="plane")
    num_envs = 1
    env_spacing = 2.5


class Gen3LiteBaseCfg(SkillsDirectRlEnvCfg):
    # env
    episode_length_s = 6.0  # 500 timesteps
    decimation = 2
    action_space = 7
    observation_space = 27
    state_space = 0

    joint_ids = [0, 1, 2, 3, 4, 5, 0]
    tcp_offset = [0.0, 0.0, 0.130, 0.70710678, 0, 0, 0.70710678]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["right_finger_bottom_joint"]

    skills = MISSING

    # simulation
    sim = SimulationCfg(
        nconmax=55,
        njmax=600,
        mujoco=MujocoCfg(
            timestep=1 / 120,
            iterations=10,
            ls_iterations=20,
            impratio=10,
            cone="elliptic",
        ),
    )

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="base_link",
        distance=1.5,
        elevation=-5.0,
        azimuth=120.0,
    )

    # scene
    scene: BaseSceneCfg = BaseSceneCfg()
