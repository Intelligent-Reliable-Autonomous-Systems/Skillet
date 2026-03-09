from skillet.envs.mujoco import SkillsDirectRLEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from dataclasses import MISSING


class BaseSceneCfg(SceneCfg):
    """Configuration for Lift Environment."""

    terrain = TerrainImporterCfg(terrain_type="plane")
    num_envs = 1
    env_spacing = 2.5
    sensors = (
        ContactSensorCfg(
            name="ee_ground_collision",
            primary=ContactMatch(
                mode="subtree",
                pattern="",  # Set per-robot (e.g., "link_6" for all geoms in end-effector).
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )


class Gen3BaseCfg(SkillsDirectRLEnvCfg):
    # env
    episode_length_s = 6.0  # 500 timesteps
    decimation = 2
    action_space = 8
    observation_space = 31
    state_space = 0

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.0, 0.0, 0.120, 1.0, 0.0, 0.0, 0.0]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["right_driver_joint"]

    skills = MISSING

    # simulation
    sim = SimulationCfg(
        nconmax=55,
        njmax=600,
        mujoco=MujocoCfg(
            timestep=0.01,
            iterations=10,
            ls_iterations=20,
            impratio=10,
            cone="elliptic",
        ),
    )

    # scene
    scene: BaseSceneCfg = BaseSceneCfg()
