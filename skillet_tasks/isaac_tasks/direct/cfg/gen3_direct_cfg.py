from skillet.envs.isaac import SkillsDirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from dataclasses import MISSING

from skillet_tasks.assets.isaac.kinova_gen3_2f85 import KINOVA_GEN3_2F85_CFG


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
    gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

    skills = MISSING

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
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = KINOVA_GEN3_2F85_CFG
