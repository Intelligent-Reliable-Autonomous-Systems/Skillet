from kinova_tasks.assets.utils import KINOVA_ASSET_DIR
from skillet.envs.isaac import SkillsDirectRLEnvCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils

from kinova_tasks.assets.kinova_gen3_2f85 import KINOVA_GEN3_2F85_CFG


class KinovaBaseCfg(SkillsDirectRLEnvCfg):
    # env
    episode_length_s = 6.0  # 500 timesteps
    decimation = 2
    action_space = 8
    observation_space = 31
    state_space = 0

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.120, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["finger_joint"]

    skills = ["reach_xyz", "orient_rpy", "gripper_oc"]

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