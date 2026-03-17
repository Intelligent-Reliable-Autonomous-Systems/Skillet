from skillet.envs.isaac import SkillsDirectRlEnvCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


class FrankaBaseCfg(SkillsDirectRlEnvCfg):
    # env
    episode_length_s = 6.0  # 500 timesteps
    decimation = 2

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    tcp_offset = [0.0, 0.0, 0.1034, 1.0, 0.0, 0.0, 0.0]
    ee_link_name = "panda_hand"
    base_link_name = "panda_link0"
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    skills = ["reach_xyz", "orient_rpy"]

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
    robot = FRANKA_PANDA_CFG
    robot.prim_path = "/World/envs/env_.*/Robot"
