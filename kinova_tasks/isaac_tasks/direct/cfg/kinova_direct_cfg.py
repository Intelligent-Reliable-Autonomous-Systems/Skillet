from kinova_tasks.assets.utils import KINOVA_ASSET_DIR
from skillet.envs.isaac import SkillsDirectRLEnvCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils


class KinovaBaseCfg(SkillsDirectRLEnvCfg):
    # env
    episode_length_s = 6.0  # 500 timesteps
    decimation = 2
    action_space = 8
    observation_space = 31
    state_space = 0

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.120, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ee_link_name = "gripper_base_link"
    base_link_name = "base_link"
    gripper_joint_names = ["finger_joint"]

    skills = ["reach_xyz"]

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
