from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from skillet.envs.util import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# Pre-defined configs
##
from isaaclab.managers import SceneEntityCfg

from skillet_tasks.assets.isaac.kinova_gen3_2f85 import KINOVA_GEN3_2F85_CFG

##
# Environment configuration
##


class Gen3LiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.tcp_offset = [0.0, 0.0, 0.120, 1.0, 0.0, 0.0, 0.0]
        # Set Franka as robot
        self.scene.robot = KINOVA_GEN3_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.ee_link_name = "end_effector_link"
        self.base_link_name = "base_link"
        self.gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

        self.skills = []

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["robotiq_85_left_knuckle_joint"],
            open_command_expr={"robotiq_85_left_knuckle_joint": 0.0},
            close_command_expr={"robotiq_85_left_knuckle_joint": 0.8},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "end_effector_link"

        self.obs_asset = SceneEntityCfg("robot")
        self.obs_asset.joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.observations.policy.joint_pos.params = {"asset_cfg": self.obs_asset}
        self.observations.policy.joint_vel.params = {"asset_cfg": self.obs_asset}

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.7, 0.7, 0.7),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Arm/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Arm/end_effector_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.120],
                    ),
                ),
            ],
        )


@configclass
class TeleOpGen3CubeLiftEnvCfg(Gen3LiftCubeEnvCfg):
    """Teleop class for Gen3LiftEnv."""

    def __post_init__(self) -> None:
        """Set up the actions and teleop devices."""
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = KINOVA_GEN3_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            body_name="end_effector_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.120]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=True,
                    sim_device=self.sim.device,
                ),
            },
        )
