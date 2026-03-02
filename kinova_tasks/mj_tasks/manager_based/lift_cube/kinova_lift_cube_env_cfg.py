"""Kinova Gen3 lift task with joint-space actions."""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg

from kinova_tasks.assets.mujoco.kinova import KINOVA_ACTION_SCALE, get_kinova_robot_cfg
from kinova_tasks.mj_tasks.manager_based.actions import HomeRelativeIKActionCfg
from skillet.envs.util import configclass

from .lift_cube_base_env_cfg import LiftCubeEnvCfg


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
    return spec


@configclass
class KinovaLiftCubeEnvCfg(LiftCubeEnvCfg):
    """Create base Kinova Gen3 lift cube environment configuration.

    Sets up scene entities, fingertip friction, collision sensor, viewer,
    and play mode overrides. Does NOT configure actions (task-specific).

    Args:
        play: If True, configure for evaluation/play mode.

    Returns:
        Manager-based RL environment configuration.

    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.entities = {
            "robot": get_kinova_robot_cfg(),
            "cube": EntityCfg(spec_fn=get_cube_spec),
        }

        # EE site for observations and rewards
        self.observations["policy"].terms["ee_to_cube"].params["asset_cfg"].site_names = ("pinch_site",)
        self.observations["critic"].terms["ee_to_cube"].params["asset_cfg"].site_names = ("pinch_site",)
        self.rewards["lift"].params["asset_cfg"].site_names = ("pinch_site",)

        self.actions["joint_pos"].scale = KINOVA_ACTION_SCALE

        # Fingertip friction randomization (Robotiq 2F-85 pads)
        fingertip_geoms = r"(left|right)_pad[12]"
        self.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
        self.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
        self.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

        # Collision sensor for end-effector ground contact
        assert self.scene.sensors is not None
        for sensor in self.scene.sensors:
            if sensor.name == "ee_ground_collision":
                assert isinstance(sensor, ContactSensorCfg)
                sensor.primary.pattern = "bracelet_link"

        self.viewer.body_name = "base_link"

        # Parallel environments
        self.scene.num_envs = 4096

        # Play mode overrides
        if False:
            self.episode_length_s = int(1e9)
            self.observations["policy"].enable_corruption = False
            self.curriculum = {}
            assert self.commands is not None
            self.commands["lift_height"].resampling_time_range = (4.0, 4.0)


@configclass
class KinovaLiftCubeIKEnvCfg(KinovaLiftCubeEnvCfg):
    """Configuration for IK control."""

    def __post_init__(self):
        super().__post_init__()

        @configclass
        class ActionsCfg:
            ik_pose = HomeRelativeIKActionCfg(
                entity_name="robot",
                actuator_names=("joint_.*",),  # Arm joints only
                frame_name="pinch_site",
                frame_type="site",
                # Home EE pose from default joint config (pinch_site FK)
                home_pos=(0.733607, -0.024850, 0.523015),
                home_quat=(0.5, 0.5, 0.5, 0.5),
                damping=0.05,
                max_dq=0.5,
                position_weight=1.0,
                orientation_weight=1.0,
                joint_limit_weight=0.1,
                posture_weight=0.02,
                pos_scale=0.1,
                ori_scale=0.1,
            )
            gripper = JointPositionActionCfg(
                entity_name="robot",
                actuator_names=("fingers_actuator",),  # Tendon-based gripper
                scale=1.0,
                use_default_offset=True,
            )

        self.actions: ActionsCfg = ActionsCfg()
