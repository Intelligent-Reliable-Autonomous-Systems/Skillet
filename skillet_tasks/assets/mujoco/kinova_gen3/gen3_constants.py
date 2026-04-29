"""Kinova Gen3 robot configuration for Mujoco."""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from skillet_tasks.assets.utils import update_assets

# MJCF and assets.
##

_HERE = Path(__file__).parent
KINOVA_GEN3_GRIPPER_XML: Path = _HERE / "xmls" / "gen3_gripper.xml"


def get_gen3_assets(meshdir: str) -> dict[str, bytes]:
    """Load Kinova Gen3 mesh assets."""
    assets: dict[str, bytes] = {}
    update_assets(assets, KINOVA_GEN3_GRIPPER_XML.parent / "assets", meshdir)
    return assets


def get_gen3_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3 with Robotiq 2F-85 gripper for position control.

    Includes the 7-DOF arm with position actuators plus the parallel gripper mechanism.
    """
    spec = mujoco.MjSpec.from_file(str(KINOVA_GEN3_GRIPPER_XML))
    spec.assets = get_gen3_assets(spec.meshdir)
    return spec


def get_gen3_robot_cfg() -> EntityCfg:
    """Get a fresh Kinova Gen3 robot configuration instance for lift task.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                # Arm joints - ready pose for lifting
                "joint_1": 0.0,  # 0°
                "joint_2": 0.3490658504,  # 20°
                "joint_3": 0.0,  # 0°
                "joint_4": 1.7453292519,  # 100°
                "joint_5": 0.0,  # 0°
                "joint_6": -0.5235987756,  # -30°
                "joint_7": -1.5707963268,  # -90°
                # Gripper joints - open position
                "right_driver_joint": 0.0,
                "left_driver_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        collisions=(),  # Use collisions from XML
        spec_fn=get_gen3_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlPositionActuatorCfg(
                    target_names_expr=(".*",),  # Match all joints (arm + gripper)
                    transmission_type=TransmissionType.JOINT,
                ),
                XmlPositionActuatorCfg(
                    target_names_expr=("fingers_actuator",),
                    transmission_type=TransmissionType.TENDON,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


class Gen3ClosedCfg(EntityCfg):
    """Kinova Gen3 configuration with gripper closed.

    The gripper starts closed and the fingers_actuator ctrl defaults to 255
    (closed) via the init state. Used for tasks like peg-in-hole where the
    robot holds an object throughout the episode.
    """

    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Arm joints - same ready pose
            "joint_1": 0.0,  # 0°
            "joint_2": 0.3490658504,  # 20°
            "joint_3": 0.0,  # 0°
            "joint_4": 1.7453292519,  # 100°
            "joint_5": 0.0,  # 0°
            "joint_6": -0.5235987756,  # -30°
            "joint_7": -1.5707963268,  # -90°
            # Gripper joints - closed position (0.8 = fully closed)
            "right_driver_joint": 0.8,
            "left_driver_joint": 0.8,
        },
        joint_vel={".*": 0.0},
    )
    collisions = ()
    spec_fn = get_gen3_spec
    articulation = (
        EntityArticulationInfoCfg(
            actuators=(
                XmlPositionActuatorCfg(
                    target_names_expr=(".*",),  # Match all joints (arm + gripper)
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


class Gen3ClosedPegCfg(EntityCfg):
    """Get Kinova Gen3 config for peg-in-hole task.

    Arm is rotated 90° at joint_1 to face the workspace side.
    Gripper is closed (driver joints at 1.0) to hold the peg.
    """

    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Arm joints
            "joint_1": 1.5707963268,  # 90°
            "joint_2": 0.5235987756,  # 30°
            "joint_3": 0.0,  # 0°
            "joint_4": 1.5707963268,  # 90°
            "joint_5": 0.0,  # 0°
            "joint_6": 1.0471975512,  # 60°
            "joint_7": -1.5707963268,  # -90°
            # Gripper joints - 60% closed (consistent 4-bar linkage state)
            "right_driver_joint": 0.503,
            "right_coupler_joint": 0.001,
            "right_spring_link_joint": 0.505,
            "right_follower_joint": -0.485,
            "left_driver_joint": 0.503,
            "left_coupler_joint": 0.001,
            "left_spring_link_joint": 0.505,
            "left_follower_joint": -0.485,
        },
        joint_vel={".*": 0.0},
    )
    collisions = ()
    spec_fn = get_gen3_spec
    articulation = (
        EntityArticulationInfoCfg(
            actuators=(
                XmlPositionActuatorCfg(
                    target_names_expr=(".*",),  # Match all joints (arm + gripper)
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
