"""Kinova Gen3Lite robot configuration for Mujoco."""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from skillet_tasks.assets.utils import update_assets

##
# MJCF and assets.
##

_HERE = Path(__file__).parent
KINOVA_GEN3LITE_GRIPPER_XML: Path = _HERE / "xmls" / "gen3_lite.xml"


def get_gen3lite_assets(meshdir: str) -> dict[str, bytes]:
    """Load Kinova Gen3Lite mesh assets."""
    assets: dict[str, bytes] = {}
    update_assets(assets, KINOVA_GEN3LITE_GRIPPER_XML.parent / "assets", meshdir)
    return assets


def get_gen3lite_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3Lite with Robotiq 2F-85 gripper for position control.

    Includes the 7-DOF arm with position actuators plus the parallel gripper mechanism.
    """
    spec = mujoco.MjSpec.from_file(str(KINOVA_GEN3LITE_GRIPPER_XML))
    spec.assets = get_gen3lite_assets(spec.meshdir)
    return spec


def get_gen3lite_robot_cfg() -> EntityCfg:
    """Get a fresh Kinova Gen3Lite robot configuration instance for lift task.

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
                # Gripper joints - open position
                "right_finger_bottom_joint": 0.0,
                "left_finger_bottom_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        collisions=(),  # Use collisions from XML
        spec_fn=get_gen3lite_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=(".*",),  # Match all joints (arm + gripper)
                    transmission_type=TransmissionType.JOINT,
                ),
                XmlActuatorCfg(
                    target_names_expr=("fingers_actuator",),
                    transmission_type=TransmissionType.TENDON,
                    command_field="position",
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
