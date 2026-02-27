"""Common base environment configuration for Kinova Gen3 tasks."""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from kinova_tasks.assets.mujoco.kinova import get_kinova_robot_cfg
from skillet.envs.mujoco import ManagerBasedRLEnvCfg
from skillet.envs.util import configclass


@configclass
class MJKinovaLiftCubeCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        num_envs=1,
        env_spacing=1.0,
        sensors=(
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
        ),
    )

    observations = {
        "actor": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=mdp.joint_pos_rel,
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "joint_vel": ObservationTermCfg(
                    func=mdp.joint_vel_rel,
                    noise=Unoise(n_min=-1.5, n_max=1.5),
                ),
                "ee_to_cube": ObservationTermCfg(
                    func=manipulation_mdp.ee_to_object_distance,
                    params={
                        "object_name": "cube",
                        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
                    },
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "cube_to_goal": ObservationTermCfg(
                    func=manipulation_mdp.object_to_goal_distance,
                    params={
                        "object_name": "cube",
                        "command_name": "lift_height",
                    },
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "actions": ObservationTermCfg(func=mdp.last_action),
            },
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=mdp.joint_pos_rel,
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "joint_vel": ObservationTermCfg(
                    func=mdp.joint_vel_rel,
                    noise=Unoise(n_min=-1.5, n_max=1.5),
                ),
                "ee_to_cube": ObservationTermCfg(
                    func=manipulation_mdp.ee_to_object_distance,
                    params={
                        "object_name": "cube",
                        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
                    },
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "cube_to_goal": ObservationTermCfg(
                    func=manipulation_mdp.object_to_goal_distance,
                    params={
                        "object_name": "cube",
                        "command_name": "lift_height",
                    },
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "actions": ObservationTermCfg(func=mdp.last_action),
            },
            enable_corruption=False,
        ),
    }
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }
    commands = {
        "lift_height": LiftingCommandCfg(
            entity_name="cube",
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
            object_pose_range=LiftingCommandCfg.ObjectPoseRangeCfg(
                x=(0.2, 0.4),
                y=(-0.2, 0.2),
                z=(0.02, 0.05),
                yaw=(-3.14, 3.14),
            ),
        )
    }
    events = {
        # For positioning the base of the robot at env_origins.
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "fingertip_friction_slide": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "field": "geom_friction",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.3, 1.5),
            },
        ),
        "fingertip_friction_spin": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "field": "geom_friction",
                "distribution": "log_uniform",
                "axes": [1],
                "ranges": (1e-4, 2e-2),
            },
        ),
        "fingertip_friction_roll": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "field": "geom_friction",
                "distribution": "log_uniform",
                "axes": [2],
                "ranges": (1e-5, 5e-3),
            },
        ),
    }

    rewards = {
        "lift": RewardTermCfg(
            func=manipulation_mdp.staged_position_reward,
            weight=1.0,
            params={
                "command_name": "lift_height",
                "object_name": "cube",
                "reaching_std": 0.2,
                "bringing_std": 0.3,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "lift_precise": RewardTermCfg(
            func=manipulation_mdp.bring_object_reward,
            weight=1.0,
            params={
                "command_name": "lift_height",
                "object_name": "cube",
                "std": 0.05,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.01,
            params={
                "max_vel": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ee_ground_collision": TerminationTermCfg(
            func=manipulation_mdp.illegal_contact,
            params={"sensor_name": "ee_ground_collision"},
        ),
    }

    curriculum = {
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=manipulation_mdp.reward_weight,
            params={
                "reward_name": "joint_vel_hinge",
                "weight_stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 500 * 24, "weight": -0.1},
                    {"step": 1000 * 24, "weight": -1.0},
                ],
            },
        ),
    }
    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="",  # Set per-robot.
        distance=1.5,
        elevation=-5.0,
        azimuth=120.0,
    )

    sim = SimulationCfg(
        nconmax=55,
        njmax=600,
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
            impratio=10,
            cone="elliptic",
        ),
    )

    decimation = 4

    episode_length_s = 20.0

    is_finite_horizon = False

    scale_rewards_by_dt = True


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


def kinova_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create base Kinova Gen3 lift cube environment configuration.

    Sets up scene entities, fingertip friction, collision sensor, viewer,
    and play mode overrides. Does NOT configure actions (task-specific).

    Args:
        play: If True, configure for evaluation/play mode.

    Returns:
        Manager-based RL environment configuration.

    """
    cfg = MJKinovaLiftCubeCfg()

    # Scene entities
    cfg.scene.entities = {
        "robot": get_kinova_robot_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
    }

    # EE site for observations and rewards
    cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = ("pinch_site",)
    cfg.rewards["lift"].params["asset_cfg"].site_names = ("pinch_site",)

    # Fingertip friction randomization (Robotiq 2F-85 pads)
    fingertip_geoms = r"(left|right)_pad[12]"
    cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
    cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
    cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

    # Collision sensor for end-effector ground contact
    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = "bracelet_link"

    # Viewer
    cfg.viewer.body_name = "base_link"

    # Parallel environments
    cfg.scene.num_envs = 4096

    # Play mode overrides
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        assert cfg.commands is not None
        cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

    return cfg
