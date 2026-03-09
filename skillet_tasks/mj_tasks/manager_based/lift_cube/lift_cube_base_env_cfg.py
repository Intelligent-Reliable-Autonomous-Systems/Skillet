"""Common base environment configuration for Kinova Gen3 tasks."""

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

from skillet.envs.mujoco import ManagerBasedRLEnvCfg
from skillet.envs.util import configclass


@configclass
class LiftSceneCfg(SceneCfg):
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


@configclass
class RewardsCfg:
    """Reward terms for the lift task."""

    lift = RewardTermCfg(
        func=manipulation_mdp.staged_position_reward,
        weight=1.0,
        params={
            "command_name": "lift_height",
            "object_name": "cube",
            "reaching_std": 0.2,
            "bringing_std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        },
    )

    lift_precise = RewardTermCfg(
        func=manipulation_mdp.bring_object_reward,
        weight=1.0,
        params={
            "command_name": "lift_height",
            "object_name": "cube",
            "std": 0.05,
        },
    )

    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

    joint_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )

    joint_vel_hinge = RewardTermCfg(
        func=manipulation_mdp.joint_velocity_hinge_penalty,
        weight=-0.01,
        params={
            "max_vel": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )


@configclass
class ActionsCfg:
    """Action configuration."""

    joint_pos = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.5,  # Override per-robot.
        use_default_offset=True,
    )  # TODO resolve the fact that the gripper is not actuated?


@configclass
class TerminationsCfg:
    """Terminations and Truncation configuration."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    ee_ground_collision = TerminationTermCfg(
        func=manipulation_mdp.illegal_contact,
        params={"sensor_name": "ee_ground_collision"},
    )


@configclass
class CurriculumCfg:
    """Curriculum Cpnfiguration for learning."""

    joint_vel_hinge_weight = CurriculumTermCfg(
        func=manipulation_mdp.reward_weight,
        params={
            "reward_name": "joint_vel_hinge",
            "weight_stages": [
                {"step": 0, "weight": -0.01},
                {"step": 500 * 24, "weight": -0.1},
                {"step": 1000 * 24, "weight": -1.0},
            ],
        },
    )


@configclass
class EventsCfg:
    """Events configuration."""

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )

    fingertip_friction_slide = EventTermCfg(
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
    )

    fingertip_friction_spin = EventTermCfg(
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
    )

    fingertip_friction_roll = EventTermCfg(
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
    )


@configclass
class CommandsCfg:
    lift_height = LiftingCommandCfg(
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


@configclass
class ObservationsCfg:
    """Configuration for observation specification."""

    @configclass
    class PolicyCfg:
        """Observation for actor and critic."""

        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        ee_to_cube = ObservationTermCfg(
            func=manipulation_mdp.ee_to_object_distance,
            params={
                "object_name": "cube",
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        cube_to_goal = ObservationTermCfg(
            func=manipulation_mdp.object_to_goal_distance,
            params={
                "object_name": "cube",
                "command_name": "lift_height",
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        actions = ObservationTermCfg(func=mdp.last_action)

    policy = ObservationGroupCfg(terms=PolicyCfg().__dict__, enable_corruption=True)  # TODO this used to be actor
    critic = ObservationGroupCfg(terms=PolicyCfg().__dict__, enable_corruption=True)


@configclass
class LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for Lift Cube environments."""

    scene: LiftSceneCfg = LiftSceneCfg()
    rewards = RewardsCfg().__dict__
    actions = ActionsCfg().__dict__
    terminations = TerminationsCfg().__dict__
    observations = ObservationsCfg().__dict__
    commands = CommandsCfg().__dict__
    events = EventsCfg().__dict__
    curriculum = CurriculumCfg().__dict__

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="",
        distance=1.5,
        elevation=-5.0,
        azimuth=120.0,
    )

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

    decimation = 2

    episode_length_s = 5.0

    is_finite_horizon = False

    scale_rewards_by_dt = True
