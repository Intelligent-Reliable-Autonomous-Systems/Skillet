"""kortex_env.py.

Main Kortex

Written by Will Solow, 2026

"""

import math
import time
from abc import abstractmethod
from typing import Any

import numpy as np
import pinocchio as pin
import torch
from kortex_api.autogen.messages import Base_pb2

from skillet.core.math import convert_quat
from skillet.core.spaces import ActionSpec
from skillet.envs.compatibility import SkilletGymEnv
from skillet.envs.compatibility.s2r import CollisionProximityMonitor
from skillet.envs.kortex.kortex_bridge import DeviceConnection
from skillet.envs.util import configure_seed

from .kortex_env_cfg import KortexEnvCfg


class KortexEnv(SkilletGymEnv):
    """The superclass for the Kortex workflow to design environments.

    This class implements the core functionality for reinforcement learning (RL)
    environments. However, it is designed to interface with Kortex and thus calls the corresponding
    API calls etc.

    """

    _current_joint_positions: np.ndarray
    _current_joint_velocities: np.ndarray
    _current_joint_efforts: np.ndarray
    _robot_links: list[str]
    _robot_joints: list[str]
    _current_robot_body_pose_w: np.ndarray
    _current_robot_root_pose_w: np.ndarray
    _current_jacobians: np.ndarray
    _current_upper_joint_limits: np.ndarray
    _current_lower_joint_limits: np.ndarray
    _current_gravity_vector: np.ndarray
    _current_mass_matrices: np.ndarray
    _current_robot_body_vel_w: np.ndarray
    _current_joint_centers: np.ndarray
    _current_prev_actions: np.ndarray

    def __init__(
        self,
        cfg: KortexEnvCfg,
        kortex_connection: DeviceConnection,
        render_mode: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
            kwargs: Additoinal arguments

        """
        self.kortex_connection = kortex_connection
        self.kortex, self.kortex_c = kortex_connection.open()
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        self.render_mode = render_mode

        self.kortex_model: pin.Model = pin.buildModelFromUrdf(self.cfg.urdf_path)
        self.kortex_data: pin.Data = self.kortex_model.createData()

        self._sim_step_counter = 0
        # -- counter for curriculum
        self._common_step_counter = 0
        # -- init buffers
        self._episode_length_buf = 0
        self.reset_terminated = False
        self.reset_time_outs = False
        self.reset_buf = False
        # allocate dictionary to store metrics
        self.extras: dict[str, Any] = {}

        # setup the action and observation spaces for Gym
        self._next_step_time = time.perf_counter()

        # Set up the collision checker and add the table
        self._collision_checker = CollisionProximityMonitor(
            self.cfg.urdf_path, self.cfg.srdf_path, self.cfg.assets_dir, distance_threshold=0.01, dt=self.physics_dt
        )
        self._collision_checker.add_box_obstacle(
            name="table",
            size_xyz=[2.0, 2.0, 0.05],
            xyz=[0.0, 0.0, -0.05],
        )

        print("[INFO][KortexEnv] Completed Environment Setup")

    def __del__(self) -> None:
        """Cleanup for the environment."""
        self.kortex_connection.close()
        self.close()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.tensor([self._episode_length_buf], device=self.device)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._episode_length_buf = value.squeeze().item()

    """
    Skillet Gymansium Interface Properties.
    """

    @property
    def _prev_actions(self) -> torch.Tensor:
        """Return the previous action taken in the environment."""
        return torch.as_tensor(self._current_prev_actions, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _joint_positions(self) -> torch.Tensor:
        """Return current joint positions."""
        return torch.as_tensor(self._current_joint_positions, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _joint_velocities(self) -> torch.Tensor:
        """Return current joint velocities."""
        return torch.as_tensor(self._current_joint_velocities, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _joint_efforts(self) -> torch.Tensor:
        """Return current joint efforts (torques)."""
        return torch.as_tensor(self._current_joint_efforts, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _robot_body_pose_w(self) -> torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        pose = torch.as_tensor(self._current_robot_body_pose_w, device=self.device, dtype=torch.float32).unsqueeze(0)
        pose[:, :, 3:7] = convert_quat(pose[:, :, 3:7], to="wxyz")
        return pose

    @property
    def _robot_root_pose_w(self) -> torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        pose = torch.as_tensor(self._current_robot_root_pose_w, device=self.device, dtype=torch.float32).unsqueeze(0)
        pose[:, 3:7] = convert_quat(pose[:, 3:7], to="wxyz")
        return pose

    @property
    def _jacobians(self) -> torch.Tensor:
        """Return the jacobian frame transforms of the robot."""
        return torch.as_tensor(self._current_jacobians, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _robot_dof_lower_limits(self) -> torch.Tensor:
        """Return the lower limits of the robot joints."""
        lower_lim = torch.as_tensor(self._current_lower_joint_limits, device=self.device, dtype=torch.float32)
        lower_lim[lower_lim == 0] = -2 * torch.pi
        return lower_lim

    @property
    def _robot_dof_upper_limits(self) -> torch.Tensor:
        """Return the upper limits of the robot joints."""
        upper_lim = torch.as_tensor(self._current_upper_joint_limits, device=self.device, dtype=torch.float32)
        upper_lim[upper_lim == 0] = 2 * torch.pi
        return upper_lim

    @property
    def _gravity_vector(self) -> torch.Tensor:
        """Return the gravity compenstation vector."""
        return torch.as_tensor(self._current_gravity_vector, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _mass_matrices(self) -> torch.Tensor:
        """Return the mass matrices."""
        return torch.as_tensor(self._current_mass_matrices, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _robot_body_vel_w(self) -> torch.Tensor:
        """Return body velocity in XYZ + Quaternion."""
        return torch.as_tensor(self._current_robot_body_vel_w, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def _joint_centers(self) -> torch.Tensor:
        """Return joint centers."""
        return torch.as_tensor(self._current_joint_centers, device=self.device, dtype=torch.float32).unsqueeze(0)

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which we publish to Kortex.
        """
        return self.cfg.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return float(self.cfg.dt * self.cfg.decimation)

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """The maximum episode length in steps adjusted from s."""
        return math.ceil(self.max_episode_length_s / (self.cfg.dt * self.cfg.decimation))

    def _find_link_idx(self, link: str) -> int:
        """Find the link index in the robot urdf.

        Args:
            link: string of link name

        Returns:
            int of index

        """
        return self._robot_links.index(link)

    def _find_joint_idx(self, joint: str) -> int:
        """Find the joint index in the robot urdf.

        Args:
            joint: string of joint name

        Returns:
            int of index

        """
        return self._robot_joints.index(joint)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations happen during reset() which are not repeated

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.


        Returns:
            A tuple containing the observations and extras.

        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx()

        self._episode_length_buf += 1
        self._common_step_counter += 1
        self._update_robot_info()
        self._next_step_time = time.monotonic()

        # return observations
        return self._get_observations(), self.extras

    def step(
        self, action: torch.Tensor, action_spec: ActionSpec[Any] | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:  # type: ignore
        """Execute one time-step of the on the robot through the Kortex API.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRlEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRlEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre process the action and store it
        2. Publish the action to the robot
        3. Compute the rewards
        4. Get the observations
        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).
            action_spec: the action specification

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.

        """
        # Perform safety check
        out = self._collision_checker.check_near_collision(
            self._current_joint_positions,
            self._current_joint_velocities,
            self._current_joint_efforts,
            self.cfg.arm_joint_names + self.cfg.gripper_joint_names,
        )
        if out.near_collision:
            print(
                f"[WARN][KORTEX] Near collision with [{out.limiting_pair[0]}, {out.limiting_pair[1]}]. Stopping robot.."
            )
            zero_action = self._pre_process_action(torch.zeros_like(action), action_spec=action_spec)
            self._publish_action_to_kortex(zero_action, duration=self.step_dt, action_spec=action_spec)
        elif out.effort_lim:
            print(f"[WARN][KORTEX] Joint effort limits reached {self._joint_efforts}. Stopping robot")
            zero_action = self._pre_process_action(torch.zeros_like(action), action_spec=action_spec)
        else:
            # Pre process the robot action
            action = self._pre_process_action(action, action_spec=action_spec)
            # Send the robot action to hardware
            self._publish_action_to_kortex(action, duration=self.step_dt, action_spec=action_spec)
        sleep_time = (time.perf_counter() - self._next_step_time) - self.step_dt

        if sleep_time < 0:
            time.sleep(min(-sleep_time, self.step_dt))
        else:
            # print(f"[WARN] full loop overran by {sleep_time * 1000:.1f}ms")
            pass
        self._next_step_time = time.perf_counter()

        self._episode_length_buf += 1
        self._common_step_counter += 1

        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        #  reset envs that terminated/timed-out and log the episode information
        if self.reset_buf:
            self._reset_idx()

        self._update_robot_info()  # Update all robot information
        # update observations
        obs = self._get_observations()
        for k, v in obs.items():
            obs[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
        self.obs_buf = obs
        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            torch.as_tensor(self.reward_buf, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.as_tensor([self.reset_terminated], dtype=torch.bool, device=self.device),
            torch.as_tensor([self.reset_time_outs], dtype=torch.bool, device=self.device),
            self.extras,
        )

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.

        """
        # set seed for torch and other libraries
        return configure_seed(seed)

    def render(self) -> None:
        """Run rendering by visualizing with RViz."""
        ...

    def close(self) -> None:
        """Cleanup for the environment."""
        self.kortex.Stop()

    """
    Helper functions.
    """

    def _reset_idx(self) -> None:
        """Reset environments based on specified indices."""
        # reset the episode length buffer
        self._episode_length_buf = 0

    """
    Implementation-specific functions.
    """

    @abstractmethod
    def _pre_process_action(self, actions: torch.Tensor, action_spec: ActionSpec[Any] | None = None) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).
            action_spec: The action specification telling the environment how to process the action

        Returns:
            The actions to publish to the robot

        """
        raise NotImplementedError(f"Please implement the '_pre_process_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _publish_action_to_kortex(
        self, actions: np.ndarray, action_spec: ActionSpec[Any] | None = None, duration: float = 1
    ) -> None:
        """Publish action to robot controller via the Kortex API.

        This function is responsible publishing actions to the correct robot controller. These positions are set by
        _pre_process_action()

        Args:
            actions: joint positions to publish to the robot
            action_spec: The actions to publish to the Kortex API
            duration: duration that the action expects, useful for some action specs

        """
        raise NotImplementedError(
            f"Please implement the '_publish_action_to_kortex' method for {self.__class__.__name__}."
        )

    @abstractmethod
    def _get_observations(self) -> dict[str, np.ndarray]:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment in the form {positions: [], velocities: []}.

        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self) -> np.ndarray:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).

        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[bool, bool]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Assumed to not be batched

        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")

    """
    Helper functions to communicate with Kortex
    """

    def switch_controllers(self, active_controller: int) -> bool:
        """Switch Kortex controllers.

        Args:
            active_controller: Controller (Enum) to activate

        Returns:
            Bool if the controller switch was successful

        """
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = active_controller
        self.kortex.SetServoingMode(base_servo_mode)

    def _update_robot_info(self) -> None:
        """Obtain the required robot information from the Kortex API."""
        feedback = self.kortex_c.RefreshFeedback()

        def deg_to_rad_wrapped(deg: np.ndarray) -> np.ndarray:
            """Shift angles > 180 to negative and convert to radians."""
            deg_wrapped = (deg + 180) % 360 - 180  # [0,360] -> [-180, 180]
            return np.deg2rad(deg_wrapped)

        # Compute the current joint positions, velocities, and efforts
        curr_arm_positions = deg_to_rad_wrapped(np.asarray([actuator.position for actuator in feedback.actuators]))
        curr_gripper_positions = np.deg2rad([feedback.interconnect.gripper_feedback.motor[0].position])
        self._current_joint_positions = np.concatenate((curr_arm_positions, curr_gripper_positions), axis=0)

        curr_arm_velocities = np.deg2rad([actuator.velocity for actuator in feedback.actuators])
        curr_gripper_velocities = np.deg2rad([feedback.interconnect.gripper_feedback.motor[0].velocity])
        self._current_joint_velocities = np.concatenate((curr_arm_velocities, curr_gripper_velocities), axis=0)

        curr_arm_efforts = np.asarray([actuator.torque for actuator in feedback.actuators])
        curr_gripper_efforts = np.asarray([feedback.interconnect.gripper_feedback.motor[0].current_motor])
        self._current_joint_efforts = np.concatenate((curr_arm_efforts, curr_gripper_efforts), axis=0)

        self._robot_links = [f.name for f in self.kortex_model.frames if f.type == pin.FrameType.BODY]
        self._robot_joints = [self.kortex_model.names[i] for i in range(1, self.kortex_model.njoints)]

        self._current_lower_joint_limits = np.asarray(self.kortex_model.lowerPositionLimit, dtype=float)
        self._current_upper_joint_limits = np.asarray(self.kortex_model.upperPositionLimit, dtype=float)
        self._current_joint_centers = (self._current_upper_joint_limits + self._current_lower_joint_limits) / 2

        jacobians = []
        poses = []
        vels = []
        # Pad gripper positions and velocities
        q = pin.neutral(self.kortex_model)
        q[: len(curr_arm_positions)] = curr_arm_positions
        dq = pin.neutral(self.kortex_model)
        dq[: len(curr_arm_velocities)] = curr_arm_velocities
        pin.computeJointJacobians(self.kortex_model, self.kortex_data, q)
        pin.forwardKinematics(self.kortex_model, self.kortex_data, q, dq)
        pin.updateFramePlacements(self.kortex_model, self.kortex_data)
        for f in self.kortex_model.frames:
            if f.type != pin.FrameType.BODY:
                continue
            frame_id = self.kortex_model.getFrameId(f.name)

            poses.append(pin.SE3ToXYZQUAT(self.kortex_data.oMf[frame_id].copy()))
            jacobians.append(
                pin.getFrameJacobian(
                    self.kortex_model, self.kortex_data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
            )
            vels.append(
                pin.getFrameVelocity(
                    self.kortex_model, self.kortex_data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
            )
        self._current_jacobians = np.asarray(jacobians)
        self._current_robot_body_pose_w = np.asarray(poses)
        self._current_robot_root_pose_w = np.asarray(poses[0])
        self._current_robot_body_vel_w = np.asarray(vels)

        self._current_mass_matrices = pin.crba(self.kortex_model, self.kortex_data, q)
        self._current_gravity_vector = pin.computeGeneralizedGravity(self.kortex_model, self.kortex_data, q)
