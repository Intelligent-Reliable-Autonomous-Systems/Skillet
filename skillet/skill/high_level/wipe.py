"""A wipe skill for wiping a bounding box in a boustrophedon (lawnmower) pattern."""

from enum import IntEnum
from typing import Generic

import torch
from jaxtyping import Float, Int

from skillet.core.math import quat_error_magnitude, quat_from_yaw, quat_mul
from skillet.core.policy import BatchedPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike, SkillParamsSpec
from skillet.envs.specs import IKEE_Obs
from skillet.skill.specs import XYZ_Yaw_XYZ_Params, XYZ_Yaw_XYZ_Params_Spec


class WipeStatusCodes(IntEnum):
    """The codes for the status of a skill.

    Note: WIPE is the *first* of a dynamically-sized run of lawnmower-pattern
    wipe stages. The number of wipe stages depends on ``num_wipes`` (set at
    construction time), so the indices for RAISE and DONE are no longer fixed
    enum values -- they're computed per-skill in ``initiate()`` and stored as
    ``self._raise_status`` / ``self._done_status``.
    """

    IDLE = 0
    """The skill is idle."""
    HOVER = 1
    """The skill is hovering above location"""
    LOWER = 2
    """The skill is lowering to the object."""
    WIPE = 3
    """The first stage of the boustrophedon (lawnmower) wipe pattern. There
    are ``2 * num_wipes - 1`` of these: each "row" is one side-to-side sweep
    across the bounding box, and rows are connected by a "transition" stage
    that steps to the next row without moving along the sweep axis. Together
    they iteratively cover the bounding box from the start corner to the end
    corner, occupying stage indices ``WIPE ... WIPE + 2*num_wipes - 2``.
    """


class WipeSkill(BatchedSkill[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params], Generic[TBAction]):
    """A wipe skill that iteratively covers a bounding box in a boustrophedon (lawnmower) pattern.

    Generic Args:
        TBAction: The type of the action for the skill.

    Parameterized by [x,y,z, yaw, xyz]: the xyz of the "start" corner of the bounding box, the
    gripper orientation (yaw), and the xyz of the "end" corner (e.g. start = top-right, end =
    bottom-left). Rather than a single linear pass or oscillating between just the two corners,
    the skill sweeps side-to-side along ``sweep_axis`` and, row by row, steps along ``row_axis``
    from the start corner toward the end corner -- so ``num_wipes`` rows collectively cover the
    full span of the bounding box instead of retracing the same line.
    """

    def __init__(
        self,
        reach_policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params],
        lift_height: float,
        gripper_close: float,
        length: int,
        num_wipes: int = 4,
        sweep_axis: int = 0,
    ) -> None:
        """Initialize the wipe skill.

        Generic Args:
            TBAction: The type of the action for the skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            lift_height: The height to lift the object to.
            gripper_close: How closed the gripper should be
            length: The number of steps to execute the skill for.
            num_wipes: The number of side-to-side sweep rows used to
                iteratively cover the bounding box while stepping from the
                start corner to the end corner (boustrophedon/lawnmower
                pattern). ``num_wipes=1`` reduces to a single straight pass
                from the start corner to the end corner.
            sweep_axis: Which axis (0=x, 1=y) is swept side-to-side within
                each row. The other of {x, y} is treated as the "row" axis
                that steps from the start corner toward the end corner
                between rows. z is always interpolated by row progress, so
                a bounding box that slopes in z is handled smoothly.

        """
        self._name = "wipe_skill"
        self._reach_policy = reach_policy
        self._lift_height = lift_height
        self._gripper_close = gripper_close
        self._length = length
        self._num_wipes = num_wipes
        if sweep_axis not in (0, 1):
            raise ValueError(f"sweep_axis must be 0 (x) or 1 (y), got {sweep_axis}")
        self._sweep_axis = sweep_axis
        self._row_axis = 1 - sweep_axis
        self._status = None
        self._wipe_status = None
        self._params = None

        # 180 degree rotation about X axis + -90 degree yaw
        self._default_quat = torch.as_tensor([[0.0, 1, 0, 0.0]])

    @property
    def param_dim(self) -> int:
        return 7

    @property
    def params_spec(self) -> SkillParamsSpec[XYZ_Yaw_XYZ_Params]:
        return XYZ_Yaw_XYZ_Params_Spec

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._reach_policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:
        """Initiate the wipe skill.

        Args:
            obs: The low-level observation for the skill.
            params: The wipe parameters, (x, y, z, yaw, x y z) as shape (b,7)
                    for the bounding box

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._wipe_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._wipe_status[:] = WipeStatusCodes.HOVER
        self._params = params
        self._n_steps = 0
        self._default_quat = self._default_quat.to(self.obs_spec.device)
        goal_quat = quat_mul(quat_from_yaw(params[:, 3]), self._default_quat.repeat(self.n_envs, 1))

        self._pos_threshold = 0.005
        self._quat_threshold = 0.05
        self._vel_threshold = 0.001
        self._joint_threshold = 0.001

        ee_pose_b = obs["tcp_pose_b"]

        # Number of rows in the lawnmower pattern. Row 0 needs only a sweep stage
        # (since LOWER already places us at its start); each subsequent row needs a
        # "transition" stage (step to the next row, no sweep-axis movement) followed
        # by its own sweep stage. So total wipe stages = 2 * num_rows - 1.
        num_rows = max(1, self._num_wipes)
        num_segments = max(1, 2 * num_rows - 1)

        # Dynamically computed stage indices (since num_segments depends on num_wipes)
        wipe_start = WipeStatusCodes.WIPE
        self._raise_status = wipe_start + num_segments
        self._done_status = self._raise_status + 1

        # Define the target poses for each stage of the wipe skill, indexed by stage number:
        # 0: IDLE (unused), 1: HOVER, 2: LOWER, 3..3+num_segments-1: lawnmower rows/transitions, then RAISE.
        # (n_envs, num_stages, 7)
        num_stages = self._done_status + 1
        target_poses = spec.zeros(shape=(self.n_envs, num_stages, 7), dtype=float)

        # HOVER[1]: Go over to the start x,y position, oriented downward (gripper open)
        target_poses[:, WipeStatusCodes.HOVER, :2] = params[:, :2]  # (x,y) from params (start corner)
        target_poses[:, WipeStatusCodes.HOVER, 2] = self._lift_height
        target_poses[:, WipeStatusCodes.HOVER, 3:7] = goal_quat
        # LOWER[2]: Go down to the start z position (gripper open)
        target_poses[:, WipeStatusCodes.LOWER, :7] = target_poses[:, WipeStatusCodes.HOVER, :7]
        target_poses[:, WipeStatusCodes.LOWER, 2] = params[:, 2]

        # WIPE[3 .. 3+num_segments-1]: boustrophedon (lawnmower) coverage of the bounding box.
        # We sweep back and forth along `sweep_axis` while stepping row-by-row along `row_axis`
        # from the start corner to the end corner, so the pattern progressively covers the whole
        # box instead of retracing the same start<->end line.
        start_xyz = params[:, 0:3]  # (n_envs, 3) start corner ("top right", say)
        end_xyz = params[:, 4:7]  # (n_envs, 3) end corner ("bottom left", say)
        sweep_axis = self._sweep_axis
        row_axis = self._row_axis

        stage_idx = wipe_start
        for i in range(num_rows):
            row_frac = i / (num_rows - 1) if num_rows > 1 else 0.0
            row_val = start_xyz[:, row_axis] + row_frac * (end_xyz[:, row_axis] - start_xyz[:, row_axis])
            z_val = start_xyz[:, 2] + row_frac * (end_xyz[:, 2] - start_xyz[:, 2])

            # Even rows sweep start->end along sweep_axis; odd rows sweep end->start (zigzag).
            if i % 2 == 0:
                sweep_from, sweep_to = start_xyz[:, sweep_axis], end_xyz[:, sweep_axis]
            else:
                sweep_from, sweep_to = end_xyz[:, sweep_axis], start_xyz[:, sweep_axis]

            if i > 0:
                # Transition stage: step to this row (row_axis + z change only), staying at
                # the sweep-axis position the previous row's sweep ended on (== sweep_from).
                target_poses[:, stage_idx, sweep_axis] = sweep_from
                target_poses[:, stage_idx, row_axis] = row_val
                target_poses[:, stage_idx, 2] = z_val
                target_poses[:, stage_idx, 3:7] = goal_quat
                stage_idx += 1

            # Sweep stage: move across the row to the opposite sweep-axis edge.
            target_poses[:, stage_idx, sweep_axis] = sweep_to
            target_poses[:, stage_idx, row_axis] = row_val
            target_poses[:, stage_idx, 2] = z_val
            target_poses[:, stage_idx, 3:7] = goal_quat
            stage_idx += 1

        # RAISE: lift straight up from wherever the last wipe segment ended
        last_wipe_idx = stage_idx - 1
        target_poses[:, self._raise_status, :7] = target_poses[:, last_wipe_idx, :7]
        target_poses[:, self._raise_status, 2] = self._lift_height
        self._target_poses = target_poses

        # Start the skill by going to the HOVER pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._wipe_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        ee_pose_b = obs["tcp_pose_b"]

        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._current_target_poses[:, 0:3], dim=1)
            < self._pos_threshold
        )

        reached_quat = (
            quat_error_magnitude(ee_pose_b[:, 3:7], self._current_target_poses[:, 3:7]) < self._quat_threshold
        )
        reached_pose = reached_pos & reached_quat
        next_pose = reached_pose

        if next_pose.any():
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._wipe_status[valid_idx] += 1
            valid_idx = valid_idx & (self._wipe_status < self._done_status)
            # print(
            #     f"[INFO][WIPE STATUS UPDATE]: {self._wipe_status.cpu().numpy()[0]} | reached_pose: {reached_pose.cpu().numpy()}"
            # )
            # Update the target pose based on the new wipe status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._wipe_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            (self._wipe_status >= WipeStatusCodes.LOWER),
            torch.ones_like(reach_actions[:, -1]) * self._gripper_close,  # Close gripper
            torch.ones_like(reach_actions[:, -1]) * 0.67,  # Open gripper
        )
        # reach_actions[:, -1] = torch.ones_like(reach_actions[:, -1]) * self._gripper_close

        self._n_steps += 1
        self._status[self._wipe_status == self._done_status] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...


class WipeSkillOld2(BatchedSkill[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params], Generic[TBAction]):
    """A wipe skill for wiping back-and-forth across a table.

    Generic Args:
        TBAction: The type of the action for the skill.

    Parameterized by [x,y,z, yaw, xyz] the x y z location to perform the wipe action, orientation (yaw)
    of the gripper, and the new xyz position to wipe to. The skill wipes back and forth between these
    two corners ``num_wipes`` times before raising.
    """

    def __init__(
        self,
        reach_policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params],
        lift_height: float,
        gripper_close: float,
        length: int,
        num_wipes: int = 2,
    ) -> None:
        """Initialize the wipe skill.

        Generic Args:
            TBAction: The type of the action for the skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            lift_height: The height to lift the object to.
            gripper_close: How closed the gripper should be
            length: The number of steps to execute the skill for.
            num_wipes: The number of back-and-forth (there-and-back) passes
                to perform across the bounding box span before raising.
                Each wipe consists of two segments: start -> end, end -> start.

        """
        self._name = "wipe_skill"
        self._reach_policy = reach_policy
        self._lift_height = lift_height
        self._gripper_close = gripper_close
        self._length = length
        self._num_wipes = num_wipes
        self._status = None
        self._wipe_status = None
        self._params = None

        # 180 degree rotation about X axis + -90 degree yaw
        self._default_quat = torch.as_tensor([[0.0, 1, 0, 0.0]])

    @property
    def param_dim(self) -> int:
        return 7

    @property
    def params_spec(self) -> SkillParamsSpec[XYZ_Yaw_XYZ_Params]:
        return XYZ_Yaw_XYZ_Params_Spec

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._reach_policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:
        """Initiate the wipe skill.

        Args:
            obs: The low-level observation for the skill.
            params: The wipe parameters, (x, y, z, yaw, x y z) as shape (b,7)
                    for the bounding box

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._wipe_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._wipe_status[:] = WipeStatusCodes.HOVER
        self._params = params
        self._n_steps = 0
        self._default_quat = self._default_quat.to(self.obs_spec.device)
        goal_quat = quat_mul(quat_from_yaw(params[:, 3]), self._default_quat.repeat(self.n_envs, 1))

        self._pos_threshold = 0.005
        self._quat_threshold = 0.05
        self._vel_threshold = 0.001
        self._joint_threshold = 0.001

        ee_pose_b = obs["tcp_pose_b"]

        # Number of one-way segments in the back-and-forth wipe: each "wipe"
        # is a there-and-back pair (start->end, end->start).
        num_segments = max(1, 2 * self._num_wipes)

        # Dynamically computed stage indices (since num_segments depends on num_wipes)
        wipe_start = WipeStatusCodes.WIPE
        self._raise_status = wipe_start + num_segments
        self._done_status = self._raise_status + 1

        # Define the target poses for each stage of the wipe skill, indexed by stage number:
        # 0: IDLE (unused), 1: HOVER, 2: LOWER, 3..3+num_segments-1: WIPE passes, then RAISE.
        # (n_envs, num_stages, 7)
        num_stages = self._done_status + 1
        target_poses = spec.zeros(shape=(self.n_envs, num_stages, 7), dtype=float)

        # HOVER[1]: Go over to the start x,y position, oriented downward (gripper open)
        target_poses[:, WipeStatusCodes.HOVER, :2] = params[:, :2]  # (x,y) from params (start corner)
        target_poses[:, WipeStatusCodes.HOVER, 2] = self._lift_height
        target_poses[:, WipeStatusCodes.HOVER, 3:7] = goal_quat
        # LOWER[2]: Go down to the start z position (gripper open)
        target_poses[:, WipeStatusCodes.LOWER, :7] = target_poses[:, WipeStatusCodes.HOVER, :7]
        target_poses[:, WipeStatusCodes.LOWER, 2] = params[:, 2]

        # WIPE[3 .. 3+num_segments-1]: alternate between the end corner and the start corner,
        # producing a back-and-forth motion across the bounding box span.
        start_pose = target_poses[:, WipeStatusCodes.LOWER, :7].clone()
        start_pose[:, 0:3] = params[:, 0:3]
        end_pose = start_pose.clone()
        end_pose[:, 0:3] = params[:, 4:7]

        for i in range(num_segments):
            stage_idx = wipe_start + i
            # Even segments (0, 2, 4, ...) head to the end corner;
            # odd segments (1, 3, 5, ...) head back to the start corner.
            target_poses[:, stage_idx, :7] = end_pose if (i % 2 == 0) else start_pose

        # RAISE: lift straight up from wherever the last wipe segment ended
        last_wipe_idx = wipe_start + num_segments - 1
        target_poses[:, self._raise_status, :7] = target_poses[:, last_wipe_idx, :7]
        target_poses[:, self._raise_status, 2] = self._lift_height
        self._target_poses = target_poses

        # Start the skill by going to the HOVER pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._wipe_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        ee_pose_b = obs["tcp_pose_b"]

        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._current_target_poses[:, 0:3], dim=1)
            < self._pos_threshold
        )

        reached_quat = (
            quat_error_magnitude(ee_pose_b[:, 3:7], self._current_target_poses[:, 3:7]) < self._quat_threshold
        )
        reached_pose = reached_pos & reached_quat
        next_pose = reached_pose

        if next_pose.any():
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._wipe_status[valid_idx] += 1
            valid_idx = valid_idx & (self._wipe_status < self._done_status)
            # print(
            #     f"[INFO][WIPE STATUS UPDATE]: {self._wipe_status.cpu().numpy()[0]} | reached_pose: {reached_pose.cpu().numpy()}"
            # )
            # Update the target pose based on the new wipe status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._wipe_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            (self._wipe_status >= WipeStatusCodes.LOWER),
            torch.ones_like(reach_actions[:, -1]) * self._gripper_close,  # Close gripper
            torch.ones_like(reach_actions[:, -1]) * 0.6,  # Open gripper
        )
        # reach_actions[:, -1] = torch.ones_like(reach_actions[:, -1]) * self._gripper_close

        self._n_steps += 1
        self._status[self._wipe_status == self._done_status] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...


class WipeSkillOld(BatchedSkill[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params], Generic[TBAction]):
    """A wipe skill for wiping an object across a table.

    Generic Args:
        TBAction: The type of the action for the skill.

    Parameterized by [x,y,z, yaw, xyz] the x y z location to perform the wipe action, orientation (yaw)
    of the gripper, and the new xyz position to wipe to
    """

    def __init__(
        self,
        reach_policy: BatchedPolicy[IKEE_Obs, TBAction, XYZ_Yaw_XYZ_Params],
        lift_height: float,
        gripper_close: float,
        length: int,
    ) -> None:
        """Initialize the wipe skill.

        Generic Args:
            TBAction: The type of the action for the skill.

        Args:
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            lift_height: The height to lift the object to.
            gripper_close: How closed the gripper should be
            length: The number of steps to execute the skill for.

        """
        self._name = "wipe_skill"
        self._reach_policy = reach_policy
        self._lift_height = lift_height
        self._gripper_close = gripper_close
        self._length = length
        self._status = None
        self._wipe_status = None
        self._params = None

        # 180 degree rotation about X axis + -90 degree yaw
        self._default_quat = torch.as_tensor([[0.0, 1, 0, 0.0]])

    @property
    def param_dim(self) -> int:
        return 7

    @property
    def params_spec(self) -> SkillParamsSpec[XYZ_Yaw_XYZ_Params]:
        return XYZ_Yaw_XYZ_Params_Spec

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._reach_policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:
        """Initiate the wipe skill.

        Args:
            obs: The low-level observation for the skill.
            params: The wipe parameters, (x, y, z, yaw, x y z) as shape (b,7)
                    for the bounding box

        """
        self.n_envs = self.obs_spec.n_envs_from(obs)
        spec = self.policy.obs_spec.with_n_envs(self.n_envs)
        self._status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._wipe_status = spec.zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._wipe_status[:] = WipeStatusCodes.HOVER
        self._params = params
        self._n_steps = 0
        self._default_quat = self._default_quat.to(self.obs_spec.device)
        goal_quat = quat_mul(quat_from_yaw(params[:, 3]), self._default_quat.repeat(self.n_envs, 1))

        self._pos_threshold = 0.005
        self._quat_threshold = 0.05
        self._vel_threshold = 0.001
        self._joint_threshold = 0.001

        ee_pose_b = obs["tcp_pose_b"]

        # Define the target poses for each stage of the wipe skill, indexed by WipeStatusCodes
        # (n_envs, num_wipe_stages, 7)
        target_poses = spec.zeros(shape=(self.n_envs, 6, 7), dtype=float)

        # HOVER[1]: Go over to the target x,y position, oriented downward (gripper open)
        target_poses[:, WipeStatusCodes.HOVER, :2] = params[:, :2]  # (x,y) from params
        target_poses[:, WipeStatusCodes.HOVER, 2] = self._lift_height
        target_poses[:, WipeStatusCodes.HOVER, 3:7] = goal_quat
        # LOWER[2]: Go down to the target z position (gripper open)
        target_poses[:, WipeStatusCodes.LOWER, :7] = target_poses[:, WipeStatusCodes.HOVER, :7]
        target_poses[:, WipeStatusCodes.LOWER, 2] = params[:, 2]

        # WIPE[3]: Wipe the object to the target location
        target_poses[:, WipeStatusCodes.WIPE, 3:7] = target_poses[:, WipeStatusCodes.LOWER, 3:7]
        target_poses[:, WipeStatusCodes.WIPE, 0:3] = params[:, 4:7]

        # RAISE[4]: Wipe the object to the target location
        target_poses[:, WipeStatusCodes.RAISE, :7] = target_poses[:, WipeStatusCodes.WIPE, :7]
        target_poses[:, WipeStatusCodes.RAISE, 2] = self._lift_height
        self._target_poses = target_poses

        # Start the skill by going to the ASCEND pose
        idx = torch.arange(self.n_envs, device=target_poses.device)
        valid_idx = self._status == SkillStatusCodes.RUNNING
        self._current_target_poses = target_poses[idx, self._wipe_status]
        env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
        if env_ids.numel():
            self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        ee_pose_b = obs["tcp_pose_b"]

        reached_pos = (
            torch.linalg.vector_norm(ee_pose_b[:, 0:3] - self._current_target_poses[:, 0:3], dim=1)
            < self._pos_threshold
        )

        reached_quat = (
            quat_error_magnitude(ee_pose_b[:, 3:7], self._current_target_poses[:, 3:7]) < self._quat_threshold
        )
        reached_pose = reached_pos & reached_quat
        next_pose = reached_pose

        if next_pose.any():
            idx = torch.arange(self.n_envs, device=reached_pose.device)
            valid_idx = (self._status == SkillStatusCodes.RUNNING) & (reached_pose)
            self._wipe_status[valid_idx] += 1
            valid_idx = valid_idx & (self._wipe_status < WipeStatusCodes.DONE)
            # print(
            #     f"[INFO][WIPE STATUS UPDATE]: {WipeStatusCodes(self._wipe_status.cpu().numpy()[0]).name} | reached_pose: {reached_pose.cpu().numpy()}"
            # )
            # Update the target pose based on the new wipe status
            self._current_target_poses[valid_idx] = self._target_poses[idx[valid_idx], self._wipe_status[valid_idx]]

            env_ids = torch.nonzero(valid_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self._reach_policy.reset(obs, self._current_target_poses, env_ids=env_ids)

        reach_actions = self._reach_policy.get_action(obs)
        reach_actions[:, -1] = torch.where(
            (self._wipe_status >= WipeStatusCodes.LOWER),
            torch.ones_like(reach_actions[:, -1]) * self._gripper_close,  # Close gripper
            torch.ones_like(reach_actions[:, -1]) * 0.6,  # Open gripper
        )
        # reach_actions[:, -1] = torch.ones_like(reach_actions[:, -1]) * self._gripper_close

        self._n_steps += 1
        self._status[self._wipe_status == WipeStatusCodes.DONE] = SkillStatusCodes.SUCCESS
        if self._n_steps >= self._length:
            self._status[self._status == SkillStatusCodes.RUNNING] = SkillStatusCodes.FAILED

        return reach_actions

    def reward(self, obs: TBSkillObs) -> Float[ArrayLike, "b"]:  # noqa: F821
        """Compute the reward of the skill."""
        ...
