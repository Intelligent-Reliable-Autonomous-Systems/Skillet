from collections.abc import Callable, Sequence
from typing import TypeAlias, override

import gymnasium as gym
import torch

from skillet.core import SkillParamsSpec
from skillet.core.skill import SingleSkill, SkillStatus, SkillStatusCodes
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action
from skillet.planning.abstract.skill_grounding import _drag_skill_5_grounding
from skillet.scene import Location
from skillet.scene.base import Scene, SceneObject
from skillet.skill.high_level.drag import DragSkill

Object_Params: TypeAlias = int
"""The parameters for selecting an object in the scene."""


class DragBlockSkill(SingleSkill[IKEE_Obs, M_Action, Object_Params]):
    """A skill for draging a block one grip space north.

    Is is discretely parameterized by the id of the block to drag.
    """

    def __init__(
        self,
        scene: Scene,
        drag_skill: DragSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
        xyz_offset: tuple[int] = (0, 0.0, 0.02),
    ) -> None:
        """Initialize the drag block skill."""
        self._scene = scene
        self._drag_skill = drag_skill
        self.max_objects = len(scene.objects) if scene.closed_set else 100

        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.Discrete(n=self.max_objects), name="block_id", is_torch=False, is_batched=False
        )
        self._status = None
        self._offset = torch.tensor(xyz_offset, device=self.obs_spec.device)
        self._drag_xyz = torch.tensor([0.05, 0.0, 0.0], device=self.obs_spec.device)

        self._vis_target_pos = vis_target_pos

    @property
    def name(self) -> str:
        """The name of the skill."""
        return "drag_block"

    @property
    @override
    def policy(self):
        return self._drag_skill.policy

    @property
    def obs_spec(self):
        """The specification of the observation space for the skill."""
        return self._drag_skill.obs_spec

    @property
    def action_spec(self):
        """The specification of the action space for the skill."""
        return self._drag_skill.action_spec

    @property
    def params_spec(self):
        """The specification of the parameters space for the skill."""
        return self._block_params_spec

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        params = self.params_spec.cast(params)

        self._target_block: SceneObject = self._scene.objects(params)
        if not self._target_block.is_pose_known():
            self._status = torch.as_tensor(SkillStatusCodes.FAILED, device=self.params_spec.device)
            return
        target_xyz = self._target_block.pose[:3] + self._offset
        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        drag_loc = target_xyz + self._drag_xyz
        target_pose = torch.tensor(
            [target_xyz[0], target_xyz[1], target_xyz[2], yaw, drag_loc[0], drag_loc[1], drag_loc[2]]
        )
        target_pose = self._drag_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._drag_skill.initiate(obs, target_pose)

    @override
    def get_action(self, obs: IKEE_Obs) -> M_Action:
        obs = self._drag_skill.obs_spec.cast(obs)
        actions = self._drag_skill.get_action(obs)
        return self.action_spec.cast(actions)

    @property
    def status(self) -> SkillStatus:
        """The status of the skills."""
        if self._status is not None:
            return self._status
        return self._drag_skill.status[0]

    @status.setter
    def status(self, st: SkillStatus) -> None:
        self._status = torch.as_tensor(st, device=self.params_spec.device)


class DragBlock2Skill(DragBlockSkill):
    def __init__(
        self,
        scene: Scene,
        drag_skill: DragSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
        xyz_offset: tuple[int] = (0, 0.0, 0.02),
    ) -> None:
        """Initialize the drag block skill."""
        super().__init__(scene, drag_skill, vis_target_pos, xyz_offset=xyz_offset)
        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.MultiDiscrete((self.max_objects,) * 2), name="block_id", is_torch=False, is_batched=False
        )
        self._params = None

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        self._params = self.params_spec.cast(params[:2])

        self._target_block: SceneObject = self._scene.get_objects_from_id(self._params)[0]
        if not self._target_block.is_pose_known() or not self._target_block.moveable:
            self._status = torch.as_tensor(SkillStatusCodes.FAILED, device=self.params_spec.device)
            print(
                f"[INFO][DRAG BLOCK][FAILED]: {self._target_block.name} | {self._scene.get_objects_from_id(self._params)[1].name}"
            )

            return
        target_xyz = self._target_block.pose[:3].to(self.obs_spec.device) + self._offset
        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        drag_loc = target_xyz + self._drag_xyz
        target_pose = torch.tensor(
            [target_xyz[0], target_xyz[1], target_xyz[2], yaw, drag_loc[0], drag_loc[1], drag_loc[2]]
        )
        target_pose = self._drag_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._drag_skill.initiate(obs, target_pose)
        print(
            f"[INFO][DRAG BLOCK]: {self._target_block.name} | {self._scene.get_objects_from_id(self._params)[1].name}"
        )

    def __str__(self) -> str:
        if self._params is not None:
            names = self._scene.resolve_ids_to_names(self._params)
            return f"Drag Block: | {names[0]} | {names[1]} |"
        return "Drag Block: | Unset | Unset |"


class DragBlock5Skill(DragBlockSkill):
    def __init__(
        self,
        scene: Scene,
        drag_skill: DragSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
        xyz_offset: tuple[int] = (0, 0.0, 0.02),
    ) -> None:
        """Initialize the drag block skill."""
        super().__init__(scene, drag_skill, vis_target_pos, xyz_offset=xyz_offset)
        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.MultiDiscrete((self.max_objects,) * 5), name="block_id", is_torch=False, is_batched=False
        )
        self._params = None

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        self._params = self.params_spec.cast(params[:5])

        objs = self._scene.get_objects_from_id(self._params)
        self._target_block: SceneObject = objs[0]
        if (
            not self._target_block.is_pose_known()
            or not self._target_block.moveable
            or _drag_skill_5_grounding(objs, self._scene)
        ):
            self._status = torch.as_tensor(SkillStatusCodes.FAILED, device=self.params_spec.device)
            print(
                f"[INFO][DRAG BLOCK][FAILED]: {self._target_block.name} | {objs[1].name} | {objs[2].name} | {objs[3].name}"
            )
            return
        target_xyz = self._target_block.pose[:3].to(self.obs_spec.device) + self._offset
        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        if isinstance(objs[2], Location):
            drag_loc = target_xyz.clone()
            drag_loc[0] = objs[2].pose[0].clone() + 0.025
        else:
            drag_loc = target_xyz + self._drag_xyz
        target_pose = torch.tensor(
            [target_xyz[0], target_xyz[1], target_xyz[2], yaw, drag_loc[0], drag_loc[1], drag_loc[2]]
        )
        target_pose = self._drag_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._drag_skill.initiate(obs, target_pose)
        print(f"[INFO][DRAG BLOCK]: {self._target_block.name} | {objs[1].name} | {objs[2].name}")

    def __str__(self) -> str:
        if self._params is not None:
            names = self._scene.resolve_ids_to_names(self._params)
            return f"Drag Block: | {names[0]} | {names[1]} |"
        return "Drag Block: | Unset | Unset |"
