from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import gymnasium as gym
import numpy as np
import torch
from typing_extensions import override

from skillet.core import SkillParamsSpec
from skillet.core.skill import SingleSkill, SkillStatus, SkillStatusCodes
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action
from skillet.planning.abstract.spatial_grounding import _is_on
from skillet.scene import Cube, Table, Location
from skillet.scene.base import Scene, SceneObject
from skillet.scene.utils import find_valid_table_xy
from skillet.skill.high_level.place import PlaceSkill

Object_Params: TypeAlias = int
"""The parameters for selecting an object in the scene."""


class PlaceBlockSkill(SingleSkill[IKEE_Obs, M_Action, Object_Params]):
    """A skill for placing a block on another block.

    Is is discretely parameterized by the id of the block to place.
    """

    def __init__(
        self,
        scene: Scene,
        place_skill: PlaceSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
    ) -> None:
        """Initialize the place block skill."""
        self._scene = scene
        self._place_skill = place_skill
        self.max_objects = len(scene.objects) if scene.closed_set else 100
        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.Discrete(n=self.max_objects), name="block_id", is_torch=False, is_batched=False
        )

        self._status = None
        self._offset = torch.tensor([0, 0.0, 0.065], device=self.obs_spec.device)

        self._vis_target_pos = vis_target_pos

    @property
    def name(self) -> str:
        """The name of the skill."""
        return "place_block"

    @property
    @override
    def policy(self):
        return self._place_skill.policy

    @property
    def obs_spec(self):
        """The specification of the observation space for the skill."""
        return self._place_skill.obs_spec

    @property
    def action_spec(self):
        """The specification of the action space for the skill."""
        return self._place_skill.action_spec

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
        target_pose = torch.tensor([target_xyz[0], target_xyz[1], target_xyz[2], yaw])
        target_pose = self._place_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._place_skill.initiate(obs, target_pose)

    @override
    def get_action(self, obs: IKEE_Obs) -> M_Action:
        obs = self._place_skill.obs_spec.cast(obs)
        actions = self._place_skill.get_action(obs)
        return self.action_spec.cast(actions)

    @property
    def status(self) -> SkillStatus:
        """The status of the skills."""
        if self._status is not None:
            return self._status
        return self._place_skill.status[0]

    @status.setter
    def status(self, st: SkillStatus) -> None:
        self._status = torch.as_tensor(st, device=self.params_spec.device)


class PlaceBlock2Skill(PlaceBlockSkill):
    def __init__(
        self,
        scene: Scene,
        place_skill: PlaceSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
    ) -> None:
        """Initialize the place block skill."""
        super().__init__(scene, place_skill, vis_target_pos)
        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.MultiDiscrete((self.max_objects,) * 2), name="block_id", is_torch=False, is_batched=False
        )
        self._params = None

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        self._params = self.params_spec.cast(params[:2])

        objs = self._scene.get_objects_from_id(self._params)
        self._target = objs[1]
        if isinstance(self._target, Cube):
            if not self._target.is_pose_known() or self._params[0] == self._params[1]:
                self._status = torch.as_tensor(SkillStatusCodes.FAILED, device=self.params_spec.device)

                return
            target_xyz = self._target.pose[:3].to(self.obs_spec.device) + self._offset
        elif isinstance(self._target, Table):
            target_xyz = find_valid_table_xy(self._scene).to(self.obs_spec.device) + (self._offset / 2)
        else:
            raise ValueError(f"Unknown place object: {self._target}.")

        # Check for blocks under the grasped block
        target_xyz = target_xyz + self._resolve_offset(objs[0]).to(target_xyz.device)

        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        target_pose = torch.tensor([target_xyz[0], target_xyz[1], target_xyz[2], yaw])
        target_pose = self._place_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._place_skill.initiate(obs, target_pose)
        print(f"[INFO][PLACE BLOCK]: {objs[0].name} | {self._target.name}")

    def __str__(self) -> str:
        if self._params is not None:
            names = self._scene.resolve_ids_to_names(self._params)
            return f"Place Block: | {names[0]} | {names[1]} |"
        return "Place Block: | Unset | Unset |"

    def _resolve_offset(self, grasped_block: Cube, cube_size: float = 0.044) -> torch.Tensor:
        """Resolve the offset due to any blocks that might be below the grasped block."""
        offset = torch.as_tensor([0.0, 0.0, 0.0])
        for obj in self._scene.objects:
            if isinstance(obj, Cube) and (obj != grasped_block) and _is_on(grasped_block, obj):
                offset = self._resolve_offset(obj, cube_size=cube_size) + torch.as_tensor([0.0, 0.0, cube_size])
                break
        return offset


class PlaceBlock3Skill(PlaceBlockSkill):
    def __init__(
        self,
        scene: Scene,
        place_skill: PlaceSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
    ) -> None:
        """Initialize the place block skill."""
        super().__init__(scene, place_skill, vis_target_pos)
        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.MultiDiscrete((self.max_objects,) * 3), name="block_id", is_torch=False, is_batched=False
        )
        self._params = None

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        self._params = self.params_spec.cast(params[:3])

        objs = self._scene.get_objects_from_id(self._params)
        self._target = objs[1]
        if isinstance(self._target, Table):
            if isinstance(objs[2], Location):
                target_xyz = objs[2].pose[:3].clone()
                target_xyz[0] = target_xyz[0] + 0.025
                target_xyz[2] = 0.0
                target_xyz = target_xyz.to(self.obs_spec.device) + (self._offset / 2)
            else:
                target_xyz = find_valid_table_xy(self._scene).to(self.obs_spec.device) + (self._offset / 2)

        else:
            raise ValueError(f"Unknown place object: {self._target}.")

        # Check for blocks under the grasped block
        target_xyz = target_xyz + self._resolve_offset(objs[0]).to(target_xyz.device)

        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        target_pose = torch.tensor([target_xyz[0], target_xyz[1], target_xyz[2], yaw])
        target_pose = self._place_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._place_skill.initiate(obs, target_pose)
        print(f"[INFO][PLACE BLOCK]: {objs[0].name} | {self._target.name} | {objs[2].name}")

    def __str__(self) -> str:
        if self._params is not None:
            names = self._scene.resolve_ids_to_names(self._params)
            return f"Place Block: | {names[0]} | {names[1]} |"
        return "Place Block: | Unset | Unset |"

    def _resolve_offset(self, grasped_block: Cube, cube_size: float = 0.044) -> torch.Tensor:
        """Resolve the offset due to any blocks that might be below the grasped block."""
        offset = torch.as_tensor([0.0, 0.0, 0.0])
        for obj in self._scene.objects:
            if isinstance(obj, Cube) and (obj != grasped_block) and _is_on(grasped_block, obj):
                offset = self._resolve_offset(obj, cube_size=cube_size) + torch.as_tensor([0.0, 0.0, cube_size])
                break
        return offset


class PlaceBlock4Skill(PlaceBlockSkill):
    def __init__(
        self,
        scene: Scene,
        place_skill: PlaceSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
    ) -> None:
        """Initialize the place block skill."""
        super().__init__(scene, place_skill, vis_target_pos)
        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.MultiDiscrete((self.max_objects,) * 4), name="block_id", is_torch=False, is_batched=False
        )
        self._params = None

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        self._params = self.params_spec.cast(params[:4])

        objs = self._scene.get_objects_from_id(self._params)
        self._target = objs[1]
        if isinstance(self._target, Cube):
            if not self._target.is_pose_known() or self._params[0] == self._params[1]:
                self._status = torch.as_tensor(SkillStatusCodes.FAILED, device=self.params_spec.device)

                return
            target_xyz = self._target.pose[:3].to(self.obs_spec.device).clone() + self._offset
        elif isinstance(self._target, Table):
            if isinstance(objs[2], Location):
                target_xyz = objs[2].pose[:3].clone()
                target_xyz[0] = target_xyz[0] + 0.025
                target_xyz[2] = 0.0
                target_xyz = target_xyz.to(self.obs_spec.device) + (self._offset / 2)
            else:
                target_xyz = find_valid_table_xy(self._scene).to(self.obs_spec.device) + (self._offset / 2)

        else:
            raise ValueError(f"Unknown place object: {self._target}.")

        # Check for blocks under the grasped block
        target_xyz = target_xyz + self._resolve_offset(objs[0]).to(target_xyz.device)

        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        target_pose = torch.tensor([target_xyz[0], target_xyz[1], target_xyz[2], yaw])
        target_pose = self._place_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._place_skill.initiate(obs, target_pose)
        print(f"[INFO][PLACE BLOCK]: {objs[0].name} | {self._target.name} | {objs[2].name}")

    def __str__(self) -> str:
        if self._params is not None:
            names = self._scene.resolve_ids_to_names(self._params)
            return f"Place Block: | {names[0]} | {names[1]} |"
        return "Place Block: | Unset | Unset |"

    def _resolve_offset(self, grasped_block: Cube, cube_size: float = 0.044) -> torch.Tensor:
        """Resolve the offset due to any blocks that might be below the grasped block."""
        offset = torch.as_tensor([0.0, 0.0, 0.0])
        for obj in self._scene.objects:
            if isinstance(obj, Cube) and (obj != grasped_block) and _is_on(grasped_block, obj):
                offset = self._resolve_offset(obj, cube_size=cube_size) + torch.as_tensor([0.0, 0.0, cube_size])
                break
        return offset
