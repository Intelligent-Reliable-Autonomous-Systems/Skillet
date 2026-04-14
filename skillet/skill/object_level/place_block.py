from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import gymnasium as gym
import numpy as np
import torch
from typing_extensions import override

from skillet.core import SkillParamsSpec
from skillet.core.skill import SingleSkill, SkillStatus, SkillStatusCodes
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action
from skillet.scene import Cube, Table
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
        self._offset = torch.tensor([0, 0.0, 0.035], device=self.obs_spec.device)

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
            self._status = SkillStatusCodes.FAILED.value
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
            space=gym.spaces.MultiDiscrete((self.max_objects, 2)), name="block_id", is_torch=False, is_batched=False
        )
        self._params = None

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        self._params = self.params_spec.cast(params)

        objs = self._scene.get_objects_from_id(self._params)
        self._target = objs[1]
        if isinstance(self._target, Cube):
            if not self._target.is_pose_known():
                self._status = SkillStatusCodes.FAILED.value
                return
            target_xyz = self._target.pose[:3] + self._offset
        elif isinstance(self._target, Table):
            target_xyz = find_valid_table_xy(self._scene).to(self._offset.device) + (self._offset / 2)
        else:
            raise ValueError(f"Unknown place object: {self._target}.")

        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        target_pose = torch.tensor([target_xyz[0], target_xyz[1], target_xyz[2], yaw])
        target_pose = self._place_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._place_skill.initiate(obs, target_pose)

    def __str__(self) -> str:
        names = self._scene.resolve_ids_to_names(self._params)
        return f"Place Block: | {names[0]} | {names[1]} |"
