from collections.abc import Callable, Sequence
from typing import TypeAlias

import gymnasium as gym
import torch
from typing_extensions import override

from skillet.core import SkillParamsSpec
from skillet.core.skill import SingleSkill, SkillStatus, SkillStatusCodes
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action
from skillet.scene.base import Scene
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
        max_objects = len(scene.objects) if scene.closed_set else 100
        self._block_params_spec = SkillParamsSpec(
            space=gym.spaces.Discrete(n=max_objects), name="block_id", is_torch=False, is_batched=False
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
        if params < 0 or params >= len(self._scene.objects):
            self._status = SkillStatusCodes.FAILED.value
            return
        self._target_block = self._scene.objects[params]
        if not self._target_block.is_pose_known():
            self._status = SkillStatusCodes.FAILED.value
            return
        target_xyz = self._target_block.pose[:3] + self._offset
        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0  # TODO: get yaw from target block
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
