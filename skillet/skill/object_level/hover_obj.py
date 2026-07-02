from collections.abc import Callable, Sequence
from typing import TypeAlias, override

import gymnasium as gym
import torch

from skillet.core import SkillParamsSpec
from skillet.core.skill import SingleSkill, SkillStatus, SkillStatusCodes
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action
from skillet.scene.base import Scene, SceneObject
from skillet.skill.high_level.hover import HoverSkill

Object_Params: TypeAlias = int
"""The parameters for selecting an object in the scene."""


class HoverObjectSkill(SingleSkill[IKEE_Obs, M_Action, Object_Params]):
    """A skill for hovering a object from a stack.

    Is is discretely parameterized by the id of the object to hover.
    """

    def __init__(
        self,
        scene: Scene,
        hover_skill: HoverSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
        xyz_offset: tuple[int] = (0, 0.0, 0.02),
    ) -> None:
        """Initialize the hover object skill."""
        self._scene = scene
        self._hover_skill = hover_skill
        self.max_objects = len(scene.objects) if scene.closed_set else 100

        self._object_params_spec = SkillParamsSpec(
            space=gym.spaces.Discrete(n=self.max_objects), name="object_id", is_torch=False, is_batched=False
        )
        self._status = None
        self._offset = torch.tensor(xyz_offset, device=self.obs_spec.device)

        self._vis_target_pos = vis_target_pos

    @property
    def name(self) -> str:
        """The name of the skill."""
        return "hover_object"

    @property
    @override
    def policy(self):
        return self._hover_skill.policy

    @property
    def obs_spec(self):
        """The specification of the observation space for the skill."""
        return self._hover_skill.obs_spec

    @property
    def action_spec(self):
        """The specification of the action space for the skill."""
        return self._hover_skill.action_spec

    @property
    def params_spec(self):
        """The specification of the parameters space for the skill."""
        return self._object_params_spec

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        params = self.params_spec.cast(params)

        self._target_object: SceneObject = self._scene.objects(params)
        if not self._target_object.is_pose_known():
            self._status = torch.as_tensor(SkillStatusCodes.FAILED, device=self.params_spec.device)
            return
        target_xyz = self._target_object.pose[:3] + self._offset
        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        target_pose = torch.tensor([target_xyz[0], target_xyz[1], target_xyz[2], yaw])
        target_pose = self._hover_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._hover_skill.initiate(obs, target_pose)

    @override
    def get_action(self, obs: IKEE_Obs) -> M_Action:
        obs = self._hover_skill.obs_spec.cast(obs)
        actions = self._hover_skill.get_action(obs)
        return self.action_spec.cast(actions)

    @property
    def status(self) -> SkillStatus:
        """The status of the skills."""
        if self._status is not None:
            return self._status
        return self._hover_skill.status[0]

    @status.setter
    def status(self, st: SkillStatus) -> None:
        self._status = torch.as_tensor(st, device=self.params_spec.device)


class HoverObject2Skill(HoverObjectSkill):
    def __init__(
        self,
        scene: Scene,
        hover_skill: HoverSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
        xyz_offset: tuple[int] = (0, 0.0, 0.02),
    ) -> None:
        """Initialize the hover object skill."""
        super().__init__(scene, hover_skill, vis_target_pos, xyz_offset=xyz_offset)
        self._object_params_spec = SkillParamsSpec(
            space=gym.spaces.MultiDiscrete((self.max_objects,) * 2), name="object_id", is_torch=False, is_batched=False
        )
        self._params = None

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        self._params = self.params_spec.cast(params[:2])

        objs = self._scene.get_objects_from_id(self._params)
        self._target_object = objs[1]

        target_xyz = self._target_object.pose[:3].to(self.obs_spec.device).clone() + self._offset
        if self._vis_target_pos is not None:
            self._vis_target_pos(target_xyz)
        yaw = 0
        target_pose = torch.tensor([target_xyz[0], target_xyz[1], 0.25, yaw])
        target_pose = self._hover_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._hover_skill.initiate(obs, target_pose)
        print(f"[INFO][HOVER OBJECT]: {self._target_object.name}")

    def __str__(self) -> str:
        if self._params is not None:
            names = self._scene.resolve_ids_to_names(self._params)
            return f"Hover Object: | {names[0]} | {names[1]} |"
        return "Hover Object: | Unset | Unset |"
