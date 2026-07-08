from collections.abc import Callable, Sequence
from typing import TypeAlias, override

import gymnasium as gym
import torch

from skillet.core import SkillParamsSpec
from skillet.core.skill import SingleSkill, SkillStatus, SkillStatusCodes
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action
from skillet.scene.base import Scene, SceneObject
from skillet.skill.high_level import WipeSkill

Object_Params: TypeAlias = int
"""The parameters for selecting an object in the scene."""


class WipeSurfaceSkill(SingleSkill[IKEE_Obs, M_Action, Object_Params]):
    """A skill for wiping across a spill.

    Is is discretely parameterized by the id spill.
    """

    def __init__(
        self,
        scene: Scene,
        wipe_skill: WipeSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
    ) -> None:
        """Initialize the pick block skill."""
        self._scene = scene
        self._wipe_skill = wipe_skill
        self.max_objects = len(scene.objects) if scene.closed_set else 100

        self._spill_params_spec = SkillParamsSpec(
            space=gym.spaces.Discrete(n=self.max_objects), name="spill_id", is_torch=False, is_batched=False
        )
        self._status = None
        self._offset = torch.tensor([0, 0.0, 0.02], device=self.obs_spec.device)

        self._vis_target_pos = vis_target_pos

    @property
    def name(self) -> str:
        """The name of the skill."""
        return "wipe_table"

    @property
    @override
    def policy(self):
        return self._wipe_skill.policy

    @property
    def obs_spec(self):
        """The specification of the observation space for the skill."""
        return self._wipe_skill.obs_spec

    @property
    def action_spec(self):
        """The specification of the action space for the skill."""
        return self._wipe_skill.action_spec

    @property
    def params_spec(self):
        """The specification of the parameters space for the skill."""
        return self._spill_params_spec

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        params = self.params_spec.cast(params)

        self._target_spill: SceneObject = self._scene.objects(params)
        if not self._target_spill.is_pose_known():
            self._status = torch.as_tensor(SkillStatusCodes.FAILED, device=self.params_spec.device)
            return
        start_xyz = self._target_spill.bbox[:, 0:3]
        start_xyz[2] = 0
        start_xyz = start_xyz + self._offset
        end_xyz = self._target_spill.bbox[:, 3:6]
        end_xyz[2] = 0
        end_xyz = end_xyz + self._offset
        if self._vis_target_pos is not None:
            self._vis_target_pos(start_xyz)
        yaw = 0
        target_pose = torch.tensor([start_xyz[0], start_xyz[1], start_xyz[2], yaw, end_xyz[0], end_xyz[1], end_xyz[2]])
        target_pose = self._wipe_skill.params_spec.with_n_envs(1).cast(target_pose)
        self._wipe_skill.initiate(obs, target_pose)

    @override
    def get_action(self, obs: IKEE_Obs) -> M_Action:
        obs = self._wipe_skill.obs_spec.cast(obs)
        actions = self._wipe_skill.get_action(obs)
        return self.action_spec.cast(actions)

    @property
    def status(self) -> SkillStatus:
        """The status of the skills."""
        if self._status is not None:
            return self._status
        return self._wipe_skill.status[0]

    @status.setter
    def status(self, st: SkillStatus) -> None:
        self._status = torch.as_tensor(st, device=self.params_spec.device)


class WipeSurface2Skill(WipeSurfaceSkill):
    def __init__(
        self,
        scene: Scene,
        wipe_skill: WipeSurfaceSkill,
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
        xyz_offset: tuple[int] = (0, 0.0, 0.02),
    ) -> None:
        """Initialize the wipe object skill."""
        super().__init__(scene, wipe_skill, vis_target_pos, xyz_offset=xyz_offset)
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

        target_xyz_xyz = torch.cat(
            (target_xyz, torch.zeros(target_xyz.shape[0], device=target_xyz.device), target_xyz), dim=1
        )
        target_xyz_xyz[:, 0] = target_xyz_xyz[:, 0] - 0.05  # TODO Sponge
        target_xyz_xyz[:, 3] = target_xyz_xyz[:, 3] + 0.05  # Wipe along the x axis 10 cm offset from the center
        target_pose = self._wipe_skill.params_spec.with_n_envs(1).cast(target_xyz_xyz)

        self._wipe_skill.initiate(obs, target_pose)
        print(f"[INFO][WIPE OBJECT]: {self._target_object.name} | {objs[1].name}")

    def __str__(self) -> str:
        if self._params is not None:
            names = self._scene.resolve_ids_to_names(self._params)
            return f"Wipe Object: | {names[0]} | {names[1]} |"
        return "Wipe Object: | Unset | Unset |"
