from collections.abc import Callable, Sequence
from typing import TypeAlias, override

import gymnasium as gym
import torch

from skillet.core import SkillParamsSpec
from skillet.core.skill import SingleSkill, SkillStatus
from skillet.envs.specs import BxM_Action, IKEE_Obs, M_Action
from skillet.scene.base import Scene
from skillet.skill.high_level import SqueezeSkill

Object_Params: TypeAlias = int
"""The parameters for selecting an object in the scene."""


class SqueezeObjSkill(SingleSkill[IKEE_Obs, M_Action, Object_Params]):
    """A skill for squeezing a sponge.

    Is is discretely parameterized by the sponge id.
    """

    def __init__(
        self,
        scene: Scene,
        squeeze_skill: SqueezeSkill[BxM_Action],
        vis_target_pos: Callable[[Sequence[float]], None] | None = None,
    ) -> None:
        """Initialize the squeeze sponge."""
        self._scene = scene
        self._squeeze_skill = squeeze_skill
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
        return "squeeze_sponge"

    @property
    @override
    def policy(self):
        return self._squeeze_skill.policy

    @property
    def obs_spec(self):
        """The specification of the observation space for the skill."""
        return self._squeeze_skill.obs_spec

    @property
    def action_spec(self):
        """The specification of the action space for the skill."""
        return self._squeeze_skill.action_spec

    @property
    def params_spec(self):
        """The specification of the parameters space for the skill."""
        return self._spill_params_spec

    def initiate(self, obs, params):
        """Initiate the skill with the given observation and parameters."""
        self._status = None
        params = self.params_spec.cast(params)

        self._squeeze_skill.initiate(obs, [])

    @override
    def get_action(self, obs: IKEE_Obs) -> M_Action:
        obs = self._squeeze_skill.obs_spec.cast(obs)
        actions = self._squeeze_skill.get_action(obs)
        return self.action_spec.cast(actions)

    @property
    def status(self) -> SkillStatus:
        """The status of the skills."""
        if self._status is not None:
            return self._status
        return self._squeeze_skill.status[0]

    @status.setter
    def status(self, st: SkillStatus) -> None:
        self._status = torch.as_tensor(st, device=self.params_spec.device)
