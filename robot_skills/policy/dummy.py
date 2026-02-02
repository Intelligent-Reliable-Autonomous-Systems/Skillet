

from typing import Any, Generic
from robot_skills.core.policy import BatchedUPolicy, Policy, TBPolicyObs, TBAction, TBPolicyParams, UPolicy
from robot_skills.core.spaces import ActionSpec, ObservationSpec, SkillParamsSpec


class RandomPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], 
            action_spec: ActionSpec[TBAction]) -> None:
        self._obs_spec = obs_spec
        self._action_spec = action_spec

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        n_envs = self._obs_spec.n_envs_from(obs)
        return self._action_spec.with_n_envs(n_envs).sample()

class ZeroPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], 
            action_spec: ActionSpec[TBAction]) -> None:
        self._obs_spec = obs_spec
        self._action_spec = action_spec

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        n_envs = self._obs_spec.n_envs_from(obs)
        return self._action_spec.with_n_envs(n_envs).zeros()