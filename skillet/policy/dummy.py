"""Define simple dummy policies for testing."""

from typing import Any, Generic

from skillet.core.policy import BatchedUPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec


class RandomPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that samples actions from the action space."""

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:  # noqa: D102
        n_envs = self._obs_spec.n_envs_from(obs)
        return self._action_spec.with_n_envs(n_envs).sample()


class ZeroPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that outputs zero actions."""

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:  # noqa: ANN401, D102
        n_envs = self._obs_spec.n_envs_from(obs)
        return self._action_spec.with_n_envs(n_envs).zeros()
