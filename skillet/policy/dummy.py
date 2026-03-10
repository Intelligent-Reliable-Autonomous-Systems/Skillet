"""Define simple dummy policies for testing."""

from typing import Any, Generic
from typing_extensions import override

import torch

from skillet.core.policy import BatchedUPolicy, TAction, TBAction, TBPolicyObs, TPolicyObs, UPolicy
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


class FixedPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that outputs zero actions."""

    def __init__(
        self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction], params: TAction
    ) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.
            params: The parameters

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._params = params

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:  # noqa: ANN401, D102
        n_envs = self._obs_spec.n_envs_from(obs)
        # TODO: Make this general regardless of the type (torch/numpy)
        return self._params.unsqueeze(0).repeat(n_envs, 1)


class RandomFixedPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that outputs zero actions."""

    def __init__(
        self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction], params: TAction
    ) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.
            params: The parameters, a 2D array of parameters to choose from

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._params = params

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:  # noqa: ANN401, D102
        n_envs = self._obs_spec.n_envs_from(obs)
        # TODO: Make this general regardless of the type (torch/numpy)

        indices = torch.multinomial(torch.ones(self._params.shape[0]), num_samples=n_envs, replacement=True)

        # Use these indices to get the random items
        return self._params[indices]


class FixedSequencePolicy(UPolicy[TPolicyObs, TAction], Generic[TPolicyObs, TAction]):
    """A policy that outputs a predefined sequence of actions actions."""

    def __init__(
        self, obs_spec: ObservationSpec[TPolicyObs], action_spec: ActionSpec[TAction], sequence: TAction
    ) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.
            sequence: The sequence of actions to output, of shape (sequence_length, action_dim)

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._sequence = sequence
        self._current_index = 0

    @property
    @override
    def obs_spec(self) -> ObservationSpec[TPolicyObs]:
        return self._obs_spec

    @property
    @override
    def action_spec(self) -> ActionSpec[TAction]:
        return self._action_spec

    @override
    def get_action(self, obs: TPolicyObs, params: Any = None) -> TAction:
        actions = self._sequence[self._current_index]
        self._current_index = (self._current_index + 1) % self._sequence.shape[0]
        # Use these indices to get the random items
        return self._action_spec.cast(actions)
