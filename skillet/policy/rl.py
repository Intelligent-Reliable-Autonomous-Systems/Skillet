"""RL policy definition."""

import io
from pathlib import Path
from typing import Any, Generic

import torch
import yaml

from skillet.core.policy import BatchedUPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec


class RLPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that samples actions from the action space."""

    def __init__(
        self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction], agent_fpath: str
    ) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.
            agent_fpath: A str (path) to a folder containing jit-compiled torch policy named `agent.pt` and a
                `config.yaml` file describing the input (observation) space and the output

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        try:
            with Path.open(f"{agent_fpath}/agent.pt", "rb") as f:
                file = io.BytesIO(f.read())
            self.policy = torch.jit.load(file)
        except FileNotFoundError:
            print(f"[WARNING][RLPolicy] Unable to load Torch Jit file: `{agent_fpath}/agent.pt`")

        try:
            with Path.open(f"{agent_fpath}/config.yaml", "rb") as f:
                file = io.BytesIO(f.read())
            self.policy_cfg = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"[WARNING][RLPolicy] Unable to load policy config file: `{agent_fpath}/config.yaml`")

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:  # noqa: D102
        policy_obs = self.cat_params_and_obs(obs, params)
        return self.policy(policy_obs)

    def cat_params_and_obs(self, obs: TBPolicyObs, params: Any = None) -> TBPolicyObs:
        """Concatenate the environment observations and parameters according to policy_cfg.

        Args:
            obs: Envirionment observations
            params: Skill parameters

        Returns:
            Input to the RL Policy

        """
        return torch.cat((obs, params), dim=1)
