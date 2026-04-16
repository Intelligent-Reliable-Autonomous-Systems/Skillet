"""RL policy definition."""

import io
import threading
import time
from pathlib import Path
from typing import Any, Generic

import torch
import yaml

from skillet.controllers import PidController
from skillet.core.policy import BatchedUPolicy, TBAction, TBPolicyObs
from skillet.core.skill import JOINT_Params, JOINT_Params_Spec, SkillParamsSpec
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.specs import JOINT_Obs


class RlPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
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
            self._policy = torch.jit.load(file)
        except FileNotFoundError:
            print(f"[WARNING][RlPolicy] Unable to load Torch Jit file: `{agent_fpath}/agent.pt`")

        try:
            with Path.open(f"{agent_fpath}/config.yaml", "rb") as f:
                file = io.BytesIO(f.read())
            self._policy_cfg = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"[WARNING][RlPolicy] Unable to load policy config file: `{agent_fpath}/config.yaml`")

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    @property
    def params_spec(self) -> SkillParamsSpec[JOINT_Params]:
        """The parameter specification for joint parameters."""
        return JOINT_Params_Spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        policy_obs = self.cat_params_and_obs(obs, params)
        return self._policy(policy_obs)


class PidRlPolicy(RlPolicy):
    """Class for the RL policy with a PID controller."""

    def __init__(
        self,
        obs_spec: ObservationSpec[TBPolicyObs],
        action_spec: ActionSpec[TBAction],
        agent_fpath: str,
        poll_rate_hz: int = 20,
    ) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.
            agent_fpath: A str (path) to a folder containing jit-compiled torch policy named `agent.pt` and a
                `config.yaml` file describing the input (observation) space and the output
            poll_rate_hz: The poll rate of the RL policy

        """
        super().__init__(self, obs_spec, action_spec, agent_fpath=agent_fpath)

        self._pid_controller = PidController()
        self._poll_rate_hz = poll_rate_hz
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._curr_obs = None
        self.run_thread()

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next velocity action from the PID controller."""
        self._curr_obs = obs
        return self._pid_controller.get_action(obs["joint_pos"])

    def reset(self, obs: JOINT_Obs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        self._params = params
        self._curr_obs = obs

    def policy_inference(self) -> None:
        """Run the policy inference thread by polling the RL policy at a specified Hz."""
        poll_period_s = 1.0 / self.poll_rate_hz
        next_poll_t = time.perf_counter()

        while not self._stop_event.is_set():
            if self._curr_obs is not None:
                self._pos_desired = self._policy(self._build_policy_obs(self._curr_obs, self._params))
                self._pid_controller.reset(self._pos_desired)
            sleep_time = (time.perf_counter() - next_poll_t) - poll_period_s
            if sleep_time < 0:
                time.sleep(min(-sleep_time, poll_period_s))
            else:
                print(f"[WARN][RlPolicy] full loop overran by {sleep_time * 1000:.1f}ms")
            next_poll_t = time.perf_counter()

    def run_thread(self) -> None:
        """Start the policy inference thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._policy_inference, name="RlPolicyInference", daemon=True)
        self._thread.start()

    def _build_policy_obs(self, obs: JOINT_Obs, params: Any = None) -> None:
        """Build the observation for the policy."""
        pass
