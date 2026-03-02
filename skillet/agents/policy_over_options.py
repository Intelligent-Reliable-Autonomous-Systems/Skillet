"""A high level model-free agent that selects options/skills to execute in parallel."""

from typing import Any, Generic, TypeVar, cast

import torch
from jaxtyping import Bool, Int

from skillet.core.env import BatchedEnvironment
from skillet.core.policy import BatchedUPolicy
from skillet.core.skill import BatchedSkill, CompositeSkill
from skillet.core.spaces import (
    BatchedAction,
    BatchedObservation,
    BatchedSkillParams,
)

THighLevelObs = TypeVar("THighLevelObs", bound=BatchedObservation)
"""The type of the high level observation, batched."""
TLowLevelObs = TypeVar("TLowLevelObs", bound=BatchedObservation)
"""The type of the low level observation, batched."""
TSkillParams = TypeVar("TSkillParams", bound=BatchedSkillParams)
"""The type of the skill parameters, batched."""
TBAction = TypeVar("TBAction", bound=BatchedAction)
"""The type of the batched action, batched."""

SelectedSkills = Int[torch.Tensor, "b"]
"""The indices of the selected skills for each environment according to the order of the skills."""


class PolicyOverOptionsAgent(Generic[THighLevelObs, TLowLevelObs, TBAction, TSkillParams]):
    """A high level model-free agent that selects options/skills to execute in parallel.

    Generic type parameters:
        THighLevelObs: The type of the high level observation, batched.
        TLowLevelObs: The type of the low level observation, batched.
        TBAction: The type of the batched action, batched.
        TSkillParams: The type of the skill parameters, batched.

    Args:
        skills: The list of skills to execute.
        high_level_policy: The high level policy to select the skills to execute.
        params_policy: The policy to sample the parameters for the skills.

    """

    def __init__(
        self,
        skills: list[BatchedSkill[TLowLevelObs, TBAction, TSkillParams]],
        high_level_policy: BatchedUPolicy[THighLevelObs, SelectedSkills],
        params_policy: BatchedUPolicy[THighLevelObs, TSkillParams] | None = None,
    ) -> None:
        self.skills = skills
        self.high_level_policy = high_level_policy
        self.params_policy = params_policy

    def get_high_level_obs(self, env: BatchedEnvironment) -> THighLevelObs:
        """Get high level policy observations."""
        return env.get_observation(self.high_level_policy.obs_spec)

    def get_low_level_obs(self, env: BatchedEnvironment) -> TLowLevelObs:
        """Get low level policyobservations."""
        return env.get_observation(self.skills[0].obs_spec)

    def execute(self, env: BatchedEnvironment[Any, TBAction]) -> None:
        """Execute the policy over the options configured."""
        n_envs = env.num_envs
        terminated = cast('Bool[torch.Tensor, "n_envs"]',
            env.obs_spec.with_n_envs(n_envs).zeros(shape=(n_envs,), dtype=torch.bool))
        composite_skill = CompositeSkill[TLowLevelObs, TBAction, TSkillParams](self.skills)

        while not terminated.all():
            # High level execution
            high_level_obs = self.get_high_level_obs(env)
            # 1. Select the skills to execute
            selected_skills = self.high_level_policy.get_action(high_level_obs)
            # 2. Sample the parameters for the skills
            if self.params_policy is not None:
                params = self.params_policy.get_action(high_level_obs)
            else:
                params = self.skills[0].params_spec.with_n_envs(n_envs).zeros()
            # override_grip = 0
            # Low level execution
            # 3. Initiate the composite skill with the selected skills and parameters
            composite_skill.initiate(env.get_observation(composite_skill.obs_spec), params, env_ids=selected_skills)
            print("initiating skills:", [(self.skills[i].name, p) for i, p in zip(selected_skills, params)])
            # 4. While not terminated, get the next action and take a step in the environment
            skill_dones = composite_skill.is_terminated(env.get_observation(composite_skill.obs_spec))
            while not skill_dones.all() and not bool(terminated.all()):
                # 4a. Get the next action with the low-level observation
                action = composite_skill.get_action(env.get_observation(composite_skill.obs_spec))
                # print("Taking action", action)
                # action[:, -1] = override_grip
                # 4b. Take a step in the environment
                _, r, term, trunc, _ = env.step(action)
                terminated = terminated | term | trunc
                # 4c. Check if the composite skill is terminated
                skill_dones = composite_skill.is_terminated(self.get_low_level_obs(env))
