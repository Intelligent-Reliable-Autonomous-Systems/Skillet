"""A Sim2Real Executor for handling policy inference and low level control."""

from typing import Any

from skillet.agents.base_agent import Agent, SelectedSkill, TAction, THighLevelObs, TLowLevelObs, TSkillParams
from skillet.core.env import BatchedEnvironment, Environment
from skillet.core.policy import UPolicy
from skillet.core.skill import SingleSkill
from skillet.scene.base import Scene


class S2RAgent(Agent):
    """Main S2R class."""

    def __init__(
        self,
        scene: Scene,
        skills: list[SingleSkill[TLowLevelObs, TAction, TSkillParams]],
        high_level_policy: UPolicy[THighLevelObs, SelectedSkill],
        params_policy: UPolicy[THighLevelObs, TSkillParams] | None = None,
    ) -> None:
        """Initialize the S2R agent.

        Args:
            scene: The scene to execute the skills in.
            skills: list of skills the agent can use
            high_level_policy: policy deciding which skills to use
            params_policy: policy deciding which parameters to use

        """
        super().__init__()

        self._scene = scene
        self.skills = skills
        self.high_level_policy = high_level_policy
        self.params_policy = params_policy

    def get_high_level_obs(self, env: Environment) -> THighLevelObs:
        """Get high level policy observations."""
        return env.get_observation(self.high_level_policy.obs_spec)

    def get_low_level_obs(self, env: BatchedEnvironment) -> TLowLevelObs:
        """Get low level policy observations."""
        return env.get_observation(self.skills[0].obs_spec)

    def execute(self, env: Environment[Any, TAction]) -> None:
        """Execute the policy over the options configured."""
        terminated = False
        while not terminated:
            # High level execution
            high_level_obs = self.get_high_level_obs(env)
            # 1. Select the skill to execute
            selected_skill_id = self.high_level_policy.get_action(high_level_obs)
            self._selected_skill = self.skills[selected_skill_id]
            # 2. Sample the parameters for the skills
            if self.params_policy is not None:
                params = self.params_policy.get_action(high_level_obs)
            else:
                params = self._selected_skill.params_spec.zeros()
            # override_grip = 0
            # Low level execution
            # 3. Initiate the composite skill with the selected skills and parameters
            self._selected_skill.initiate(env.get_observation(self._selected_skill.obs_spec), params)
            print("initiating skill:", (self._selected_skill.name, params))
            # 4. While not terminated, get the next action and take a step in the environment
            skill_done = self._selected_skill.is_terminated(env.get_observation(self._selected_skill.obs_spec))

            while not skill_done and not bool(terminated):
                # 4a. Get the next action with the low-level observation
                action = self._selected_skill.get_action(env.get_observation(self._selected_skill.obs_spec))
                # action[:, -1] = override_grip
                # 4b. Take a step in the environment
                _, _, term, trunc, _ = env.step(action, action_spec=self._selected_skill.action_spec)
                terminated = terminated | term | trunc
                # 4c. Check if the composite skill is terminated
                skill_done = self._selected_skill.is_terminated(env.get_observation(self._selected_skill.obs_spec))
