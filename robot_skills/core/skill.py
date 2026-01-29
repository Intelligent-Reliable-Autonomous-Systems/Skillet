

class Skill(abc.ABC):
    """A skill that represents a high-level action in the environment."""

    @property
    @abc.abstractmethod
    def policy(self) -> Policy:
        """The policy for the skill."""
        raise NotImplementedError

    @property
    def observation_spec(self) -> ObservationSpec:
        """The specification of the observation space for the skill."""
        return self.policy.observation_spec

    @property
    def params_spec(self) -> SkillParamsSpec:
        """The specification of the parameters space for the skill."""
        return SkillParamsSpec.EMPTY

    @property
    @abc.abstractmethod
    def status(self) -> SkillStatus:
        """The status of the skill."""
        raise NotImplementedError

    @abc.abstractmethod
    def can_initiate(self, obs: Observation) -> bool:
        """Check if the skill can be initiated with the given observation."""
        raise NotImplementedError

    def initiate(self, obs: Observation, params: SkillParams) -> None:
        """Initiate the skill with the given observation."""
        pass

    @abc.abstractmethod
    def get_action(self, obs: Observation) -> Action | SkillStatus:
        """Get the next action for the skill based on the observation. Return the action if the skill is not terminated, otherwise return the result of the skill."""
        raise NotImplementedError
