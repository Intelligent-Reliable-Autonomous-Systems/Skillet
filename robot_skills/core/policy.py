import abc

class Policy(abc.ABC):
    """A policy that takes an observation and outputs an action."""

    @property
    @abc.abstractmethod
    def observation_spec(self) -> ObservationSpec:
        """The specification of the observation space for the policy."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, obs: Observation, params: SkillParams) -> Action:
        """Get the next low-level action for the robot based on the observation and parameters."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the policy. Useful if policy is stateful."""
        pass