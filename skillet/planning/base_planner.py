"""The superclass for the skillet planning API."""

from abc import ABC, abstractmethod
from typing import Any

from skillet.scene.base import Scene


class BasePlanner(ABC):
    """The superclass for the skillet planning API."""

    @property
    def init_state(self) -> Any:
        raise NotImplementedError

    @property
    def goal(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, scene: Scene | None = None) -> None:
        """Initialize the abstract model with the given scene."""
        raise NotImplementedError

    def get_abstract_state(self, goal: dict[str, Any] | None = None) -> Any:
        """Get the current abstract state of the scene."""
        raise NotImplementedError

    def reset_abstract_state(self, problem: Any, state: Any) -> None:
        """Update the initial state of the problem based on a ParsedUpProblem.

        Note: Assumes that the problem is empty (no objects, goals, initial values).

        """
        raise NotImplementedError

    def plan(
        self,
        abstract_state: Any | None = None,
        timeout: float = 10.0,
    ) -> Any:
        """Plan the sequence of actions to execute to complete the task.

        Args:
            abstract_state: Unified planning dict of current abstract state.

        """
        raise NotImplementedError

    @abstractmethod
    def _create_goal(self, goal: dict[str, Any], object_state: list[Any]) -> list[Any]:
        raise NotImplementedError
