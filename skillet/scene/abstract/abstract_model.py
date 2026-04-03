"""An abstract model of the scene."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from unified_planning.engines import PlanGenerationResultStatus as PGResultStatus
from unified_planning.io import PDDLReader
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import OneshotPlanner

from skillet.scene.abstract.spatial_grounding import ground_on_relations
from skillet.scene.abstract.up_utils import UPDictState
from skillet.scene.base import Scene, SceneObject

AbstractState: TypeAlias = Any
AbstractTask: TypeAlias = Any
# AbstractAction = tuple[str, *tuple[SceneObject]]
@dataclass(frozen=True)
class AbstractAction:
    """An abstract action."""

    name: str
    args: tuple[SceneObject]
    up_action: ActionInstance

class AbstractModel:
    """An abstract model of the scene."""

    def __init__(self, domain_file: Path) -> None:
        """Initialize the abstract model.

        Args:
            domain_file: The path to the PDDL domain file.

        """
        self.domain_file = domain_file
        self.domain = PDDLReader(domain_file).parse_domain()

    def initialize(self, scene: Scene, task: str) -> None:
        """Initialize the abstract model with the given scene and task."""
        self._scene = scene
        self._task = task

        # TODO: Support getting the problem from a VLM
        try:
            PDDLReader().parse_problem_string(self.domain_file.read_text(), task)
        except Exception as e:
            raise PDDLParsingError(f"Error parsing PDDL file: {e}") from e

    def get_abstract_state(self) -> UPDictState:
        """Get the current abstract state of the scene."""
        # TODO: Get state from VLM or predicate grounding
        abstract_state = {}
        if 'on' in [f.name for f in self.domain.fluents]:
            on_fluent = self.domain.fluent('on')
            for on_pred in ground_on_relations(self._scene):
                abstract_state[on_fluent(on_pred[1].identifier, on_pred[2].identifier)] = True
        return abstract_state

    def plan(self, abstract_state: UPDictState, task: str | None = None, timeout: float = 10.0) -> \
            tuple[bool, list[AbstractAction] | None]:
        """Plan the sequence of actions to execute to complete the task."""
        if task is None:
            task = self._task

        problem = self.domain.clone()
        for fluent, value in abstract_state.items():
            problem.set_initial_value(fluent, value)

        # TODO: Convert the task into a PDDL problem

        with OneshotPlanner(name="fast-downward", problem_kind=problem.kind) as planner:
            result = planner.solve(problem, timeout=timeout)

            status = result.status
            if status not in (PGResultStatus.SOLVED_SATISFICING, PGResultStatus.SOLVED_OPTIMALLY):
                return (False,None)
        return True, [AbstractAction(action.action.name, action.args, action) for action in result.plan]

class PDDLParsingError(Exception):
    """An error that occurs when parsing a PDDL file."""

    pass
