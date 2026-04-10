"""An abstract model of the scene."""

from dataclasses import dataclass
from typing import Any, TypeAlias

import re
from unified_planning.engines import PlanGenerationResultStatus as PGResultStatus
from unified_planning.io import PDDLReader
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.model import Fluent, Object, Problem

from skillet.scene.abstract.spatial_grounding import ground_cube_on_relations
from skillet.scene.abstract.up_utils import UPDictState
from skillet.scene.base import Scene, SceneObject

AbstractState: TypeAlias = Any
AbstractTask: TypeAlias = Any


@dataclass
class AbstractAction:
    action: str
    parameters: list[str]


def parse_action(action_str: str) -> AbstractAction:
    match = re.match(r"([\w-]+)\((.*)\)", action_str.strip())

    action = match.group(1).replace("-", "_")
    parameters = [p.strip() for p in match.group(2).split(",")]

    return AbstractAction(action=action, parameters=parameters)


class AbstractModel:
    """An abstract model of the scene."""

    def __init__(self, domain_file: str, task_file: str | None = None, scene: Scene | None = None) -> None:
        """Initialize the abstract model.

        Args:
            domain_file: The path to the PDDL domain file.
            task_file: The path to the PDDL task file
            scene: The scene object

        """
        self._pddl_reader = PDDLReader()

        self._domain_file = domain_file
        self._task_file = task_file
        self._scene = scene
        self._prefixes = ["block", "cube"]
        self._problem: Problem = None

        if self._scene is not None and self._task_file is not None:
            self.initialize()

    def initialize(self, scene: Scene | None = None, task_file: str | None = None) -> None:
        """Initialize the abstract model with the given scene and task."""
        if scene is not None:
            self._scene = scene
        if task_file is not None:
            self._task_file = task_file
        # TODO: Support getting the task file from VLM
        try:
            self._problem: Problem = self._pddl_reader.parse_problem(self._domain_file, self._task_file)
        except Exception as e:
            raise PDDLParsingError(f"Error parsing PDDL file: {e}") from e

    def get_abstract_state(self) -> UPDictState | None:
        """Get the current abstract state of the scene."""
        # TODO: Get state from VLM or predicate grounding
        if self._scene is None:
            return
        abstract_state = {}
        on_pred, clear_pred = ground_cube_on_relations(self._scene)
        if "on" in [f.name for f in self._problem.fluents]:
            for op in on_pred:
                o_fluent = self._problem.fluent(op[0])(
                    *(self._problem.object(stripw(op[1].name)), self._problem.object(stripw(op[2].name)))
                )
                abstract_state[o_fluent] = True

        if "clear" in [f.name for f in self._problem.fluents]:
            for cp in clear_pred:
                fluent = self._problem.fluent(cp[0])(*(self._problem.object(stripw(cp[1].name)),))
                abstract_state[fluent] = True
        return abstract_state

    def update_initial_state(self, problem: Problem, abstract_state: UPDictState) -> None:
        """Update the initial state of a problem based on a list of true fluent tuples.

        Sets all provided fluents to True and infers which previously-true fluents
        should now be False based on conflicts.

        """
        # Pop all references to "clear" and "on" from the initial values
        # which are grounded by the abstract state
        for fnode, _ in list(problem.explicit_initial_values.items()):
            if fnode.is_fluent_exp():
                fluent = fnode.fluent()
                if fluent.name == "on" or fluent.name == "clear":
                    problem.explicit_initial_values.pop(fnode)

        # Add in the new values from the grounding
        for fluent, value in abstract_state.items():
            problem.set_initial_value(fluent, value)

        return problem

    def plan(
        self,
        abstract_state: UPDictState | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, list[AbstractAction] | None]:
        """Plan the sequence of actions to execute to complete the task.

        Args:
            abstract_state: Unified planning dict of current abstract state.

        """
        if abstract_state is not None:
            self._problem = self.update_initial_state(self._problem, abstract_state)

        with OneshotPlanner(name="fast-downward") as planner:
            result = planner.solve(self._problem, timeout=timeout)

            status = result.status

            if status not in (PGResultStatus.SOLVED_SATISFICING, PGResultStatus.SOLVED_OPTIMALLY):
                return (False, None)
        return True, [parse_action(str(action)) for action in result.plan.actions]

    def get_fluent(self, name: str) -> Fluent:
        """Return a problem fluent."""
        for fl in self._problem.fluents:
            if fl._name == name:
                return fl
        raise ValueError(f"No fluent `{name}` found in {self._problem}")


def stripw(name: str, words: list[str] = ["block", "cube"]) -> str:
    """Strip words from a name."""
    for word in words:
        if word in name:
            name = name.replace(word, "").strip("_")
    return name


def strip_words(result: list[tuple[str]], words: list[str]) -> list[tuple[str]]:
    """Strip words from the objects and remove underscores.

    Args:
        result: list of predicates from abstract model state grounding.
        words: words to remove

    """

    def clean(s):
        for word in words:
            if word in s:
                s = s.replace(word, "").strip("_")
        return s

    return [tuple(clean(item) for item in entry) for entry in result]


class PDDLParsingError(Exception):
    """An error that occurs when parsing a PDDL file."""

    pass
