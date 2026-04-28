"""An abstract model of the scene."""

from typing import Any

from unified_planning.engines import PlanGenerationResultStatus as PGResultStatus
from unified_planning.io import PDDLReader
from unified_planning.model import Object, Problem
from unified_planning.shortcuts import And, OneshotPlanner

from skillet.planning.abstract import (
    AbstractGoal,
    AbstractPlan,
    AbstractState,
    ParsedUpProblem,
    UPDictFluent,
    UPListGoal,
    ground_cube_on_relations,
    ground_gripper_relations,
    parse_action,
    parse_value,
)
from skillet.planning.base_planner import BasePlanner
from skillet.scene.base import Scene
from skillet.scene.cube import Cube


class AbstractModel(BasePlanner):
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
        self._problem: Problem = None
        self._init_state: AbstractState = None
        self._goal: UPListGoal = None

    @property
    def init_state(self) -> AbstractState:
        return self._init_state

    @property
    def goal(self) -> UPListGoal:
        return self._goal

    def initialize(self, scene: Scene | None = None, task_file: str | None = None) -> None:
        """Initialize the abstract model with the given scene and task."""
        if scene is not None:
            self._scene = scene
        if task_file is not None:
            self._task_file = task_file
        try:
            # If task file is none, will return an incomplete problem which can be filled in get_abstract_state
            self._problem: Problem = self._pddl_reader.parse_problem(self._domain_file, self._task_file)
        except Exception as e:
            raise PDDLParsingError(f"Error parsing PDDL file: {e}") from e

    def get_abstract_state(self, goal: dict[str, Any] | None = None) -> ParsedUpProblem:
        """Get the current abstract state of the scene."""
        assert self._scene is not None
        # Create all the objects in the scene (Cubes + Table)
        block_type = self._problem.user_type("block")
        object_state = {ob_name: Object(ob_name, block_type) for ob_name in self._scene.get_object_names(Cube)}
        object_state[self._scene.table.name] = Object(self._scene.table.name, self._problem.user_type("table"))

        # Perform predicate grounding for on, clear, small, handempty and holding
        fluent_state = {}
        on_pred, clear_pred = ground_cube_on_relations(self._scene)
        empty_pred, holding_pred = ground_gripper_relations(self._scene)
        if "on" in [f.name for f in self._problem.fluents]:
            for op in on_pred:
                o_fluent = self._problem.fluent(op[0])(*(object_state[op[1].name], object_state[op[2].name]))
                fluent_state[o_fluent] = True

        if "clear" in [f.name for f in self._problem.fluents]:
            for cp in clear_pred:
                fluent = self._problem.fluent(cp[0])(*(object_state[cp[1].name],))
                fluent_state[fluent] = True
        if "small" in [f.name for f in self._problem.fluents]:
            for ob in object_state.values():
                if ob.type == block_type:
                    fluent = self._problem.fluent("small")(*(ob,))
                    fluent_state[fluent] = True
        if "handempty" in [f.name for f in self._problem.fluents]:
            fluent = self._problem.fluent("handempty")
            fluent_state[fluent] = empty_pred
        if "holding" in [f.name for f in self._problem.fluents]:
            for hp in holding_pred:
                fluent = self._problem.fluent(hp[0])(*(object_state[hp[1].name],))
                fluent_state[fluent] = True

        # Parse the goal
        if goal is not None:
            self._scene.goal = goal

        goals = (
            self._create_goal(self._scene.goal, object_state) if self._scene.goal is not None else self._problem.goals
        )
        if len(goals) == 0:
            print("[WARN][ABSTRACT MODEL] Empty goal list!")
        return ParsedUpProblem(fluents=fluent_state, objects=object_state, goals=goals)

    def reset_abstract_state(self, problem: Problem, state: ParsedUpProblem) -> None:
        """Update the initial state of the problem based on a ParsedUpProblem.

        Note: Assumes that the problem is empty (no objects, goals, initial values).

        TODO: Support non-empty problems?
        """
        problem.add_objects(list(state.objects.values()))
        problem.add_goal(And(*list(state.goals)))
        [problem.set_initial_value(fluent, value) for fluent, value in state.fluents.items()]

        return problem

    def plan(
        self,
        abstract_state: ParsedUpProblem | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, AbstractPlan | None]:
        """Plan the sequence of actions to execute to complete the task.

        Args:
            abstract_state: Unified planning dict of current abstract state.

        """
        if abstract_state is not None:
            self._problem = self.reset_abstract_state(self._problem, abstract_state)

        self._init_state = AbstractState(
            states=[parse_value(str(v)) for v in list(self._problem.explicit_initial_values.keys())]
        )
        self._goal = AbstractGoal(goals=[parse_value(str(g)) for g in abstract_state.goals])

        with OneshotPlanner(name="fast-downward") as planner:
            result = planner.solve(self._problem, timeout=timeout)

            status = result.status

            if status not in (PGResultStatus.SOLVED_SATISFICING, PGResultStatus.SOLVED_OPTIMALLY):
                return (False, None)

        return True, AbstractPlan(actions=[parse_action(str(action)) for action in result.plan.actions])

    def _create_goal(self, goal: dict[str, Any], object_state: list[Object]) -> list[UPDictFluent]:
        """Parse the output of the VLM goal into the problem.

        Note: currently only supports parsing one `on` goal.
        """
        return [
            self._problem.fluent(g["goal_predicate"])(*(object_state[g["args"][0]], object_state[g["args"][1]]))
            for g in goal
        ]


class PDDLParsingError(Exception):
    """An error that occurs when parsing a PDDL file."""

    pass
