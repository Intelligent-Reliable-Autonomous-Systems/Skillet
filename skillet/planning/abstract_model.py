"""An abstract model of the scene."""

import random
from typing import Any, Literal

from unified_planning.engines import PlanGenerationResultStatus as PGResultStatus
from unified_planning.engines import UPSequentialSimulator
from unified_planning.environment import Environment
from unified_planning.io import PDDLReader
from unified_planning.model import MinimizeSequentialPlanLength, Object, Problem, UPState
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import And, AnytimePlanner, Not, OneshotPlanner

from skillet.planning.abstract import (
    AbstractAction,
    AbstractGoal,
    AbstractPlan,
    AbstractState,
    ParsedUpProblem,
    UPDictFluent,
    UPListGoal,
    ground_cube_relations,
    ground_gripper_relations,
    ground_location_relations,
    ground_sponge_gripper_relations,
    ground_sponge_relations,
    parse_action,
    parse_value,
)
from skillet.planning.abstract.up_utils import up_state_to_dict
from skillet.planning.base_planner import BasePlanner
from skillet.scene.base import Scene
from skillet.scene.scene_objs import Bin, Can, Cube, Location, Plate, Spill, Sponge, Target


class AbstractModel(BasePlanner):
    """An abstract model of the scene."""

    def __init__(
        self,
        domain_file: str,
        task_file: str | None = None,
        scene: Scene | None = None,
        environment: Environment | None = None,
        domain: Literal["blocks", "sponge"] = "blocks",
    ) -> None:
        """Initialize the abstract model.

        Args:
            domain_file: The path to the PDDL domain file.
            task_file: The path to the PDDL task file
            scene: The scene object

        """
        self._environment = environment
        self._pddl_reader = PDDLReader(environment=environment)

        self._domain_file = domain_file
        self._task_file = task_file
        self._scene = scene
        self._problem: Problem = None
        self._init_state: AbstractState = None
        self._goal: UPListGoal = None
        self._domain = domain

    @property
    def problem(self) -> Problem:
        if self._problem is None:
            self.initialize()
        return self._problem

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
            self._simulator = UPSequentialSimulator(self._problem)

        except Exception as e:
            raise PDDLParsingError(f"Error parsing PDDL file: {e}") from e

    def get_abstract_state(self, goal: dict[str, Any] | None = None) -> ParsedUpProblem:
        """Get the current abstract state of the scene."""
        assert self._scene is not None
        if self._problem is None:
            self.initialize()

        object_state = {}
        fluent_state = {}
        user_types = [t.name for t in self._problem.user_types]

        if self._domain == "blocks":
            object_state = self._ground_blocks_domain_objects(object_state, user_types)
            fluent_state = self._ground_blocks_domain_preds(fluent_state, object_state)

        elif self._domain == "sponge":
            object_state = self._ground_sponge_domain_objects(object_state, user_types)
            fluent_state = self._ground_sponge_domain_preds(fluent_state, object_state)

        else:
            raise ValueError(f"Unknown domain `{self._domain}`")

        # Parse the goal
        if goal is not None:
            self._scene.goal = goal

        goals = (
            self._create_goal(self._scene.goal, object_state) if self._scene.goal is not None else self._problem.goals
        )

        return ParsedUpProblem(fluents=fluent_state, objects=object_state, goals=goals)

    def reset_abstract_state(self, problem: Problem, state: ParsedUpProblem) -> Problem:
        """Update the initial state of the problem based on a ParsedUpProblem.

        Note: Assumes that the problem is empty (no objects, goals, initial values).

        TODO: Support non-empty problems?
        """
        problem.add_objects(list(state.objects.values()))
        problem.add_goal(And(*list(state.goals)))
        [problem.set_initial_value(fluent, value) for fluent, value in state.fluents.items()]

        return problem

    def reset_up_problem_state(self) -> UPDictFluent:
        """Copy the problem and return a new instantance of the problem."""
        self._problem: Problem = self._pddl_reader.parse_problem(self._domain_file, self._task_file)
        state = self.get_abstract_state()

        self._problem.add_objects(list(state.objects.values()))
        # self._problem.add_goal(And(*list(state.goals)))
        [self._problem.set_initial_value(fluent, value) for fluent, value in state.fluents.items()]
        self._simulator = UPSequentialSimulator(self._problem)
        self._init_state = AbstractState(
            states=[parse_value(str(f), v) for f, v in self._problem.explicit_initial_values.items()]
        )
        return up_state_to_dict(self._simulator.get_initial_state())

    def get_random_action(self, state: UPDictFluent) -> tuple[AbstractAction, ActionInstance]:
        """Get a random action from the available actions in the state."""
        applicable = list(self._simulator.get_applicable_actions(UPState(state, self._problem)))
        action_instance = random.choice(applicable)
        action, params = action_instance
        return parse_action(str(ActionInstance(action, params))), ActionInstance(action, params)

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
            states=[parse_value(str(f), v) for f, v in self._problem.explicit_initial_values.items()]
        )
        self._goal = AbstractGoal(goals=[parse_value(str(g)) for g in abstract_state.goals])


        self._problem.add_quality_metric(MinimizeSequentialPlanLength())
        with AnytimePlanner(
            problem_kind=self._problem.kind, anytime_guarantee="INCREASING_QUALITY"
        ) as planner:
            result = None
            for i, p in enumerate(planner.get_solutions(self._problem, timeout=timeout)):
                result = p
        # with OneshotPlanner(name="fast-downward") as planner:
        #     result = planner.solve(self._problem, timeout=timeout)

        status = result.status

        if status not in (PGResultStatus.SOLVED_SATISFICING, PGResultStatus.SOLVED_OPTIMALLY):
            return (False, None, None)

        return (
            True,
            AbstractPlan(actions=[parse_action(str(action)) for action in result.plan.actions]),
            result.plan.actions,
        )

    def _up_states_from_plan(self, plan, delta: bool = True, as_dict: bool = True):
        s = self._simulator.get_initial_state()
        states = [up_state_to_dict(s) if as_dict else s]
        for action in plan.actions:
            s = self._simulator.apply(s, action)
            states.append(up_state_to_dict(s) if as_dict else s)

        if delta:
            deltas = []
            for s0, s1 in zip(states[:-1], states[1:]):
                delta = {x: s1[x] for x in s1 if (x not in s0) or (s0[x] != s1[x])}
                delta.update({x: False for x in s0 if (x not in s1) and s0[x]})
                deltas.append(delta)
            return deltas
        return states

    # def _create_goal(self, goal: dict[str, Any], object_state: list[Object]) -> list[UPDictFluent]:
    #     """Parse the output of the VLM goal into the problem."""
    #     return [self._problem.fluent(g["predicate"])(*(object_state[arg] for arg in g["args"])) for g in goal]

    def _create_goal(self, goal: dict[str, Any], object_state: list[Object]) -> list[UPDictFluent]:
        """Parse the output of the VLM goal into the problem."""
        goals = []
        for g in goal:
            predicate = g["predicate"]
            negated = predicate.startswith("not ")
            if negated:
                predicate = predicate[len("not ") :]

            fluent_expr = self._problem.fluent(predicate)(*(object_state[arg] for arg in g["args"]))

            if negated:
                fluent_expr = Not(fluent_expr)

            goals.append(fluent_expr)

        return goals

    def _ground_blocks_domain_objects(self, object_state: dict, user_types: list) -> dict[str, Object]:
        """Ground the objects in the blocks domain."""
        # Create all the objects in the scene (Cubes + Table + Locations + Targets)
        if "block" in user_types:
            for ob_name in self._scene.get_object_names(Cube):
                object_state[ob_name] = Object(ob_name, self._problem.user_type("block"), environment=self._environment)
        if "table" in user_types:
            object_state[self._scene.table.name] = Object(
                self._scene.table.name, self._problem.user_type("table"), environment=self._environment
            )
        if "location" in user_types:
            location_type = self._problem.user_type("location")
            for ob_name in self._scene.get_object_names(Location):
                object_state[ob_name] = Object(ob_name, location_type, environment=self._environment)
        if "target" in user_types:
            target_type = self._problem.user_type("target")
            for ob_name in self._scene.get_object_names(Target):
                object_state[ob_name] = Object(ob_name, target_type, environment=self._environment)

        return object_state

    def _ground_blocks_domain_preds(self, fluent_state: dict, object_state: dict[str, Object]) -> dict:
        """Perform predicate grounding in the blocks domain."""
        on_pred, clear_pred, north_pred, color_pred, material_pred = ground_cube_relations(self._scene)
        empty_pred, grasping_pred, lifted_pred, two_held_pred, three_held_pred = ground_gripper_relations(self._scene)
        above_loc_pred, north_loc_pred, at_pred, occ_pred, ob_above_pred, ob_north_pred, ob_south_pred = (
            ground_location_relations(self._scene)
        )
        prob_fluents = [f.name for f in self._problem.fluents]
        block_type = self._problem.user_type("block")
        if "on" in prob_fluents:
            for op in on_pred:
                o_fluent = self._problem.fluent(op[0])(*(object_state[op[1].name], object_state[op[2].name]))
                fluent_state[o_fluent] = True
        if "north-of" in prob_fluents:
            for np in north_pred:
                n_fluent = self._problem.fluent(np[0])(*(object_state[np[1].name], object_state[np[2].name]))
                fluent_state[n_fluent] = True
        if "clear" in prob_fluents:
            for cp in clear_pred:
                fluent = self._problem.fluent(cp[0])(*(object_state[cp[1].name],))
                fluent_state[fluent] = True
        if "small" in prob_fluents:
            for ob in object_state.values():
                if ob.type == block_type:
                    fluent = self._problem.fluent("small")(*(ob,))
                    fluent_state[fluent] = True
        if "gripper-full" in prob_fluents:
            fluent = self._problem.fluent("gripper-full")
            fluent_state[fluent] = not empty_pred
        if "gripper-lifted" in prob_fluents:
            fluent = self._problem.fluent("gripper-lifted")
            fluent_state[fluent] = lifted_pred
        if "grasping" in prob_fluents:
            for hp in grasping_pred:
                fluent = self._problem.fluent(hp[0])(*(object_state[hp[1].name],))
                fluent_state[fluent] = True
        if "two-held" in prob_fluents:
            fluent = self._problem.fluent("two-held")
            fluent_state[fluent] = two_held_pred
        if "three-held" in prob_fluents:
            fluent = self._problem.fluent("three-held")
            fluent_state[fluent] = three_held_pred
        if "loc-above" in prob_fluents:
            for la in above_loc_pred:
                fluent = self._problem.fluent(la[0])(*(object_state[la[1].name], object_state[la[2].name]))
                fluent_state[fluent] = True
        if "loc-north-of" in prob_fluents:
            for ln in north_loc_pred:
                fluent = self._problem.fluent(ln[0])(*(object_state[ln[1].name], object_state[ln[2].name]))
                fluent_state[fluent] = True
        if "at-loc" in prob_fluents:
            for al in at_pred:
                fluent = self._problem.fluent(al[0])(*(object_state[al[1].name], object_state[al[2].name]))
                fluent_state[fluent] = True
        if "occupied" in prob_fluents:
            for op in occ_pred:
                fluent = self._problem.fluent(op[0])(*(object_state[op[1].name],))
                fluent_state[fluent] = True
        if "obstructed-above" in prob_fluents:
            for oa in ob_above_pred:
                fluent = self._problem.fluent(oa[0])(*(object_state[oa[1].name],))
                fluent_state[fluent] = True
        if "obstructed-north" in prob_fluents:
            for on in ob_north_pred:
                fluent = self._problem.fluent(on[0])(*(object_state[on[1].name],))
                fluent_state[fluent] = True
        if "obstructed-south" in prob_fluents:
            for os in ob_south_pred:
                fluent = self._problem.fluent(os[0])(*(object_state[os[1].name],))
                fluent_state[fluent] = True
        for mp in material_pred:
            if mp[0] in prob_fluents:
                fluent = self._problem.fluent(mp[0])(*(object_state[mp[1].name],))
                fluent_state[fluent] = True
        for cp in color_pred:
            if cp[0] in prob_fluents:
                fluent = self._problem.fluent(cp[0])(*(object_state[cp[1].name],))
                fluent_state[fluent] = True

        return fluent_state

    def _ground_sponge_domain_objects(self, object_state: dict, user_types: list) -> dict[str, Object]:
        """Ground the objects in the sponge domain."""
        if "table" in user_types:
            object_state[self._scene.table.name] = Object(
                self._scene.table.name, self._problem.user_type("table"), environment=self._environment
            )
        if "location" in user_types:
            location_type = self._problem.user_type("location")
            for ob_name in self._scene.get_object_names(Location):
                object_state[ob_name] = Object(ob_name, location_type, environment=self._environment)
        if "sponge" in user_types:
            target_type = self._problem.user_type("sponge")
            for ob_name in self._scene.get_object_names(Sponge):
                object_state[ob_name] = Object(ob_name, target_type, environment=self._environment)
        if "spill" in user_types:
            target_type = self._problem.user_type("spill")
            for ob_name in self._scene.get_object_names(Spill):
                object_state[ob_name] = Object(ob_name, target_type, environment=self._environment)
        if "bin" in user_types:
            target_type = self._problem.user_type("bin")
            for ob_name in self._scene.get_object_names(Bin):
                object_state[ob_name] = Object(ob_name, target_type, environment=self._environment)
        if "can" in user_types:
            target_type = self._problem.user_type("can")
            for ob_name in self._scene.get_object_names(Can):
                object_state[ob_name] = Object(ob_name, target_type, environment=self._environment)
        if "plate" in user_types:
            target_type = self._problem.user_type("plate")
            for ob_name in self._scene.get_object_names(Plate):
                object_state[ob_name] = Object(ob_name, target_type, environment=self._environment)
        if "target" in user_types:
            target_type = self._problem.user_type("target")
            for ob_name in self._scene.get_object_names(Target):
                object_state[ob_name] = Object(ob_name, target_type, environment=self._environment)

        return object_state

    def _ground_sponge_domain_preds(self, fluent_state: dict, object_state: dict[str, Object]) -> dict:
        """Perform predicate grounding in the sponge domain."""
        # ; dynamic predicates
        # (on ?b - item ?s - surface) ; item b is on surface s
        # (grasping ?b - item) ; the gripper is closed around movable b
        # (obstructed ?b - surface) ; the surface is obstructed
        # (hover ?g - item ?s - surface); the surface being hovered over

        # ; pseudo-derived predicates
        # (gripper-full) ; the gripper is occupied <-> exists ?b. (grasping ?b)
        on_pred, obs_pred, material_pred, color_pred = ground_sponge_relations(self._scene)
        hover_pred, empty_pred, grasping_pred = ground_sponge_gripper_relations(self._scene)

        prob_fluents = [f.name for f in self._problem.fluents]
        if "on" in prob_fluents:
            for op in on_pred:
                fluent = self._problem.fluent(op[0])(*(object_state[op[1].name], object_state[op[2].name]))
                fluent_state[fluent] = True
        if "obstructed" in prob_fluents:
            for ob in obs_pred:
                fluent = self._problem.fluent(ob[0])(*(object_state[ob[1].name],))
                fluent_state[fluent] = True
        if "hover" in prob_fluents:
            for hp in hover_pred:
                fluent = self._problem.fluent(hp[0])(*(object_state[hp[1].name], object_state[hp[2].name]))
                fluent_state[fluent] = True
        if "grasping" in prob_fluents:
            for gp in grasping_pred:
                fluent = self._problem.fluent(gp[0])(*(object_state[gp[1].name],))
                fluent_state[fluent] = True
        if "gripper-full" in prob_fluents:
            fluent = self._problem.fluent("gripper-full")
            fluent_state[fluent] = not empty_pred
        for mp in material_pred:
            if mp[0] in prob_fluents:
                fluent = self._problem.fluent(mp[0])(*(object_state[mp[1].name],))
                fluent_state[fluent] = True
        for cp in color_pred:
            if cp[0] in prob_fluents:
                fluent = self._problem.fluent(cp[0])(*(object_state[cp[1].name],))
                fluent_state[fluent] = True

        return fluent_state


class PDDLParsingError(Exception):
    """An error that occurs when parsing a PDDL file."""

    pass
