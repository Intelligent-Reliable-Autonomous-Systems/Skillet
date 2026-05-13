"""Unified Planning utility functions."""

import itertools
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from unified_planning.engines.compilers import GrounderHelper
from unified_planning.model import Action, InstantaneousAction, OperatorKind, Problem, UPState
from unified_planning.model import Fluent as UPFluent
from unified_planning.model import Object as UPObject
from unified_planning.model import Parameter as UPParameter
from unified_planning.model import Type as UPType
from unified_planning.model.types import domain_item, domain_size
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import Bool as UPBool

if TYPE_CHECKING:
    from unified_planning.model.fnode import FNode

UPDictFluent = dict["FluentExpLike", bool]
"""A dictionary of assignments to Unified Planning fluents."""

UPDictObject = dict[str, "ObjectExpLike"]
"""A dictionary of of Unified Planning objects."""

UPListGoal = list["FluentExpLike"]


@dataclass
class ParsedUpProblem:
    """Parsed problem class for Unified Planning Problem."""

    fluents: UPDictFluent
    objects: UPDictObject
    goals: UPListGoal
    exclude_list = ["clear", "small", "loc_north_of", "loc_above", "occupied"]

    def __str__(self) -> str:
        print_str = "Abstract State:\n"
        for f, v in self.fluents.items():
            f = parse_value(str(f), v)
            if f.value in self.exclude_list:
                continue
            print_str += f"{f!s}\n"
        return print_str


@dataclass
class AbstractAction:
    action: str
    parameters: list[str]

    def __str__(self) -> str:
        print_str = f"{self.action}: |"
        for p in self.parameters:
            print_str += f" {p} |"
        return print_str


@dataclass
class AbstractValue:
    value: str
    parameters: list[str]

    def __str__(self) -> str:
        print_str = f"{self.value}: |"
        for p in self.parameters:
            print_str += f" {p} |"
        return print_str


@dataclass
class AbstractPlan:
    actions: list[AbstractAction]
    exclude_list = []

    def __str__(self) -> str:
        print_str = "Plan:\n"
        for ac in self.actions:
            if ac.action in self.exclude_list:
                continue
            print_str += f"{ac}\n"
        return print_str


@dataclass
class AbstractState:
    states: list[AbstractValue]
    exclude_list = ["clear", "small", "loc_north_of", "loc_above", "occupied"]

    def __str__(self) -> str:
        print_str = "Abstract State:\n"
        for s in self.states:
            if s.value in self.exclude_list:
                continue
            print_str += f"{s}\n"
        return print_str


@dataclass
class AbstractGoal:
    goals: list[AbstractValue]

    def __str__(self) -> str:
        print_str = "Abstract Goal:\n"
        for g in self.goals:
            print_str += f"{g}\n"
        return print_str


def parse_action(action_str: str) -> AbstractAction:
    """Parse an UP FNode into a string for an AbstractAction."""
    match = re.match(r"([\w-]+)\((.*)\)", action_str.strip())

    action = match.group(1).replace("-", "_")
    parameters = [p.strip() for p in match.group(2).split(",")]

    return AbstractAction(action=action, parameters=parameters)


def parse_value(value_str: str, v: bool | None = None) -> AbstractValue:
    """Parse a UP FNode into a string for an AbstractValue."""
    match = re.match(r"([\w-]+)\((.*)\)", value_str.strip())

    if match:
        value = match.group(1).replace("-", "_")
        parameters = [p.strip() for p in match.group(2).split(",")]
    else:
        value = value_str.strip().replace("-", "_")
        parameters = [v]

    return AbstractValue(value=value, parameters=parameters)


### Interfaces for UP FNode types
class ObjectExpLike(Protocol):
    """A UP Expression representing an object."""

    @property
    def node_type(self) -> Literal[OperatorKind.OBJECT_EXP]: ...

    @property
    def type(self) -> Literal[UPType]: ...

    def object(self) -> UPObject: ...

class FluentExpLike(Protocol):
    """A UP Expression representing a fluent."""
    @property
    def node_type(self) -> Literal[OperatorKind.FLUENT_EXP]: ...

    @property
    def args(self) -> list[ObjectExpLike]: ...

    @property
    def type(self) -> Literal[bool]: ...

    def fluent(self) -> UPFluent: ...

class NotFluentExpLike(Protocol):
    """A UP Expression representing a negated fluent."""
    @property
    def node_type(self) -> Literal[OperatorKind.NOT]: ...

    @property
    def type(self) -> Literal[bool]: ...

    @property
    def args(self) -> list[FluentExpLike]: ...

class ParameterExpLike(Protocol):
    """A UP Expression representing a parameter."""
    @property
    def node_type(self) -> Literal[OperatorKind.PARAM_EXP]: ...

    @property
    def type(self) -> Literal[UPType]: ...

    def parameter(self) -> UPParameter: ...

### State conversions
def up_state_to_dict(state: UPState | Sequence[FluentExpLike | NotFluentExpLike], closed_world: bool = False) -> dict[FluentExpLike, bool]:
    """Convert a UP State to a dictionary of fluent expressions and their boolean assignments.

    Args:
        state: The UP State to convert
        closed_world: Enforce the closed world assumption. If True, any fluent that is not in the state is assumed to be false.
    Returns:
        A dictionary of fluent expressions and their boolean assignments.
    """
    if isinstance(state, UPState):
        state._condense_state()
        state_dict = {k: v.bool_constant_value() for k, v in state._values.items()}
    else:
        dict_state = {}
        for fexp in state:
            if fexp.node_type == OperatorKind.NOT:
                dict_state[fexp.args[0]] = False
            else:
                dict_state[fexp] = True
        state_dict = dict_state
    if closed_world:
        all_fluents = enumerate_groundings_from_state(state_dict)
        for fluent in all_fluents:
            if fluent not in state_dict:
                state_dict[fluent] = False
    return state_dict

def dict_state_to_up_state(state: dict[FluentExpLike, bool], problem: Problem) -> UPState:
    """Convert a dictionary of fluent expressions and their boolean assignments to a UP State."""
    return UPState({k: UPBool(v) for k, v in state.items()}, problem)

def all_objects_from_state(state: dict[FluentExpLike, bool]) -> list[UPObject]:
    """Get all objects referenced in a dictionary of fluent expressions and their boolean assignments."""
    return list(set([oexp.object() for fexp in state.keys() for oexp in fexp.args]))

def enumerate_groundings_from_state(state: dict[FluentExpLike, bool]) -> Iterator[FluentExpLike]:
    """Enumerate all grounded fluent expressions using fluents and objects from the state.

    Args:
        state: A dictionary of fluent expressions and their boolean assignments.
    Returns:
        An iterator over all grounded fluent expressions.
    """
    all_fluents = set(state.keys())
    all_objects = set(all_objects_from_state(state))
    for grounded_fluent_exp in all_fluents:
        fluent = grounded_fluent_exp.fluent()
        if len(fluent.signature) == 0:
            yield fluent()
            continue
        compatible_objects = []
        for param in fluent.signature:
            param: UPParameter
            compatible_objects.append([obj for obj in all_objects if param.type.is_compatible(obj.type)])
        for objects in itertools.product(*compatible_objects):
            yield fluent(*objects)

def get_all_literal_expressions(problem: Problem, with_positive: bool = True, with_negated: bool = False) -> Iterator[FluentExpLike]:
    """Get all possible grounded literal assignments for a problem.

    Args:
        problem: The problem to get all possible grounded literal assignments from
        with_positive: Whether to include positive literal assignments
        with_negated: Whether to include negated literal assignments
    Returns:
        An iterator over all possible grounded literal assignments.
    """
    for fluent in problem.fluents:
        for params in get_fluent_possible_parameters(problem, fluent):
            literal = fluent(*params)
            if with_positive:
                yield literal
            if with_negated:
                negated = literal.Not()
                yield negated

def get_fluent_possible_parameters(problem: Problem, fluent: UPFluent) -> Iterator[tuple[ParameterExpLike, ...]]:
    """Get all possible parameter groundings for a fluent.

    Args:
        problem: The problem to get all possible parameter groundings from
        fluent: The fluent to get all possible parameter groundings for
    Returns:
        An iterator over all possible parameter groundings.
    """
    # if the action does not have parameters, it has only one possible parameter: the empty tuple
    if len(fluent.signature) == 0:
        return iter([tuple()])

    type_list: list[UPType] = [param.type for param in fluent.signature]

    # a list containing the list of object in the self._problem of the given type.
    # So, if the self._problem has 2 Locations l1 and l2, and 2 Robots r1 and r2, and
    # the fluent in-loc takes as parameters a Robot and a Location,
    # the variable state at this point will be the following:
    # type_list = [Robot, Location]
    # objects_list = [[r1, r2], [l1, l2]]
    # the product of *objects_list will be:
    # [(r1, l1), (r1, l2), (r2, l1), (r2,l2)]
    ground_size = 1
    domain_sizes = []
    for t in type_list:
        ds = domain_size(problem, t)
        domain_sizes.append(ds)
        ground_size *= ds
    items_list: list[list[FNode]] = []
    for size, type in zip(domain_sizes, type_list):
        items_list.append(
            [domain_item(problem, type, j) for j in range(size)]
        )

    return itertools.product(*items_list)

### Action utilities
def action_for_problem(action: ActionInstance, problem: Problem) -> ActionInstance:
    """UP actions and objects are associated with the problem instance.
    This function returns the action instance associated with the problem instance.
    """
    problem_action = problem.action(action.action.name)
    args = [problem.object(obj.object().name) for obj in action.actual_parameters]
    return ActionInstance(problem_action, args)

def ground_action_instances(problem) -> Iterator[ActionInstance]:
    """Ground all actions in the problem.

    Args:
        problem: The problem to ground actions from
        max_actions: The maximum number of actions to ground
    Returns:
        A list of grounded action instances.
    """
    grounded = []
    for action in problem.actions:
        domains = []
        for parameter in action.parameters:
            objs = list(problem.objects(parameter.type))
            np.random.shuffle(objs) # randomize to mitigate bias slightly
            domains.append(objs)
        if not domains:
            yield ActionInstance(action)
            continue
        for actual_parameters in itertools.product(*domains):
            yield ActionInstance(action, actual_parameters)
    return grounded

def get_possible_actions(problem: Problem, action: Action, partial_params: tuple[ObjectExpLike, ...]) -> Iterator[ActionInstance]:
    grounder = GrounderHelper(problem)
    partial_action = action.clone()
    for _ in partial_params:
        partial_action._parameters.popitem(last=False)
    params_iter = grounder.get_possible_parameters(partial_action)
    for grounded_params in params_iter:
        yield ActionInstance(action, (*partial_params, *grounded_params))

def sample_action_from_state(
        problem: Problem,
        state: UPState | dict[FluentExpLike, bool],
        action_name: str | None = None,
        applicable_only: bool = True,
        relax_terms: int = 0) -> ActionInstance:
    """Sample an action based on the state and present objects.

    Can sample either an applicable action or a random action grounding.
    Can relax some precondition terms to allow inapplicable actions that are closer to the preconditions.

    Args:
        problem: The problem to sample an action from
        state: The state to sample an action in
        action_name: Optionally, the name of the action to sample
        applicable_only: Whether to only sample applicable actions. If False, will sample a random grounding.
        relax_terms: The number of terms to relax in the preconditions. Only used if applicable_only is False.

    Returns:
        An action instance

    """
    if isinstance(state, UPState):
        state = up_state_to_dict(state)
    max_tries = 5
    for _ in range(max_tries):
        if action_name is None:
            up_action: InstantaneousAction = np.random.choice(problem.actions)
        else:
            up_action = problem.action(action_name)
        if not applicable_only:
            # sample a random grounding
            partial = {}
            break
        pre_fluents = []
        for precondition in up_action.preconditions:
            precondition: FNode
            if precondition.is_fluent_exp():
                pre_fluents.append((precondition, True))
            elif precondition.is_not():
                pre_fluents.append((precondition.args[0], False))
            elif precondition.is_and():
                for child in precondition.args:
                    if child.is_fluent_exp():
                        pre_fluents.append((child, True))
                    elif child.is_not():
                        pre_fluents.append((child.args[0], False))
                    elif child.is_or():
                        children = []
                        for or_child in child.args:
                            if or_child.is_fluent_exp():
                                children.append((or_child, True))
                            elif or_child.is_not():
                                children.append((or_child.args[0], False))
                        pre_fluents.append((('or', *children), True))
            elif precondition.is_or():
                children = []
                for child in precondition.args:
                    if child.is_fluent_exp():
                        children.append((child, True))
                    elif child.is_not():
                        children.append((child.args[0], False))
                pre_fluents.append((('or', *children), True))

        for _ in range(relax_terms):
            # randomly remove a precondition
            pre_fluents.pop(np.random.randint(len(pre_fluents)))
        partial = next(_unify_pattern(pre_fluents, state, problem), None)
        if partial is not None:
            break
        else:
            partial = {}
    # fill in any unbound parameters randomly
    return _uniform_action_grounding(problem, up_action, partial)

def _unify_pattern(pattern: list[FluentExpLike, bool], state: dict[FluentExpLike, bool], problem: Problem) -> Iterator[dict[UPParameter, UPObject]]:
    """Find all valid groundings of pattern against state.

    Args:
        pattern: Conjunction of (Literal, bool) — literal and required truth value
        state: dict[Atom, bool]
        problem: The problem to unify the pattern against
    Returns:
        A list of dict[str, str] — variable-to-object bindings

    """
    # Index state for fast lookup
    by_predicate = defaultdict(list[FluentExpLike])
    for atom, val in state.items():
        if val:
            by_predicate[atom.fluent().name].append((atom.args))

    # Find candidate atoms for each pattern literal
    candidate_lists: tuple[tuple[ParameterExpLike], list[ObjectExpLike]] = []
    for literal, value in pattern:
        if isinstance(literal, tuple) and literal[0] == 'or':
            dict_candidates = []
            args = []
            for child, cvalue in literal[1:]:
                child_candidates = by_predicate.get(child.fluent().name, [])
                if not cvalue:
                    child_candidates = list(set(get_fluent_possible_parameters(problem, child.fluent())) - set(child_candidates))
                for candidate in child_candidates:
                    dict_candidates.append({param: obj for param, obj in zip(child.args, candidate)})
                for arg in child.args:
                    if arg not in args:
                        args.append(arg)
            candidates = [tuple(dc.get(arg, None) for arg in args) for dc in dict_candidates]
        else:
            args = literal.args
            candidates = by_predicate.get(literal.fluent().name, [])
            if not value:
                all_bindings = get_fluent_possible_parameters(problem, literal.fluent())
                candidates = list(set(all_bindings) - set(candidates))
        if not candidates:
            return None # No way to satisfy this literal
        np.random.shuffle(candidates)
        candidate_lists.append((args, candidates))
    # satisfy (not) predicates first
    # least constrained variables first heuristic
    candidate_lists.sort(key=lambda x: len(x[1]))
    # Backtracking search
    def search(idx, bindings):
        if idx == len(candidate_lists):
            yield {param.parameter(): obj.object() for param, obj in bindings.items()}
            return
        var_args, candidates = candidate_lists[idx]
        for atom_args in candidates:
            new_bindings = dict(bindings)
            ok = True
            for var, obj in zip(var_args, atom_args):
                if obj is None:
                    continue
                if var in new_bindings:
                    if new_bindings[var] != obj:
                        ok = False
                        break
                else:
                    new_bindings[var] = obj
            if ok:
                yield from search(idx + 1, new_bindings)

    yield from search(0, {})

def _uniform_action_grounding(problem: Problem, action: InstantaneousAction, partial: dict[UPParameter, UPObject]) -> ActionInstance:
    grounding = dict(partial)
    for param in action.parameters:
        if param not in grounding:
            param: UPParameter
            # avoid grounding the same object to multiple parameters -- identifiability assumption
            compatible = [obj for obj in problem.all_objects if obj not in grounding.values() and param.type.is_compatible(obj.type)]
            if not compatible:
                return None
            grounding[param] = np.random.choice(compatible)
    return ActionInstance(action, [grounding.get(param, None) for param in action.parameters])

### Problem utilities
def initialize_problem_from_state(problem: Problem, state: dict[FluentExpLike, bool]) -> Problem:
    """Define a problem from an initial state."""
    new_problem = problem.clone()
    new_problem.add_objects(all_objects_from_state(state))
    for fexp, val in state.items():
        new_problem.set_initial_value(fexp, val)
    return new_problem

def reorder_problem_actions(problem: Problem, target: Problem):
    """Reorder the actions of a problem to match the actions of a target problem."""
    new_actions = []
    for target_action in target.actions:
        for action in problem.actions:
            root = action.name.split('__')[0]
            if root == target_action.name:
                new_actions.append(action)
    for action in problem.actions:
        if action not in new_actions:
            new_actions.append(action)
    problem._actions = new_actions

def flatten_preconditions(problem: Problem):
    for action in problem.actions:
        all_preconditions = []
        for precondition in action.preconditions:
            if precondition.is_and():
                all_preconditions.extend(precondition.args)
            else:
                all_preconditions.append(precondition)
        action._preconditions = all_preconditions

def sort_preconditions(problem: Problem):
    flatten_preconditions(problem)
    for action in problem.actions:
        all_preconditions = []
        for precondition in action.preconditions:
            all_preconditions.append(precondition)
        action._preconditions = sorted(all_preconditions, key=lambda x: str(x.args[0]) if x.is_not() else str(x))

def sort_effects(problem: Problem):
    for action in problem.actions:
        all_effects = []
        for effect in action.effects:
            all_effects.append(effect)
        action._effects = sorted(action.effects, key=lambda x: str(x))
