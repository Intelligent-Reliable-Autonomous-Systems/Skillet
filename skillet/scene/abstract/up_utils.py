"""Unified Planning utility functions."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from unified_planning.model import Fluent as UPFluent
from unified_planning.model import Object as UPObject
from unified_planning.model import OperatorKind, Problem
from unified_planning.model import State as UPState
from unified_planning.model import Type as UPType
from unified_planning.shortcuts import Bool as UPBool

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

    def __str__(self) -> str:
        print_str = "Abstract State:\n"
        for f in self.fluents:
            print_str += f"{parse_value(str(f))!s}\n"
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

    def __str__(self) -> str:
        print_str = "Plan:\n"
        for ac in self.actions:
            print_str += f"{ac}\n"
        return print_str


@dataclass
class AbstractState:
    states: list[AbstractValue]

    def __str__(self) -> str:
        print_str = "Abstract State:\n"
        for s in self.states:
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


def parse_value(value_str: str) -> AbstractValue:
    """Parse a UP FNode into a string for an AbstractValue."""
    match = re.match(r"([\w-]+)\((.*)\)", value_str.strip())

    if match:
        value = match.group(1).replace("-", "_")
        parameters = [p.strip() for p in match.group(2).split(",")]
    else:
        value = value_str.strip().replace("-", "_")
        parameters = []

    return AbstractValue(value=value, parameters=parameters)


class ObjectExpLike(Protocol):
    """Unified Planning FNode specialized to object expressions."""

    @property
    def node_type(self) -> Literal[OperatorKind.OBJECT_EXP]:
        """The type of the node must be OBJECT_EXP."""
        ...

    @property
    def type(self) -> Literal[UPType]:
        """The Unified Planning type of the object."""
        ...

    def object(self) -> UPObject:
        """Get the Unified Planning object from the object expression."""
        ...


class FluentExpLike(Protocol):
    """Unified Planning FNode specialized to fluent expressions."""

    @property
    def node_type(self) -> Literal[OperatorKind.FLUENT_EXP]:
        """The type of the node must be FLUENT_EXP."""
        ...

    @property
    def args(self) -> list[ObjectExpLike]:
        """The arguments of the fluent expression."""
        ...

    @property
    def type(self) -> Literal[bool]:
        """The Unified Planning type of the fluent expression."""
        ...

    def fluent(self) -> UPFluent:
        """Get the Unified Planning fluent from the fluent expression."""
        ...


class NotFluentExpLike(Protocol):
    """Unified Planning FNode specialized to the negation of fluent expressions."""

    @property
    def node_type(self) -> Literal[OperatorKind.NOT]:
        """The type of the node must be NOT."""
        ...

    @property
    def type(self) -> Literal[bool]:
        """The Unified Planning type of the not fluent expression."""
        ...

    @property
    def args(self) -> list[FluentExpLike]:
        """The arguments of the not fluent expression should be a single element list."""
        ...


def up_state_to_dict(state: UPState | Sequence[FluentExpLike | NotFluentExpLike]) -> dict[FluentExpLike, bool]:
    """Convert a Unified Planning state to a dictionary of assignments to fluents."""
    if isinstance(state, UPState):
        # State is a dict of assignments to fluents
        state._condense_state()
        return {k: v.bool_constant_value() for k, v in state._values.items()}

    # State is a sequence of (my-pred arg1 arg2 ...) and (not (my-pred arg1 arg2 ...)) fluents.
    dict_state = {}
    for fexp in state:
        if fexp.node_type == OperatorKind.NOT:
            dict_state[fexp.args[0]] = False
        else:
            dict_state[fexp] = True
    return dict_state


def dict_state_to_up_state(state: dict[FluentExpLike, bool], problem: Problem) -> UPState:
    """Convert a dictionary of assignments to fluents to a Unified Planning state."""
    return UPState({k: UPBool(v) for k, v in state.items()}, problem)
