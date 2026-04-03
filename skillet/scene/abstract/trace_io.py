"""PDDL trace file parser and writer."""

import re
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Literal

from unified_planning.exceptions import UPValueError
from unified_planning.model import Fluent, Object, Problem
from unified_planning.model import Type as UPType
from unified_planning.plans import ActionInstance

from skillet.scene.abstract.up_utils import FluentExpLike, UPDictState

PREDICATE_RE = re.compile(r"\(([a-zA-Z0-9_-]+)(?:[ \t]+([a-zA-Z0-9_-]+))*\)")

STATE_PREFIX = "(:state"
ACTION_PREFIX = "(:action"
DELTA_PREFIX = "(:delta"
OBSERVE_PREFIX = "(:observe"
OBJECTS_PREFIX = "(:objects"
GOAL_REACHED_STR = "(:goal-reached)"
INAPPLICABLE_STR = "(:inapplicable)"


class PDDLTraceIO:
    """PDDL trace file parser and writer."""

    def __init__(self, domain: Problem, prefer_delta: bool = False) -> None:
        """Initialize the PDDL trace file parser and writer.

        Args:
            domain: The PDDL domain.
            prefer_delta: Whether to prefer delta traces over full state traces.

        """
        self.domain: Problem = domain
        self.prefer_delta: bool = prefer_delta
        self._obj_name_to_obj: dict[str, Object] = {}

    def write_trace_file(
        self,
        plan_file: Path,
        states: list[UPDictState],
        actions: list[ActionInstance],
        executions: list[Literal["applicable", "goal_reached", "inapplicable"]] | None = None,
    ) -> None:
        """Write a PDDL trace file from a list of states, actions, and executions.

        Args:
            plan_file: The path to the PDDL trace file.
            states: The list of abstract states.
            actions: The list of grounded action instances.
            executions: The list of execution outcomes for each action.

        """
        all_objects = set()
        object_types = defaultdict(set)
        for state in states:
            for pred in state:
                for arg in pred.args:
                    all_objects.add(arg.object())
                    object_types[arg.object().type.name].add(arg.object().name)
        for action in actions:
            for arg in action.actual_parameters:
                all_objects.add(arg.object())
                object_types[arg.object().type.name].add(arg.object().name)
        if executions is None:
            executions = []
        with Path(plan_file).open("w") as f:
            type_strs = [" ".join(names) + " - " + type for type, names in object_types.items()]
            f.write(f"(:objects {' '.join(type_strs)})\n")
            prev_state = None
            for state, action, execution in zip_longest(states, actions, executions):
                if state is not None:
                    if self.prefer_delta and prev_state is not None:
                        delta = {x: state[x] for x in state if x not in prev_state or state[x] != prev_state[x]}
                        delta.update({x: False for x in prev_state if x not in state})
                        pred_strs = []
                        for pred, value in delta.items():
                            pred_str = f'({pred.fluent().name} {" ".join([str(arg) for arg in pred.args])})'
                            if not value:
                                pred_str = f"(not {pred_str})"
                            pred_strs.append(pred_str)
                        f.write(f"(:delta {' '.join(pred_strs)})\n")
                    else:
                        pred_strs = []
                        for pred, value in state.items():
                            pred_str = f'({pred.fluent().name} {" ".join([str(arg) for arg in pred.args])})'
                            if not value:
                                pred_str = f"(not {pred_str})"
                            pred_strs.append(pred_str)
                        f.write(f"(:state {' '.join(pred_strs)})\n")
                    prev_state = state
                if action is not None:
                    action_str = f'({action.action.name} {" ".join([str(arg) for arg in action.actual_parameters])})'
                    f.write(f"(:action {action_str})\n")

                if execution is not None:
                    if execution == "goal_reached":
                        f.write(f"{GOAL_REACHED_STR}\n")
                    elif execution == "inapplicable":
                        f.write(f"{INAPPLICABLE_STR}\n")

    def parse_trace_file(self, trace_file: Path) -> tuple[list[UPDictState], list[ActionInstance], list[str]]:
        """Parse a PDDL trace file into a list of states, actions, and executions.

        Args:
            trace_file: The path to the PDDL trace file.

        Returns:
            A tuple of lists of states, actions, and executions.

        """
        with Path(trace_file).open("r") as f:
            return self._parse_plan(f.read())

    def _parse_plan(self, plan_text: str) -> tuple[list[dict[FluentExpLike, bool]], list[ActionInstance], list[str]]:
        self._domain_instance = self.domain.clone()
        self._obj_name_to_obj = {}

        cur_state = None
        step = 0
        states = []
        actions = []
        executions = []
        for line in plan_text.splitlines():
            line = line.strip()
            if line.startswith(";"):
                line = line[: line.index(";")]
            if len(line) == 0:
                continue

            if line.startswith(OBJECTS_PREFIX):
                self._parse_objects(line[len(OBJECTS_PREFIX) : -len(")")])
                continue
            # A state uses closed world assumption; overrides all previous predicates, including not stated ones
            if line.startswith("(:state"):
                cur_state = self._parse_predicates(line[len(STATE_PREFIX) : -len(")")])
                while len(states) <= step:
                    # fill in the missing states with the previous state
                    if len(states) == 0:
                        states.append(None)
                    else:
                        states.append(states[-1])

                states[step] = cur_state
                continue

            # An observation can be a partial state, modifying a subset of the predicates
            if line.startswith(OBSERVE_PREFIX):
                cur_observation = self._parse_predicates(line[len(OBSERVE_PREFIX) : -len(")")])
                if cur_state is None:
                    cur_state = {}
                next_state = {**cur_state, **cur_observation}
                while len(states) <= step:
                    # don't fill in the missing states; nothing was observed
                    states.append(None)
                states[step] = next_state
                cur_state = next_state

                continue

            # A delta specifies only the predicates that changed. It is a strict observation
            if line.startswith(DELTA_PREFIX):
                assignments = self._parse_predicates(line[len(DELTA_PREFIX) : -len(")")])
                if cur_state is None:
                    raise ValueError("Full state or observation must be specified before deltas")
                next_state = {**cur_state, **assignments}
                while len(states) <= step:
                    # fill in the missing states with the previous state
                    if len(states) == 0:
                        states.append(None)
                    else:
                        states.append(states[-1])
                states[step] = next_state
                cur_state = next_state

                continue
            if line.startswith(ACTION_PREFIX):
                grounded_action = self._parse_action(line[len(ACTION_PREFIX) : -len(")")])
                actions.append(grounded_action)
                step += 1
                continue
            if line.startswith(GOAL_REACHED_STR):
                while len(executions) <= step:
                    executions.append("applicable")
                executions[step] = "goal_reached"
                continue
            if line.startswith(INAPPLICABLE_STR):
                while len(executions) < step:
                    executions.append("applicable")
                executions[step - 1] = "inapplicable"
                continue

        while len(executions) < len(actions):
            executions.append("applicable")
        return states, actions, executions

    def _parse_predicates(self, state_text: str):
        text = state_text.strip()
        predicates: dict[FluentExpLike, bool] = {}
        while len(text) > 0:
            if not text.startswith("("):
                break
            predicate, is_positive, remainder = self._parse_predicate(text)
            predicates[predicate] = is_positive
            text = remainder.strip()
        return predicates

    def _parse_action(self, action_text: str) -> ActionInstance:
        """Parse a grounded PDDL action string into name and argument list.

        Returns:
            (name, args)

        Examples:
            "(move robot loc1 loc2)"  -> ("move", ["robot", "loc1", "loc2"])
            "(pick-up robot box1)"    -> ("pick-up", ["robot", "box1"])
            "(halt)"                  -> ("halt", [])

        """
        action_text = action_text.strip()
        if not (action_text.startswith("(") and action_text.endswith(")")):
            raise ValueError(f"Invalid action: {action_text!r}")

        tokens = action_text[1:-1].split()
        if not tokens:
            raise ValueError("Empty action")

        action = next((a for a in self.domain.actions if a.name == tokens[0]), None)
        if action is None:
            raise ValueError(f"No action found for {tokens[0]}")

        objects = [
            self._match_object(token, param.type) for token, param in zip(tokens[1:], action.parameters, strict=True)
        ]
        return ActionInstance(action, objects)

    def _parse_predicate(self, predicate: str) -> tuple[FluentExpLike, bool, str]:
        """Parse a PDDL predicate string, handling negation.

        Returns:
            (name, args, is_positive)

        Examples:
            "(at robot loc1)"           -> ("at", ["robot", "loc1"], True)
            "(not (at robot loc1))"     -> ("at", ["robot", "loc1"], False)
            "(handempty)"               -> ("handempty", [], True)
            "(not (handempty))"         -> ("handempty", [], False)

        """
        s = predicate.strip()
        if not s.startswith("("):
            raise ValueError(f"Expected '(' at start of: {s!r}")

        def find_matching_close(s: str, open_pos: int) -> int:
            """Return index of the closing paren matching the one at open_pos."""
            depth = 0
            for i in range(open_pos, len(s)):
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                    if depth == 0:
                        return i
            raise ValueError(f"Unmatched '(' in: {s!r}")

        close = find_matching_close(s, 0)
        remainder = s[close + 1 :]
        inner = s[1:close].strip()

        if inner.startswith("not"):
            rest = inner[3:].strip()
            if not rest.startswith("("):
                raise ValueError(f"Expected predicate inside (not ...): {predicate!r}")
            inner_close = find_matching_close(rest, 0)
            inner = rest[1:inner_close].strip()
            positive = False
        else:
            positive = True

        tokens = inner.split()
        if not tokens:
            raise ValueError("Empty predicate")

        try:
            matched_fluent: Fluent = self.domain.fluent(tokens[0])
            if matched_fluent.arity != len(tokens[1:]):
                raise ValueError
        except ValueError:
            raise ValueError(f"No fluent found for {tokens[0]} with signature {tokens[1:]}")
        objects = [self._match_object(token, param.type) for token, param in zip(tokens[1:], matched_fluent.signature)]
        fluent_exp: FluentExpLike = get_fluent_exp(matched_fluent, objects)

        return fluent_exp, positive, remainder

    def _parse_objects(self, objects_text: str) -> None:
        """Parse a PDDL typed list string into a dict mapping names to types.

        Examples:
            "obj1 obj2 - mytype1 obj3 - mytype2"
            -> {'obj1': 'mytype1', 'obj2': 'mytype1', 'obj3': 'mytype2'}

        """
        result = {}
        tokens = objects_text.split()
        current_names = []

        i = 0
        while i < len(tokens):
            if tokens[i] == "-":
                type_name = tokens[i + 1]
                for name in current_names:
                    result[name] = type_name
                current_names = []
                i += 2
            else:
                current_names.append(tokens[i])
                i += 1

        for name in current_names:
            result[name] = "object"

        for name, type_name in result.items():
            up_type = self._domain_instance.user_type(type_name)
            self._match_object(name, up_type)

    def _match_object(self, token: str, type: UPType) -> Object:
        if token in self._obj_name_to_obj:
            return self._obj_name_to_obj[token]
        try:
            obj = self._domain_instance.object(token)
        except UPValueError:
            obj = Object(token, type)
        self._obj_name_to_obj[token] = obj
        return obj
