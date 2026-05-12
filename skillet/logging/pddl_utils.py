import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PDDLProblem:
    name: str
    domain: str
    objects: dict[str, str]
    init: list[tuple]
    goal: list[tuple]

    def __repr__(self):
        return (
            f"PDDLProblem(name={self.name!r}, domain={self.domain!r},\n"
            f"  objects={self.objects},\n"
            f"  init={self.init},\n"
            f"  goal={self.goal})"
        )


@dataclass
class TraceStep:
    """Helper class for the PDDL Trace."""

    action: tuple | None = None
    state_before: dict | None = None
    delta_add: list[tuple] | None = None
    delta_del: list[tuple] | None = None
    state_after: dict | None = None
    goal_reached: bool = False

    def __repr__(self):
        lines = [f"  action      : {self.action}"]
        lines.append(f"  +added      : {self.delta_add}")
        lines.append(f"  -removed    : {self.delta_del}")
        lines.append(f"  goal_reached: {self.goal_reached}")
        return "TraceStep(\n" + "\n".join(lines) + "\n)"


@dataclass
class Trace:
    """Class for storing a PDDL trace."""

    objects: dict[str, str]
    initial_state: dict
    steps: list[TraceStep] = field(default_factory=list)

    def __repr__(self):
        parts = [f"Trace(objects={self.objects}, steps=["]
        for i, s in enumerate(self.steps):
            parts.append(f"  [{i}] {s}")
        parts.append("])")
        return "\n".join(parts)


def _parse_predicate(s: str) -> tuple:
    """Parse a single predicate string like '(on red table0)' or '(gripper-full )'.

    Returns:
        a tuple, e.g. ('on', 'red', 'table0') or ('gripper-full',).

    """
    s = s.strip().lstrip("(").rstrip(")")
    tokens = s.split()
    return tuple(tokens)


def _parse_predicate_list(block: str) -> list[tuple]:
    """Extract all top-level predicates from a block.

    May contain multiple parenthesised expressions, including negations.

    Returns:
        a list of tuples for positive predicates and
    ('not', inner_tuple) for negations.

    """
    results = []
    depth = 0
    current = []

    for ch in block:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
            if depth == 0:
                expr = "".join(current).strip()
                current = []
                # negation: (not (pred args...))
                if re.match(r"\(\s*not\s*\(", expr, re.I):
                    inner = re.search(r"\(\s*not\s*(\(.*\))\s*\)", expr, re.I)
                    if inner:
                        results.append(("not", _parse_predicate(inner.group(1))))
                elif expr:
                    results.append(_parse_predicate(expr))
        elif depth > 0:
            current.append(ch)

    return results


def _state_key(pred: tuple) -> str:
    return " ".join(pred)


def _apply_delta(state: dict, adds: list[tuple], dels: list[tuple]) -> dict:
    new_state = dict(state)
    for p in dels:
        new_state[_state_key(p)] = False
    for p in adds:
        new_state[_state_key(p)] = True
    return new_state


def parse_pddl_problem(text: str) -> PDDLProblem:
    """Parse a PDDL problem definition into a PDDLProblem dataclass."""
    # Problem name
    name_m = re.search(r"\(define\s+\(problem\s+(\S+)\)", text)
    name = name_m.group(1) if name_m else "unknown"

    # Domain
    domain_m = re.search(r"\(:domain\s+(\S+)\)", text)
    domain = domain_m.group(1) if domain_m else "unknown"

    # Objects  -- typed: "red green blue - block  table0 - table"
    objects: dict[str, str] = {}
    obj_m = re.search(r"\(:objects(.*?)\)", text, re.S)
    if obj_m:
        obj_text = obj_m.group(1).strip()
        # Split on '-' type separators
        segments = re.split(r"\s+-\s+", obj_text)
        current_type = None
        for i in range(len(segments) - 1, -1, -1):
            seg = segments[i].split()
            if i == len(segments) - 1:
                # last segment is a pure type name
                current_type = seg[0]
            else:
                # seg contains names, and the last token of next segment is the type
                for tok in seg:
                    objects[tok] = current_type
                current_type = seg[0] if i > 0 else current_type

        # Simpler re-parse: walk left-to-right
        objects = {}
        parts = re.split(r"\s+", obj_text)
        names_buf = []
        i = 0
        while i < len(parts):
            if parts[i] == "-":
                typ = parts[i + 1]
                for n in names_buf:
                    objects[n] = typ
                names_buf = []
                i += 2
            else:
                if parts[i]:
                    names_buf.append(parts[i])
                i += 1

    # Init
    init_m = re.search(r"\(:init(.*?)\)(?=\s*\(:goal)", text, re.S)
    init_preds = []
    if init_m:
        init_preds = _parse_predicate_list(init_m.group(1))

    # Goal -- may be wrapped in (and ...)
    goal_preds = []
    goal_start = text.find("(:goal")
    if goal_start != -1:
        depth = 0
        goal_raw = ""
        for ch in text[goal_start:]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            goal_raw += ch
        goal_block = re.sub(r"^\(:goal\s*", "", goal_raw)
        and_m = re.match(r"\s*\(\s*and\s+(.*)", goal_block, re.S)
        if and_m:
            goal_block = and_m.group(1)
        goal_preds = _parse_predicate_list(goal_block)

    return PDDLProblem(
        name=name,
        domain=domain,
        objects=objects,
        init=init_preds,
        goal=goal_preds,
    )


def parse_trace(text: str) -> Trace:
    """Parse a trace string into a Trace dataclass with a list of TraceSteps.

    Expected trace format (lines/tokens):
      (:objects ...)
      (:state ...)
      (:action ...)
      (:delta ...)
      (:goal-reached)   -- optional
      (:action ...)
      (:delta ...)
      ...

    """
    lines = text.strip().splitlines()
    full = " ".join(l.strip() for l in lines)

    # Parse objects
    objects: dict[str, str] = {}
    obj_m = re.search(r"\(:objects(.*?)\)(?=\s*\()", full)
    if obj_m:
        obj_text = obj_m.group(1).strip()
        parts = re.split(r"\s+", obj_text)
        names_buf = []
        i = 0
        while i < len(parts):
            if parts[i] == "-":
                typ = parts[i + 1]
                for n in names_buf:
                    objects[n] = typ
                names_buf = []
                i += 2
            else:
                if parts[i]:
                    names_buf.append(parts[i])
                i += 1

    # Parse initial state
    state_m = re.search(r"\(:state(.*?)\)(?=\s*\(:)", full)
    initial_state: dict[str, bool] = {}
    if state_m:
        for pred in _parse_predicate_list(state_m.group(1)):
            if pred[0] == "not":
                initial_state[_state_key(pred[1])] = False
            else:
                initial_state[_state_key(pred)] = True

    blocks = []
    depth = 0
    start = None
    i = 0
    while i < len(full):
        if (
            (full[i] == "(" and full[i:].startswith("(:action"))
            or (full[i] == "(" and full[i:].startswith("(:delta"))
            or (full[i] == "(" and full[i:].startswith("(:goal-reached"))
        ) and depth == 0:
            start = i
        if full[i] == "(":
            depth += 1
        elif full[i] == ")":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(full[start : i + 1])
                start = None
        i += 1

    # Build trace steps
    steps: list[TraceStep] = []
    current_action = None
    current_state = dict(initial_state)
    pending_goal = False

    for block in blocks:
        if block.startswith("(:action"):
            inner = re.match(r"\(:action\s+\(?(.*?)\)?\s*\)", block, re.S)
            if inner:
                tokens = inner.group(1).strip().split()
                current_action = tuple(tokens)

        elif block.startswith("(:goal-reached"):
            pending_goal = True

        elif block.startswith("(:delta"):
            inner = re.match(r"\(:delta(.*)\)", block, re.S)
            adds, dels = [], []
            if inner:
                preds = _parse_predicate_list(inner.group(1))
                for p in preds:
                    if p[0] == "not":
                        dels.append(p[1])
                    else:
                        adds.append(p)

            state_before = dict(current_state)
            state_after = _apply_delta(current_state, adds, dels)
            current_state = state_after

            step = TraceStep(
                action=current_action,
                state_before=state_before,
                delta_add=adds,
                delta_del=dels,
                state_after=state_after,
                goal_reached=pending_goal,
            )
            steps.append(step)
            current_action = None
            pending_goal = False

    return Trace(objects=objects, initial_state=initial_state, steps=steps)


def print_trace_reconstruction(trace: Trace) -> None:
    """Print the trace for testing."""
    print("=" * 60)
    print("TRACE RECONSTRUCTION")
    print("=" * 60)
    print(f"\nObjects: {trace.objects}")
    print("\nInitial state:")
    for k, v in trace.initial_state.items():
        if v:
            print(f"  {k}")

    for i, step in enumerate(trace.steps):
        print(f"\n── Step {i + 1} {'─' * 46}")
        print(f"  Action : {step.action}")
        print(f"  Added  : {[' '.join(p) for p in step.delta_add]}")
        print(f"  Removed: {[' '.join(p) for p in step.delta_del]}")
        print("  State after:")
        for k, v in step.state_after.items():
            if v:
                print(f"    {k}")
        if step.goal_reached:
            print("  *** GOAL REACHED ***")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\nParsing Problem 1...")
    with open("skillet/scene/abstract/assets/3-block-table.problem.pddl", "r") as f:
        PROBLEM_1 = f.read()
    p1 = parse_pddl_problem(PROBLEM_1)
    print(p1)

    with open("skillet/scene/abstract/assets/3-block-table-restack.problem.pddl", "r") as f:
        PROBLEM_2 = f.read()
    print("\nParsing Problem 2...")
    p2 = parse_pddl_problem(PROBLEM_2)
    print(p2)

    print("\nParsing and reconstructing trace...")
    with open("data/test/20260403_140731/exp_0/trace.txt", "r") as f:
        TRACE = f.read()
    trace = parse_trace(TRACE)
    print_trace_reconstruction(trace)
