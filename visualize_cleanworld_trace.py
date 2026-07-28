"""Visualize cleanworld (sponge) trace files as spatial diagrams.

A trace is a sequence of ``(:state ...)`` / ``(:delta ...)`` snapshots separated
by ``(:action ...)`` steps, produced by the conditional-repair simulator (see the
``experiments/conditional-repair/cleanworld/.../traces/*.trace`` files). Traces
are parsed with the project's own ``conditional_repair.trace.PDDLPlanParser``
(which needs the domain, so it resolves fluents/actions and reports each action's
execution status). Each state is rendered as a support tree:

  * table          -> a wide brown bar at the bottom, spanning its subtree
  * targets        -> flat disks (the circular placement spots)
  * plates         -> flat ovals with a rim; cross-hatched when ``dirty``
  * cans           -> cylinders, drawn on their side when not ``upright``
  * sponges        -> rounded rectangles, with a blue band when ``wet``
  * spills/messes  -> irregular blobs
  * grasped items  -> hung under a gripper claw to the right of the scene
  * inapplicable   -> the (unchanged) state with a small red X to its left

Structure is derived entirely from ``on``: each object sits in the cell above its
support, and siblings are laid out side by side. ``obstructed`` is drawn as a
badge on a surface's top edge, and is outlined in red when it disagrees with the
``on`` facts (obstructed but empty, or occupied but not obstructed) -- a common
symptom worth seeing while debugging repairs.

Traces declare only a subset of the objects they mention, so this script first
infers a type for every object it sees -- from the ``(:objects ...)`` line where
that is specific, and otherwise from the object's name and its type predicates
(``supportable``, ``graspable``, ``wipeable``, ...) -- and seeds the domain with
them. Without this the parser falls back to the fluent signature's declared type
(``on ?b - object ...``) and rejects the resulting expression as ill-typed.

Reference domain: experiments/conditional-repair/cleanworld/
                  conditional-sponge.domain.pddl

Example:
    python scripts/visualize_cleanworld_trace.py path/to/foo.trace
    python scripts/visualize_cleanworld_trace.py path/to/foo.trace -o foo.png --cols 3
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    Ellipse,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

# Make the conditional_repair package importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conditional_repair.trace import PDDLPlanParser  # noqa: E402
from unified_planning.io import PDDLReader  # noqa: E402
from unified_planning.model import Object as UPObject  # noqa: E402

DEFAULT_DOMAIN = _REPO_ROOT / "experiments" / "conditional-repair" / "cleanworld" / "conditional-sponge.domain.pddl"

# Named colors we recognize inside object names, mapped to a fill color.
COLOR_WORDS = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "yellow": "#e6c200",
    "cyan": "#17becf",
    "pink": "#e377c2",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "magenta": "#d62796",
    "brown": "#8c564b",
    "gray": "#7f7f7f",
    "grey": "#7f7f7f",
    "black": "#222222",
    "white": "#eeeeee",
}
TABLE_COLOR = "#6b4f3a"
FALLBACK_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

# Every (name arg ...) group, ignoring the (:state ...) / (not ...) wrappers.
PRED_RE = re.compile(r"\(([a-zA-Z][a-zA-Z0-9_-]*)((?:[ \t]+[a-zA-Z0-9_-]+)*)[ \t]*\)")

# Name fragments that imply a role, when the trace does not declare the type.
NAME_ROLES = [
    ("table", "table"),
    ("bin", "bin"),
    ("plate", "plate"),
    ("circle", "target"),
    ("target", "target"),
    ("sponge", "sponge"),
    ("cloth", "sponge"),
    ("rag", "sponge"),
    ("can", "can"),
    ("bottle", "can"),
    ("cup", "can"),
    ("spill", "spill"),
    ("scribble", "spill"),
    ("stain", "spill"),
    ("crumb", "spill"),
    ("mess", "spill"),
]
SURFACE_ROLES = {"table", "plate", "target", "bin"}
CONCRETE_TYPES = {"table", "plate", "target", "bin", "sponge", "can", "spill"}
# Cell width, in shape-widths: object names are long, so siblings need the room.
SPACING = 1.7


# --------------------------------------------------------------------------- #
# Object typing
# --------------------------------------------------------------------------- #
def _parse_typed_list(text):
    """``"a b - plate c - can"`` -> ``{'a': 'plate', 'b': 'plate', 'c': 'can'}``."""
    result, pending, tokens = {}, [], text.split()
    i = 0
    while i < len(tokens):
        if tokens[i] == "-" and i + 1 < len(tokens):
            for name in pending:
                result[name] = tokens[i + 1]
            pending = []
            i += 2
        else:
            pending.append(tokens[i])
            i += 1
    return result


def scan_trace_text(text):
    """Collect what the raw trace says about its objects.

    Returns ``(declared, attrs, supports)``: the ``(:objects ...)`` declarations,
    each object's unary predicates, and the objects used as a support in ``on``.
    """
    declared, attrs, supports = {}, defaultdict(set), set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("(:objects"):
            declared.update(_parse_typed_list(line[len("(:objects") : -1]))
            continue
        if not line.startswith(("(:state", "(:delta", "(:observe")):
            continue
        for match in PRED_RE.finditer(line):
            name, args = match.group(1), match.group(2).split()
            if name == "on" and len(args) == 2:
                supports.add(args[1])
            elif len(args) == 1:
                attrs[args[0]].add(name)
    return declared, attrs, supports


def infer_object_types(text):
    """Give every object in the trace a concrete domain type.

    Declarations win when they are specific; otherwise the name decides, and
    failing that the object's own predicates do.
    """
    declared, attrs, supports = scan_trace_text(text)
    types = {}
    for name in set(declared) | set(attrs) | supports:
        if declared.get(name) in CONCRETE_TYPES:
            types[name] = declared[name]
            continue
        role = next((r for frag, r in NAME_ROLES if frag in name.lower()), None)
        if role is None:
            a = attrs.get(name, set())
            if "wipeable" in a:
                role = "spill"
            elif "deformable" in a:
                role = "sponge"
            elif "supportable" in a or name in supports:
                role = "target"
            elif "graspable" in a:
                role = "can"
            else:
                role = declared.get(name, "item")
        types[name] = role
    return types


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _state_to_preds(state):
    """Convert a UP state dict {FluentExp: bool} into ``[["name", arg, ...], ...]``.

    Only positive facts are kept (the drawing code is closed-world).
    """
    preds = []
    for fexp, value in state.items():
        if not value:
            continue
        preds.append([fexp.fluent().name] + [str(a) for a in fexp.args])
    return preds


def _action_label(action):
    """Human-readable grounded-action string, e.g. ``pick blue_can purple_circle``."""
    args = " ".join(str(p) for p in action.actual_parameters)
    return f"{action.action.name} {args}".strip()


def load_trace(path, domain_path):
    """Parse a trace with the project's PDDLPlanParser and build a panel list.

    Returns ``(obj_types, panels)`` where each panel is a tuple
    ``(title, preds, inapplicable)``. Action titles carry the action's index in
    the plan. An ``inapplicable`` action still shows its (unchanged) state, but
    the panel is flagged so a small separator X is drawn to mark the failure.
    """
    domain = PDDLReader().parse_problem(str(domain_path))
    text = Path(path).read_text()
    for name, type_name in sorted(infer_object_types(text).items()):
        if not domain.has_object(name):
            domain.add_object(UPObject(name, domain.user_type(type_name)))
    trace = PDDLPlanParser(domain).parse_trace_file(path)

    obj_types = {obj.name: obj.type.name for obj in trace.objects}

    panels = [("initial state", _state_to_preds(trace.states[0]), False)]
    for i, action in enumerate(trace.actions):
        label = _action_label(action)
        preds = _state_to_preds(trace.states[i + 1])
        execution = trace.executions[i]
        if execution == "inapplicable":
            panels.append((f"[{i}] inapplicable: {label}", preds, True))
        else:
            title = f"[{i}] after {label}"
            if execution == "goal_reached":
                title += "  (goal reached)"
            panels.append((title, preds, False))
    return obj_types, panels


# --------------------------------------------------------------------------- #
# Scene semantics
# --------------------------------------------------------------------------- #
def object_color(name):
    """Pick a fill color for an object from its name."""
    for part in name.lower().replace("-", "_").split("_"):
        if part in COLOR_WORDS:
            return COLOR_WORDS[part]
    idx = sum(ord(c) for c in name) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def short_label(name):
    """Drop a trailing ``_0``-style index; keep the rest of the name."""
    parts = name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def collect_state(preds):
    """Extract the drawable facts from a state's predicate list."""
    on = {}  # item -> surface it rests on
    attrs = defaultdict(set)  # object -> {'wet', 'dirty', 'upright', ...}
    grasping = set()
    for p in preds:
        if p[0] == "on":
            on[p[1]] = p[2]
        elif p[0] == "grasping":
            grasping.add(p[1])
        elif len(p) == 2:
            # every remaining unary fluent is an attribute of its argument
            attrs[p[1]].add(p[0])
    return on, attrs, grasping


def object_roles(obj_types, on, attrs):
    """Map every mentioned object to the shape it should be drawn as."""
    names = set(obj_types) | set(on) | set(on.values()) | set(attrs)
    roles = {}
    for name in names:
        type_name = obj_types.get(name)
        if type_name in CONCRETE_TYPES:
            roles[name] = type_name
        else:
            roles[name] = "target" if name in set(on.values()) else "item"
    return roles


def build_tree(on, roles, grasping):
    """Return ``(coords, spans, roots, floating)`` for the support tree.

    ``coords[obj] = (center_x, depth)``; ``spans[obj] = (x_start, x_end)``.
    Roots are supports that rest on nothing (normally the table). ``floating``
    holds objects that no root supports and the gripper does not hold -- either
    nothing supports them, or noisy ``on`` facts put them in a support cycle.
    """
    children = defaultdict(list)
    for item, support in on.items():
        children[support].append(item)

    supported = set(on)
    mentioned = set(on) | set(on.values()) | set(roles)
    roots = sorted(obj for obj in mentioned if obj not in supported and obj not in grasping and obj in children)
    # Prefer tables first so the scene reads bottom-up from the table.
    roots.sort(key=lambda o: (roles.get(o) != "table", o))

    coords, spans = {}, {}

    def place(node, x0, depth):
        # Skipping already-placed kids guards against `on` cycles, which noisy
        # traces do contain; the placeholder marks `node` as visited.
        kids = sorted(k for k in children.get(node, []) if k not in coords)
        coords[node] = None
        width = 0
        for kid in kids:
            width += place(kid, x0 + width, depth + 1)
        width = max(width, 1)
        coords[node] = (x0 + width / 2.0, depth)
        spans[node] = (x0, x0 + width)
        return width

    x = 0.0
    for root in roots:
        x += place(root, x, 0) + 0.4

    # Lay out in unit cells above, then widen the cells so labels have room.
    coords = {o: (cx * SPACING, d) for o, (cx, d) in coords.items()}
    spans = {o: (x0 * SPACING, x1 * SPACING) for o, (x0, x1) in spans.items()}

    floating = sorted(obj for obj in mentioned if obj not in coords and obj not in grasping)
    return coords, spans, roots, floating


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _label(ax, cx, y, name):
    """Name tag next to a shape; names are far wider than the shapes they name."""
    ax.text(cx, y, short_label(name), ha="center", va="center", fontsize=6, color="#444444", zorder=6)


def draw_table(ax, x0, x1, y, name):
    ax.add_patch(
        Rectangle(
            (x0 + 0.05, y + 0.18),
            (x1 - x0) - 0.1,
            0.34,
            facecolor=TABLE_COLOR,
            edgecolor="black",
            linewidth=1.0,
            zorder=2,
        )
    )
    ax.text((x0 + x1) / 2.0, y + 0.35, short_label(name), ha="center", va="center", fontsize=6, color="white", zorder=6)


def draw_target(ax, cx, y, name, color, dirty):
    ax.add_patch(
        Ellipse(
            (cx, y + 0.16),
            0.78,
            0.2,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            hatch="xx" if dirty else None,
            zorder=3,
        )
    )
    _label(ax, cx, y - 0.1, name)


def draw_plate(ax, cx, y, name, color, dirty):
    ax.add_patch(Ellipse((cx, y + 0.2), 0.92, 0.3, facecolor=color, edgecolor="#555555", linewidth=1.2, zorder=3))
    ax.add_patch(
        Ellipse(
            (cx, y + 0.22),
            0.62,
            0.18,
            facecolor=color,
            edgecolor="#888888",
            linewidth=0.7,
            hatch="xxx" if dirty else None,
            zorder=4,
        )
    )
    _label(ax, cx, y - 0.08, name)


def draw_bin(ax, cx, y, name, color):
    ax.add_patch(
        Polygon(
            [(cx - 0.36, y + 0.9), (cx + 0.36, y + 0.9), (cx + 0.28, y + 0.08), (cx - 0.28, y + 0.08)],
            closed=True,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
    )
    _label(ax, cx, y + 0.98, name)


def draw_can(ax, cx, y, name, color, upright):
    if upright:
        ax.add_patch(
            Rectangle((cx - 0.17, y + 0.1), 0.34, 0.6, facecolor=color, edgecolor="black", linewidth=1.0, zorder=3)
        )
        ax.add_patch(Ellipse((cx, y + 0.7), 0.34, 0.12, facecolor=color, edgecolor="black", linewidth=1.0, zorder=4))
    else:
        ax.add_patch(
            Rectangle((cx - 0.32, y + 0.12), 0.6, 0.34, facecolor=color, edgecolor="#d62728", linewidth=1.4, zorder=3)
        )
        ax.add_patch(
            Ellipse((cx + 0.28, y + 0.29), 0.12, 0.34, facecolor=color, edgecolor="#d62728", linewidth=1.4, zorder=4)
        )
        ax.text(cx - 0.02, y + 0.29, "tipped", ha="center", va="center", fontsize=5, color="white", zorder=6)
    _label(ax, cx, y + 0.9, name)


def draw_sponge(ax, cx, y, name, color, wet):
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.3, y + 0.14),
            0.6,
            0.46,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
    )
    if wet:
        ax.add_patch(
            Rectangle((cx - 0.3, y + 0.14), 0.6, 0.14, facecolor="#4fa8e0", edgecolor="none", alpha=0.85, zorder=4)
        )
        ax.add_patch(Circle((cx + 0.36, y + 0.5), 0.05, facecolor="#4fa8e0", edgecolor="none", zorder=4))
        ax.text(cx, y + 0.21, "wet", ha="center", va="center", fontsize=5, color="white", zorder=6)
    _label(ax, cx, y + 0.78, name)


def draw_spill(ax, cx, y, name, color):
    blob = [
        (cx - 0.34, y + 0.12),
        (cx - 0.14, y + 0.32),
        (cx + 0.06, y + 0.14),
        (cx + 0.3, y + 0.3),
        (cx + 0.36, y + 0.1),
        (cx + 0.12, y + 0.02),
        (cx - 0.1, y + 0.16),
        (cx - 0.3, y + 0.0),
    ]
    ax.add_patch(Polygon(blob, closed=True, facecolor=color, edgecolor=color, linewidth=1.0, alpha=0.85, zorder=5))
    ax.text(cx, y + 0.42, short_label(name), ha="center", va="bottom", fontsize=5.5, color="#444444", zorder=6)


def draw_object(ax, name, cx, y, role, attrs):
    color = object_color(name)
    a = attrs.get(name, set())
    if role == "plate":
        draw_plate(ax, cx, y, name, color, "dirty" in a)
    elif role == "target":
        draw_target(ax, cx, y, name, color, "dirty" in a)
    elif role == "bin":
        draw_bin(ax, cx, y, name, color)
    elif role == "can":
        draw_can(ax, cx, y, name, color, "upright" in a)
    elif role == "sponge":
        draw_sponge(ax, cx, y, name, color, "wet" in a)
    elif role == "spill":
        draw_spill(ax, cx, y, name, color)
    else:
        ax.add_patch(
            Rectangle((cx - 0.3, y + 0.12), 0.6, 0.5, facecolor=color, edgecolor="black", linewidth=1.0, zorder=3)
        )
        _label(ax, cx, y + 0.8, name)


def draw_state(ax, title, preds, obj_types, inapplicable=False):
    on, attrs, grasping = collect_state(preds)
    roles = object_roles(obj_types, on, attrs)
    coords, spans, roots, floating = build_tree(on, roles, grasping)

    max_depth = max((d for _, d in coords.values()), default=0)
    max_x = max((x1 for _, x1 in spans.values()), default=1.0)

    for name, (cx, depth) in sorted(coords.items()):
        y = depth
        role = roles.get(name, "item")
        if role == "table":
            draw_table(ax, spans[name][0], spans[name][1], y, name)
        else:
            draw_object(ax, name, cx, y, role, attrs)

    # obstructed badges: a small square on the surface's top-right. Red when the
    # flag disagrees with `on` (obstructed & empty, or occupied & unobstructed).
    occupied = set(on.values())
    for name, (cx, depth) in coords.items():
        if roles.get(name) not in SURFACE_ROLES:
            continue
        obstructed = "obstructed" in attrs.get(name, set())
        holds = name in occupied
        if not obstructed and not holds:
            continue
        conflict = obstructed != holds
        ax.add_patch(
            Rectangle(
                (cx + 0.3, depth + 0.02),
                0.12,
                0.12,
                facecolor="#333333" if obstructed else "none",
                edgecolor="#d62728" if conflict else "#333333",
                linewidth=1.4 if conflict else 0.6,
                zorder=7,
            )
        )

    # Gripper: grasped items hang under a claw to the right of the scene.
    held = sorted(grasping)
    gx = max_x + 1.0
    if held:
        top = max_depth + 0.9
        ax.plot([gx, gx], [top + 0.45, top + 0.1], color="black", linewidth=2, zorder=3)
        ax.text(gx, top + 0.55, "gripper", ha="center", va="bottom", fontsize=6, color="#555555")
        for i, name in enumerate(held):
            draw_object(ax, name, gx, top - 1 - i, roles.get(name, "item"), attrs)

    # Floating objects: parked in a row under the scene, so a long list widens
    # the panel instead of squashing it. Ones that do have an `on` fact are in a
    # support cycle, so name the support that failed to anchor them.
    floor = -1.5
    if floating:
        ax.text(-0.2, floor + 0.3, "no support", ha="right", va="center", fontsize=5.5, color="#999999")
        for i, name in enumerate(floating):
            cx = (i + 0.5) * SPACING
            draw_object(ax, name, cx, floor, roles.get(name, "item"), attrs)
            if name in on:
                ax.text(
                    cx,
                    floor - 0.34,
                    f"on {short_label(on[name])}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="#d62728",
                    zorder=6,
                )

    # Inapplicable: a small X to the left, separating this from the prior state.
    left = -0.3
    if inapplicable:
        xc, yc, s = -0.85, (max_depth + 1) / 2.0, 0.3
        ax.plot([xc - s, xc + s], [yc - s, yc + s], color="#d62728", linewidth=2.2, zorder=7)
        ax.plot([xc - s, xc + s], [yc + s, yc - s], color="#d62728", linewidth=2.2, zorder=7)
        left = -1.5

    right = max(gx + (0.8 if held else -0.6), len(floating) * SPACING)
    low = floor - 0.6 if floating else -0.3
    ax.set_xlim(left, right)
    ax.set_ylim(low, max_depth + (2.2 if held else 1.3))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8, color="#d62728" if inapplicable else "black")


def visualize(path, output=None, cols=None, num=-1, start=0, domain=None):
    obj_types, panels = load_trace(path, domain or DEFAULT_DOMAIN)
    if not panels:
        raise SystemExit(f"No states found in {path}")

    if num == -1:
        panels = panels[start:]
    else:
        panels = panels[start : start + num]
    n = len(panels)
    ncols = cols or min(n, 4)
    nrows = (n + ncols - 1) // ncols
    # Scenes are wide and short (one row per support level), so panels are too.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.0 * nrows), squeeze=False)
    for i, (title, preds, inapplicable) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        draw_state(ax, title, preds, obj_types, inapplicable=inapplicable)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(Path(path).name, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if output is None:
        output = Path(path).with_suffix(".png")
    fig.savefig(output, dpi=130, bbox_inches="tight")
    print(f"Wrote {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, help="path to a .trace file")
    ap.add_argument("--output", "-o", type=Path, default=None, help="output image path (default: <trace>.png)")
    ap.add_argument("--domain", type=Path, default=None, help=f"domain PDDL (default: {DEFAULT_DOMAIN.name})")
    ap.add_argument("--num", "-n", type=int, default=-1, help="number of panels to visualize (default: all)")
    ap.add_argument("--start", "-s", type=int, default=0, help="start panel index (default: 0)")
    ap.add_argument("--cols", type=int, default=None, help="number of panels per row")
    args = ap.parse_args()
    visualize(args.trace, args.output, args.cols, args.num, args.start, args.domain)


if __name__ == "__main__":
    main()
