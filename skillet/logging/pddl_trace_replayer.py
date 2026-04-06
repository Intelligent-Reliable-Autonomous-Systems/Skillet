"""PDDL Trace Visualizer.

SPACE       - advance to next step
Q / ESC     - quit
"""

import argparse

import cv2
import numpy as np
from utils import filled_rect, outline_rect, pill, put, put_c, tw

from skillet.logging.pddl_utils import Trace, parse_trace

BG = (23, 17, 15)
PANEL = (38, 28, 24)
PANEL_EDGE = (65, 48, 40)
TITLE_FG = (255, 210, 200)
LABEL_FG = (160, 125, 110)
ADD_BG = (30, 55, 20)
ADD_FG = (100, 210, 80)
DEL_BG = (20, 20, 60)
DEL_FG = (80, 80, 220)
PRED_BG = (50, 35, 30)
PRED_FG = (240, 195, 180)
ACTION_BG = (80, 45, 30)
ACTION_FG = (255, 175, 100)
GOAL_BG = (40, 55, 20)
GOAL_FG = (130, 230, 60)
DIM_FG = (80, 60, 55)
STEP_FG = (120, 85, 70)
HINT_FG = (95, 65, 55)
PANEL_EDGE2 = (90, 60, 48)


W, H = 840, 640


def render_frame(trace: Trace, step_idx: int) -> np.ndarray:
    """Render the Trace frame in the visualizer.

    step_idx == -1    - initial state, showing first action upcoming
    step_idx == k     - state after step k, showing step k+1 upcoming

    Args:
        trace: Trace parsed from text file
        step_idx: current step

    Returns:
        np.ndarray of image to display

    """

    def _state_key(pred: tuple) -> str:
        return " ".join(pred)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = BG

    total = len(trace.steps)

    # Header
    cv2.rectangle(img, (0, 0), (W, 52), PANEL, -1)
    cv2.line(img, (0, 52), (W, 52), PANEL_EDGE2, 1)
    put(img, "PDDL Trace Visualizer", 18, 34, TITLE_FG, scale=0.62, thick=1)
    label = "Initial state" if step_idx < 0 else f"After step {step_idx + 1} of {total}"
    put(img, label, W - tw(label, 0.42) - 18, 34, STEP_FG, scale=0.42)

    # Object row
    ox = 18
    put(img, "Objects:", ox, 76, LABEL_FG, scale=0.38)
    ox += tw("Objects:", 0.38) + 10
    for name, typ in trace.objects.items():
        ox = pill(img, f"{name}:{typ}", ox, 79, PRED_FG, PRED_BG, scale=0.36)

    cv2.line(img, (18, 92), (W - 18, 92), PANEL_EDGE, 1)

    # Figure out what predicates to display
    if step_idx < 0:
        state = trace.initial_state
        next_act = trace.steps[0].action if total > 0 else None
        d_add, d_del = [], []
        goal_done = False
    else:
        step = trace.steps[step_idx]
        state = step.state_after
        next_act = trace.steps[step_idx + 1].action if step_idx + 1 < total else None
        d_add = step.delta_add
        d_del = step.delta_del
        goal_done = step.goal_reached

    active = sorted(k for k, v in state.items() if v)

    # Display the world state in the left panel
    lx, ly, lw, lh = 18, 102, 490, 468
    filled_rect(img, lx, ly, lw, lh, PANEL, radius=8)
    outline_rect(img, lx, ly, lw, lh, PANEL_EDGE, radius=8)

    put(img, "World State", lx + 14, ly + 26, LABEL_FG, scale=0.40)
    cv2.line(img, (lx + 1, ly + 36), (lx + lw - 1, ly + 36), PANEL_EDGE, 1)

    row_y = ly + 58
    row_h = 27

    # Added predicates (display in green)
    for pred_str in active:
        if row_y > ly + lh - 20:
            break
        is_new = pred_str in {_state_key(a) for a in d_add}
        fg = ADD_FG if is_new else PRED_FG
        bg = ADD_BG if is_new else PRED_BG
        marker = "+  " if is_new else "   "
        filled_rect(img, lx + 12, row_y - 17, lw - 24, 23, bg, radius=4)
        put(img, marker + pred_str, lx + 20, row_y, fg, scale=0.40)
        row_y += row_h

    # Removed predicates (display in red)
    for d in d_del:
        if row_y > ly + lh - 20:
            break
        pred_str = _state_key(d)
        filled_rect(img, lx + 12, row_y - 17, lw - 24, 23, DEL_BG, radius=4)
        put(img, "-  " + pred_str, lx + 20, row_y, DEL_FG, scale=0.40)
        row_y += row_h

    # Right panel (deltas)
    rx, ry, rw, rh = 524, 102, W - 524 - 18, 468
    filled_rect(img, rx, ry, rw, rh, PANEL, radius=8)
    outline_rect(img, rx, ry, rw, rh, PANEL_EDGE, radius=8)

    # Next action
    put(img, "Next Action", rx + 14, ry + 26, LABEL_FG, scale=0.40)
    cv2.line(img, (rx + 1, ry + 36), (rx + rw - 1, ry + 36), PANEL_EDGE, 1)

    if next_act:
        name_s = next_act[0]
        args_s = "  ".join(next_act[1:])
        filled_rect(img, rx + 10, ry + 46, rw - 20, 42, ACTION_BG, radius=6)
        put_c(img, name_s, rx + rw // 2, ry + 65, ACTION_FG, scale=0.50, thick=1)
        if args_s:
            put_c(img, args_s, rx + rw // 2, ry + 82, LABEL_FG, scale=0.36)
    else:
        put_c(img, "(end of trace)", rx + rw // 2, ry + 68, DIM_FG, scale=0.40)

    # Most recent delta (change in action)
    cv2.line(img, (rx + 10, ry + 104), (rx + rw - 10, ry + 104), PANEL_EDGE, 1)
    put(img, "Last Delta", rx + 14, ry + 124, LABEL_FG, scale=0.40)

    dy = ry + 148
    if step_idx < 0:
        put(img, "(no previous action)", rx + 14, dy, DIM_FG, scale=0.37)
    else:
        for a in d_add:
            if dy > ry + rh - 20:
                break
            put(img, "+  " + " ".join(a), rx + 14, dy, ADD_FG, scale=0.38)
            dy += 24
        for d in d_del:
            if dy > ry + rh - 20:
                break
            put(img, "-  " + " ".join(d), rx + 14, dy, DEL_FG, scale=0.38)
            dy += 24

    # Display goal
    if goal_done:
        gy = ry + rh - 52
        filled_rect(img, rx + 10, gy, rw - 20, 38, GOAL_BG, radius=6)
        outline_rect(img, rx + 10, gy, rw - 20, 38, GOAL_FG, radius=6)
        put_c(img, "GOAL REACHED", rx + rw // 2, gy + 25, GOAL_FG, scale=0.52, thick=1)

    # Footer for instructions
    hint = "SPACE  next step          Q / ESC  quit"
    put_c(img, hint, W // 2, H - 12, HINT_FG, scale=0.36)

    return img


def run(trace: Trace) -> None:
    """Run the PDDL Trace window.

    Args:
        trace: Trace object.

    """
    win = "PDDL Trace"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    step_idx = -1
    total = len(trace.steps)
    alive = True

    while alive:
        frame = render_frame(trace, step_idx)
        cv2.imshow(win, frame)

        key = cv2.waitKey(50) & 0xFF

        # window closed via ✕
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        if key == ord(" "):
            if step_idx < total - 1:
                step_idx += 1
            # if already at last step, space does nothing (already showing final state)
        elif key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, default="data/test/20260403_140731/exp_0/trace.txt")
    args = parser.parse_args()
    with open(args.fname, "r") as f:
        trace_txt = f.read()

    run(parse_trace(trace_txt))
