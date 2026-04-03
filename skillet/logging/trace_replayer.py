import cv2
import numpy as np

# Parse the PDDL trace
trace = """(:objects green red - block table0 - table)
(:state (small red) (small green) (handempty ) (clear table0) (clear red) (clear green) (on red table0) (on green table0))
(:action (pick-block red table0))
(:delta (holding red) (not (handempty )) (not (clear red)) (not (on red table0)))
(:action (place-block red green))
(:goal-reached)
(:delta (handempty ) (clear red) (on red green) (not (holding red)) (not (clear green))))"""


def parse_state_list(line: str) -> dict[str, str]:
    """Parse state predicates from a PDDL line."""
    match_start = line.find("(")
    if match_start == -1:
        return {}
    content = line[match_start + 1 : -1]
    predicates = []
    depth = 0
    current = ""
    for char in content:
        if char == "(":
            depth += 1
            if depth > 1:
                current += char
        elif char == ")":
            depth -= 1
            if depth > 0:
                current += char
            elif current.strip():
                predicates.append(current.strip())
                current = ""
        elif char == " " and depth == 1:
            if current.strip():
                predicates.append(current.strip())
                current = ""
        else:
            current += char

    if current.strip():
        predicates.append(current.strip())

    state = {}
    for pred in predicates:
        parts = pred.split()
        key = parts[0]
        value = " ".join(parts[1:]) if len(parts) > 1 else True
        state[key] = value

    return state


def apply_delta(current_state: dict, delta: list[str]) -> dict[str, str]:
    """Apply delta changes to the current state."""
    new_state = current_state.copy()
    for key in delta:
        if key.startswith("not"):
            # Parse "not (predicate args)" or similar
            real_key = key.replace("not", "").strip()
            if real_key in new_state:
                del new_state[real_key]
        else:
            new_state[key] = delta[key]
    return new_state


def parse_trace(trace_text: str) -> list[str]:
    """Parse entire PDDL trace into steps."""
    lines = [l.strip() for l in trace_text.split("\n") if l.strip()]
    states = []
    current_state = None
    step = 0

    for i, line in enumerate(lines):
        if line.startswith("(:state"):
            current_state = parse_state_list(line)
            states.append(
                {"step": step, "action": None, "state": current_state.copy(), "delta": None, "goal_reached": False}
            )
            step += 1

        elif line.startswith("(:action"):
            action = line[8:-1].strip()
            states.append(
                {"step": step, "action": action, "state": current_state.copy(), "delta": None, "goal_reached": False}
            )
            step += 1

        elif line.startswith("(:delta"):
            delta = parse_state_list(line)
            if states:
                states[-1]["delta"] = delta
            current_state = apply_delta(current_state, delta)
            states.append(
                {"step": step, "action": None, "state": current_state.copy(), "delta": None, "goal_reached": False}
            )
            step += 1

        elif line.startswith("(:goal-reached"):
            if states:
                states[-1]["goal_reached"] = True

    return states


def draw_world_state(state: dict, img_width: int = 800, img_height: int = 600) -> np.ndarray:
    """Draw the PDDL world state predicates on an image."""
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240

    y_offset = 30

    # Title
    cv2.putText(img, "World State Predicates:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    y_offset += 50

    # Display all predicates
    for key, value in sorted(state.items()):
        text = f"  ✓ {key}" if value else f"  ✓ ({key} {value})"

        cv2.putText(img, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 1)
        y_offset += 30

    return img


def main():
    states = parse_trace(trace)
    current_step = 0

    print("PDDL World State Visualizer")
    print("=" * 40)
    print(f"Total states: {len(states)}")
    print("\nControls:")
    print("  SPACE / RIGHT ARROW - Next step")
    print("  LEFT ARROW - Previous step")
    print("  ESC - Exit")
    print("=" * 40)

    while True:
        step_data = states[current_step]

        # Draw world predicates
        img = draw_world_state(step_data["state"])

        # Add step and action information at top
        y_offset = img.shape[0] - 100
        cv2.rectangle(img, (0, y_offset), (img.shape[1], img.shape[0]), (230, 230, 230), -1)

        y_offset += 25
        cv2.putText(img, f"Step: {step_data['step']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        y_offset += 35

        if step_data["action"]:
            cv2.putText(
                img, f"Action: {step_data['action']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2
            )
        elif step_data["goal_reached"]:
            cv2.putText(img, "GOAL REACHED!", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 150, 0), 3)
        else:
            cv2.putText(img, "Initial State", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        cv2.imshow("PDDL World State", img)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        if key == ord(" ") or key == 83:  # SPACE or RIGHT ARROW
            if current_step < len(states) - 1:
                current_step += 1
        elif key == 81 and current_step > 0:  # LEFT ARROW
            current_step -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
