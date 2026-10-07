import threading
import tkinter as tk
import traceback
from tkinter import ttk

from skillet.agents.tamp import PlanningAgent
from skillet.envs.skillet_env import SkilletEnv
from skillet.planning.abstract_model import AbstractModel
from skillet.scene import Cube
from skillet.scene.base import Scene


def run_gui(
    env: SkilletEnv, tamp_agent: PlanningAgent, abs_model: AbstractModel, domains: dict[str, str], scene: Scene
) -> None:
    root = tk.Tk()
    root.title("Blocks Demo")
    root.geometry("1000x600")

    style = ttk.Style()
    style.configure("TLabel", font=("Helvetica", 18))
    style.configure("TButton", font=("Helvetica", 18), padding=10)

    ttk.Label(root, text="Input PDDL goal").pack(padx=20, pady=(20, 5), anchor="w")
    entry = ttk.Entry(root, font=("Helvetica", 18))
    entry.pack(padx=20, pady=5, fill="x")
    entry.focus_set()

    button = ttk.Button(root, text="Run")
    button.pack(padx=20, pady=(20, 5))

    magnet_mode = False
    toggle_button = ttk.Button(root, text="Switch to Magnet")
    toggle_button.pack(padx=20, pady=(5, 5))

    def on_no_magnets() -> None:
        print("[INFO][Demo] Setting scene to 0 magnet blocks")
        for b in scene.get_object_from_type((Cube,)):
            b.material = "wooden"

    def on_two_magnets() -> None:
        print("[INFO][Demo] Setting scene to Red/Pink magnet blocks")
        for b in scene.get_object_from_type((Cube,)):
            if "red" in b.name or "pink" in b.name:
                b.material = "plastic"
            else:
                b.material = "wooden"

    def on_three_magnets() -> None:
        print("[INFO][Demo] Setting scene to Red/Pink/Blue magnet blocks")
        for b in scene.get_object_from_type((Cube,)):
            if "red" in b.name or "pink" in b.name or "blue" in b.name:
                b.material = "plastic"
            else:
                b.material = "wooden"

    action_frame = ttk.Frame(root)
    action_frame.pack(padx=20, pady=(5, 20))  # default anchor is centered

    action_buttons = [
        ttk.Button(action_frame, text="0 Magnet Blocks", command=on_no_magnets),
        ttk.Button(action_frame, text="Red/Pink Magnet Blocks", command=on_two_magnets),
        ttk.Button(action_frame, text="Red/Pink/Blue Magnet Blocks", command=on_three_magnets),
    ]
    for b in action_buttons:
        b.pack(side="left", padx=5)

    def set_controls_state(state: str) -> None:
        """Enable/disable everything except the Run button's own logic."""
        button.config(state=state)
        toggle_button.config(state=state)
        for b in action_buttons:
            b.config(state=state)

    def on_mode_changed(is_magnet: bool) -> None:
        """Called whenever the mode changes. Put your logic here."""
        print(f"[INFO][Demo] Domain is now {'Magnet' if is_magnet else 'Simple'}")
        abs_model._domain_file = domains["magnet"] if is_magnet else domains["simple"]

    def on_toggle() -> None:
        nonlocal magnet_mode
        magnet_mode = not magnet_mode
        toggle_button.config(text="Switch to Simple" if magnet_mode else "Switch to Magnet")
        on_mode_changed(magnet_mode)

    toggle_button.config(command=on_toggle)

    def finish() -> None:
        set_controls_state("normal")
        entry.focus_set()

    def worker(goal: str) -> None:
        try:
            if len(goal) == 0:
                raise ValueError("Goal cannot be empty")
            env.reset()
            if goal.startswith("("): # PDDL goal
                print(f"[INFO][Main] Executing PDDL goal: {goal}")
                tamp_agent.execute(env, task_pddl=goal)
            else:
                print(f"[INFO][Main] Executing NL goal: {goal}")
                tamp_agent.execute(env, task_nl=goal)
            print("[INFO][Main] finished experiment")
        except Exception:
            traceback.print_exc()
        finally:
            root.after(0, finish)  # re-enable controls once ready for the next goal

    def on_run(event=None) -> None:
        if str(button["state"]) == "disabled":
            return
        goal = entry.get().strip()
        if not goal:
            return
        set_controls_state("disabled")
        threading.Thread(target=worker, args=(goal,), daemon=True).start()

    button.config(command=on_run)
    root.bind("<Return>", on_run)  # Enter key triggers the button
    root.mainloop()
