import threading
import tkinter as tk
import traceback
from tkinter import ttk

from skillet.agents.tamp import PlanningAgent
from skillet.envs.skillet_env import SkilletEnv
from skillet.planning.abstract_model import AbstractModel
from skillet.scene import Cube
from skillet.scene.base import Scene

# Tweak these to scale the whole UI up or down
FONT_FAMILY = "Helvetica"
TITLE_SIZE = 40
TEXT_SIZE = 34
BUTTON_SIZE = 30


def run_gui(
    env: SkilletEnv, tamp_agent: PlanningAgent, abs_model: AbstractModel, domains: dict[str, str], scene: Scene
) -> None:
    root = tk.Tk()
    root.title("Blocks Demo")

    # Fill the whole screen
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")
    root.minsize(1200, 800)

    style = ttk.Style()
    style.configure("TLabel", font=(FONT_FAMILY, TITLE_SIZE, "bold"))
    style.configure("TButton", font=(FONT_FAMILY, BUTTON_SIZE), padding=(30, 20))
    style.configure("Big.TButton", font=(FONT_FAMILY, BUTTON_SIZE + 6, "bold"), padding=(60, 25))
    style.configure("TEntry", padding=(15, 15))

    # Container so content has generous margins and stays centered
    container = ttk.Frame(root, padding=(60, 50))
    container.pack(fill="both", expand=True)

    ttk.Label(container, text="Input PDDL goal").pack(pady=(0, 15), anchor="w")
    entry = ttk.Entry(container, font=(FONT_FAMILY, TEXT_SIZE))
    entry.pack(pady=10, fill="x", ipady=10)
    entry.focus_set()

    button = ttk.Button(container, text="Run", style="Big.TButton")
    button.pack(pady=(40, 15))

    magnet_mode = False
    toggle_button = ttk.Button(container, text="Switch to Magnet")
    toggle_button.pack(pady=(15, 15))

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

    action_frame = ttk.Frame(container)
    action_frame.pack(pady=(30, 20))  # default anchor is centered

    action_specs = [
        ("0 Magnet Blocks", on_no_magnets),
        ("Red/Pink Magnet Blocks", on_two_magnets),
        ("Red/Pink/Blue Magnet Blocks", on_three_magnets),
    ]
    action_buttons: list[ttk.Button] = []
    selected_action = 0  # index of the currently selected magnet configuration

    def refresh_action_buttons(enabled: bool = True) -> None:
        """Grey out the selected magnet button; the others follow `enabled`."""
        for i, b in enumerate(action_buttons):
            if i == selected_action or not enabled:
                b.config(state="disabled")
            else:
                b.config(state="normal")

    def select_action(index: int) -> None:
        nonlocal selected_action
        selected_action = index
        action_specs[index][1]()
        refresh_action_buttons()

    for i, (text, _) in enumerate(action_specs):
        b = ttk.Button(action_frame, text=text, command=lambda i=i: select_action(i))
        b.pack(side="left", padx=15)
        action_buttons.append(b)

    # 0 magnet blocks is selected on start up
    select_action(0)

    def set_controls_state(state: str) -> None:
        """Enable/disable everything except the Run button's own logic."""
        button.config(state=state)
        toggle_button.config(state=state)
        # Keep the selected magnet button greyed out even when re-enabling
        refresh_action_buttons(enabled=(state == "normal"))

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
            if goal.startswith("("):  # PDDL goal
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
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
    root.bind("<F11>", lambda e: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
    root.mainloop()
