import sys
import termios
import threading
import tty
from enum import IntEnum
from typing import Any

import torch

from skillet.core.env import Environment
from skillet.core.skill import Skill, SkillStatusCodes


class KeyboardListener:
    """A non-blocking keyboard input listener that runs in a background thread.

    Register callbacks for specific keys or a catch-all handler.
    """

    def __init__(self):
        self._thread = None
        self._running = False
        self._key_callbacks = {}
        self._default_callback = None

    def on_key(self, key: str) -> None:
        """Decorator to register a callback for a specific key."""

        def decorator(func):
            self._key_callbacks[key] = func
            return func

        return decorator

    def on_any_key(self, func):
        """Register a catch-all callback that receives every keypress."""
        self._default_callback = func
        return func

    def _read_key(self) -> str:
        """Read a single raw keypress from stdin (Unix)."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            # Handle escape sequences (arrows, F-keys, etc.)
            if ch == "\x1b":
                ch += sys.stdin.read(2)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def _listen(self):
        while self._running:
            try:
                key = self._read_key()
            except Exception:
                break

            # Dispatch to specific handler first, then default
            handler = self._key_callbacks.get(key) or self._default_callback
            if handler:
                handler(key)

    def start(self):
        """Start listening in a background daemon thread (non-blocking)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None

    def wait(self):
        """Block until the listener stops (useful to keep main thread alive)."""
        if self._thread:
            self._thread.join()


HOME_TCP_CART = [0.25, 0.0, 0.3, -180, 0, 90, 0.0]


class ExpStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    QUIT = 0
    """The experiment should exit."""
    RUNNING = 1
    """The experiment is running."""
    STOP = 2
    """The experiment is stopped."""
    HOME = 3
    """The experiment is resetting to safe position."""
    RESUME = 4
    """The experiment is resuming."""


class SkilletModerator:
    def __init__(self, device: str = "cuda"):
        self._listener = KeyboardListener()
        self._device = device
        self._home = None
        self._home_pos = torch.as_tensor(HOME_TCP_CART, device=self._device).unsqueeze(0)
        self._status = ExpStatusCodes.RUNNING
        self._exp_paused = False

        @self._listener.on_key("q")
        def quit_handler(key):
            self._status = ExpStatusCodes.QUIT
            self._intervention = True

        @self._listener.on_key("x")
        def stop_handler(key):
            self._status = ExpStatusCodes.STOP
            self._intervention = True

        @self._listener.on_key("h")
        def home_handler(key):
            self._status = ExpStatusCodes.HOME
            self._intervention = True

        @self._listener.on_key(" ")
        def resume_handler(key):
            self._status = ExpStatusCodes.RUNNING
            self._intervention = True

        self._listener.start()

    def poll(self, env: Environment, skill) -> Any:
        if self._intervention:
            if self._status == ExpStatusCodes.QUIT:
                print("[INFO][MODERATOR] Quitting.")
            elif self._status == ExpStatusCodes.HOME:
                print("[INFO][MODERATOR] Returning to home position.")
                env.step(self._home_pos, action_spec=skill.action_spec)
            elif self._status == ExpStatusCodes.STOP:
                print("[INFO][MODERATOR] Stopping the robot.")
                skill.status = SkillStatusCodes.FAILED
            elif self._status == ExpStatusCodes.RESUME:
                print("[INFO][MODERATOR] Resuming the robot experiment.")
                self._status = ExpStatusCodes.RUNNING

    def run_skill(self, env: Environment, skill: Skill, args: list[str]) -> None:
        """Run the skill in the environment."""
        obs = env.get_observation(skill.obs_spec)
        skill.initiate(obs, args)
        skill_done = skill.is_terminated(env.get_observation(skill.obs_spec))

        terminated = False
        while not skill_done or terminated or self._exp_paused:
            self.poll()
            if self._status != ExpStatusCodes.RUNNING:
                continue
            # Get the next action with the low-level observation
            action = skill.get_action(env.get_observation(skill.obs_spec))
            # Take a step in the environment
            _, _, term, trunc, _ = env.step(action, action_spec=skill.action_spec)
            terminated = term | trunc
            # Check if the skill is terminated
            skill_done = skill.is_terminated(env.get_observation(skill.obs_spec))
