import select
import sys
import termios
import threading
import tty
from enum import IntEnum
from typing import Literal

import numpy as np
import torch

from skillet.core import SingleSkill
from skillet.core.env import Environment
from skillet.core.skill import Skill, SkillStatusCodes
from skillet.core.spaces import ActionSpec


class KeyboardListener:
    def __init__(self):
        self._thread = None
        self._running = False
        self._key_callbacks = {}
        self._default_callback = None
        self._fd = None
        self._old_termios = None

    def _listen(self) -> None:
        while self._running:
            try:
                key = self._read_key()
            except Exception:
                break

            handler = self._key_callbacks.get(key) or self._default_callback
            if handler:
                handler(key)

    def on_key(self, key: str) -> None:
        """Register a callback for a specific key."""

        def decorator(func):
            self._key_callbacks[key] = func
            return func

        return decorator

    def _read_key(self) -> str:
        ch = sys.stdin.read(1)
        if ch == "\x1b" and select.select([sys.stdin], [], [], 0)[0]:
            ch += sys.stdin.read(2)
        return ch

    def start(self):
        if self._running:
            return
        self._fd = sys.stdin.fileno()
        if sys.stdin.isatty():
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # or tty.setraw(self._fd) if you need full raw
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        if self._old_termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            self._old_termios = None


HOME_TCP_CART = [0.25, 0.0, 0.3, np.pi, 0.0, np.pi / 2, 0.0]


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
        self._intervention = False
        self._action = None
        self._action_spec = None
        print(
            "===[SkilletExpModerator]===\nQ: Quit Experiment\nX: Stop Robot\nH: Return Robot to Home\nR: Resume Robot Experiment\n"
        )

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

        @self._listener.on_key("r")
        def resume_handler(key):
            if self._status != ExpStatusCodes.RUNNING:
                self._status = ExpStatusCodes.RESUME
                self._intervention = True

        self._listener.start()

    def poll(self, env: Environment, skill: Skill) -> tuple[torch.Tensor, ActionSpec]:
        if self._intervention:
            if self._status == ExpStatusCodes.QUIT:
                print("[INFO][MODERATOR] Quitting.")
                skill.status = SkillStatusCodes.FAILED
                self._action_spec = env.coerce_action_spec("twist_tcp")
                self._action = torch.as_tensor([0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).to(self._action_spec.device)
            elif self._status == ExpStatusCodes.HOME:
                print("[INFO][MODERATOR] Returning to home position.")
                self._action_spec = env.coerce_action_spec("tcp_cart")
                skill.status = SkillStatusCodes.FAILED
                self._action = self._home_pos.to(self._action_spec.device)
            elif self._status == ExpStatusCodes.STOP:
                print("[INFO][MODERATOR] Stopping the robot.")
                skill.status = SkillStatusCodes.FAILED
                self._action_spec = env.coerce_action_spec("twist_tcp")
                obs = env.get_observation(obs_spec=env.coerce_obs_spec("gripper"))
                self._action = (
                    torch.as_tensor([0, 0, 0, 0, 0, 0, obs["gripper"].squeeze().item()])
                    .unsqueeze(0)
                    .to(self._action_spec.device)
                )
            elif self._status == ExpStatusCodes.RESUME:
                print("[INFO][MODERATOR] Resuming the robot experiment.")
            self._intervention = False
        return self._action, self._action_spec

    def run_skill(
        self, env: Environment, skill: SingleSkill, args: list[str]
    ) -> tuple[bool, Literal[SkillStatusCodes.SUCCESS, SkillStatusCodes.FAILED]]:
        """Run the skill in the environment."""
        obs = env.get_observation(skill.obs_spec)
        skill.initiate(obs, args)
        skill_done = skill.is_terminated(env.get_observation(skill.obs_spec))

        terminated = False
        while not skill_done or terminated or self._exp_paused:
            recovery_action, action_spec = self.poll(env, skill)
            if self._status != ExpStatusCodes.RUNNING:
                if recovery_action is not None and action_spec is not None:
                    _, _, _, _, _ = env.step(recovery_action, action_spec=action_spec)
                if self._status == ExpStatusCodes.RESUME:
                    self._status = ExpStatusCodes.RUNNING
                    break
                if self._status == ExpStatusCodes.QUIT:
                    terminated = True
                    self._listener.stop()
                    break
                continue
            # Get the next action with the low-level observation
            action = skill.get_action(env.get_observation(skill.obs_spec))
            # Take a step in the environment
            _, _, term, trunc, _ = env.step(action, action_spec=skill.action_spec)
            terminated = term | trunc
            # Check if the skill is terminated
            skill_done = skill.is_terminated(env.get_observation(skill.obs_spec))
        return terminated, skill.status
