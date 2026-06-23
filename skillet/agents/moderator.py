from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from pynput import keyboard as pynput_keyboard

from skillet.core import SingleSkill
from skillet.core.env import Environment
from skillet.core.skill import Skill, SkillStatusCodes
from skillet.core.spaces import ActionSpec
from skillet.envs.skillet_env import SkilletEnv

if TYPE_CHECKING:
    from skillet.controllers.devices import DeviceBase


class KeyboardListener:
    def __init__(self):
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._pressed_keys: set[str] = set()
        self._additional_callbacks: dict[str, Callable] = {}

    def stop(self):
        """Stop the keyboard listener."""
        if hasattr(self, "_listener") and self._listener.is_alive():
            self._listener.stop()

    def start(self):
        self._listener.start()

    def _key_to_char(self, key) -> str | None:
        """Convert a pynput key object to an uppercase character string, or None."""
        try:
            return key.char.upper()
        except AttributeError:
            return None

    def _on_press(self, key):
        char = self._key_to_char(key)
        if char is None:
            return

        # Guard against key-repeat events firing duplicate additions
        if char in self._pressed_keys:
            return
        self._pressed_keys.add(char)

        # User-registered callbacks
        if char in self._additional_callbacks:
            self._additional_callbacks[char]()

    def _on_release(self, key):
        char = self._key_to_char(key)
        if char is None:
            return

        self._pressed_keys.discard(char)

    def add_callback(self, key: str, func: Callable):
        """Register a function to call when a key is pressed.

        Args:
            key: Single character string, e.g. "P".
            func: Zero-argument callable.

        """
        self._additional_callbacks[key.upper()] = func


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
            "===[SkilletExpModerator]===\nO: Quit Experiment\nP: Stop Robot\nK: Return Robot to Home\nL: Resume Robot Experiment\n"
        )

        def quit_handler():
            self._status = ExpStatusCodes.QUIT
            self._intervention = True

        def stop_handler():
            self._status = ExpStatusCodes.STOP
            self._intervention = True

        def home_handler():
            self._status = ExpStatusCodes.HOME
            self._intervention = True

        def resume_handler():
            if self._status != ExpStatusCodes.RUNNING:
                self._status = ExpStatusCodes.RESUME
                print("[INFO][MODERATOR] Resuming the robot experiment.")
                self._intervention = True

        self._listener.add_callback("o", quit_handler)
        self._listener.add_callback("p", stop_handler)
        self._listener.add_callback("k", home_handler)
        self._listener.add_callback("l", resume_handler)

        self._listener.start()

    def poll(self, env: Environment, skill: Skill = None) -> tuple[torch.Tensor, ActionSpec]:
        """Poll the correct action on a skill."""
        if self._intervention:
            if self._status == ExpStatusCodes.QUIT:
                print("[INFO][MODERATOR] Quitting.")
                if skill is not None:
                    skill.status = SkillStatusCodes.FAILED
                self._action_spec = env.coerce_action_spec("twist_tcp")
                self._action = torch.as_tensor([0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).to(self._action_spec.device)
            elif self._status == ExpStatusCodes.HOME:
                print("[INFO][MODERATOR] Returning to home position.")
                self._action_spec = env.coerce_action_spec("tcp_cart")
                if skill is not None:
                    skill.status = SkillStatusCodes.FAILED
                self._action = self._home_pos.to(self._action_spec.device)
            elif self._status == ExpStatusCodes.STOP:
                print("[INFO][MODERATOR] Stopping the robot.")
                if skill is not None:
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
        self, env: SkilletEnv, skill: SingleSkill, args: list[str]
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

    def run_teleop_loop(self, env: SkilletEnv, teleop_interface: "DeviceBase") -> None:
        """Run the teleop in the environment."""
        terminated = False
        while not terminated or not self._exp_paused:
            curr_tcp_pose = env._get_tcp_pose_b()
            teleop_actions = teleop_interface.advance(curr_tcp_pose)

            # assuming teleop is a tensor
            actions = teleop_actions.repeat(env.num_envs, 1)
            recovery_action, action_spec = self.poll(env)
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
            env.step(actions, action_spec=env.action_spec_twist_tcp)
