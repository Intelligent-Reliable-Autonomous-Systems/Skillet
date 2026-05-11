"""DiscardSkill: picks a defective block and places it at the discard location."""

from __future__ import annotations

from typing import Any

from skillet.core.checked_skill import CheckedSkill, FailureReason, SkillResult
from skillet.core.skill import SkillStatusCodes
from skillet.scene.base import Scene
from skillet.scene.objects.discard_location import DiscardLocation
from skillet.scene.objects.inspectable_cube import InspectableCube


class DiscardSkill(CheckedSkill):
    """Picks a defective InspectableCube and places it within the discard region.

    Preconditions (checked against the world model):
      - A DiscardLocation with a known pose exists in the scene.
      - Target block is an InspectableCube with defective=True and a known pose.

    Postconditions:
      - Block's XY position lies within the discard location's footprint.
    """

    def __init__(self, scene: Scene) -> None:
        """Initialize.

        Args:
            scene: The world-model scene; used for object lookups.

        """
        self._scene = scene
        self._target_block_id: int | None = None
        self._status: int = SkillStatusCodes.UNINITIATED

    def set_target(self, block_id: int) -> None:
        """Set the target block id before calling preconditions or execute."""
        self._target_block_id = block_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_discard_location(scene: Scene) -> DiscardLocation | None:
        for obj in scene.objects:
            if isinstance(obj, DiscardLocation):
                return obj
        return None

    # ------------------------------------------------------------------
    # CheckedSkill contract
    # ------------------------------------------------------------------

    def preconditions(self, world: Scene) -> bool:
        """Return True iff there is a known discard region and the block is defective."""
        discard = self._find_discard_location(world)
        if discard is None or not discard.is_pose_known():
            return False
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        if not isinstance(block, InspectableCube):
            return False
        if block.defective is not True:
            return False
        return block.is_pose_known()

    def postconditions(self, world: Scene) -> bool:
        """Return True iff the block's XY centre is inside the discard footprint."""
        discard = self._find_discard_location(world)
        if discard is None or not discard.is_pose_known():
            return False
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        if not block.is_pose_known():
            return False
        aabb = discard.aabb  # (min_x, min_y, min_z, max_x, max_y, max_z)
        bx = float(block.pose[0])
        by = float(block.pose[1])
        return float(aabb[0]) <= bx <= float(aabb[3]) and float(aabb[1]) <= by <= float(aabb[4])

    # ------------------------------------------------------------------
    # Convenience method (used by tests and task scripts)
    # ------------------------------------------------------------------

    def execute(self, scene: Scene) -> SkillResult:
        """Check preconditions and return a structured result """
        if not self.preconditions(scene):
            return SkillResult.fail(FailureReason.PRECONDITION_NOT_MET)
        return SkillResult.ok()

    # ------------------------------------------------------------------
    # SingleSkill interface — stubs; wired to PickSkill+PlaceSkill
    # ------------------------------------------------------------------

    @property
    def policy(self) -> Any:
        raise NotImplementedError("motion policy not wired")

    @property
    def status(self) -> int:
        return self._status

    def initiate(self, obs: Any, params: Any) -> None:
        self._target_block_id = int(params)
        self._status = SkillStatusCodes.RUNNING

    def get_action(self, obs: Any) -> Any:
        raise NotImplementedError("motion action not wired")
