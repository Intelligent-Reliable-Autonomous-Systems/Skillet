"""InspectSkill: moves arm to an inspection viewpoint above a block  """

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skillet.core.checked_skill import CheckedSkill, FailureReason, SkillResult
from skillet.core.skill import SkillStatusCodes
from skillet.scene.base import Scene

if TYPE_CHECKING:
    pass


class InspectSkill(CheckedSkill):
    """Moves the arm to a top-face inspection viewpoint above a target block.

    The resulting position must be in the robot workspace.
    """

    def __init__(self, scene: Scene) -> None:
        """Initialize.

        Args:
            scene: The world-model scene; used for pose lookups.

        """
        self._scene = scene
        self._target_block_id: int | None = None
        self._status: int = SkillStatusCodes.UNINITIATED

    def set_target(self, block_id: int) -> None:
        """Set the target block id before calling preconditions or execute."""
        self._target_block_id = block_id

    # ------------------------------------------------------------------
    # CheckedSkill contract
    # ------------------------------------------------------------------

    def preconditions(self, world: Scene) -> bool:
        """Return True iff the target block is reachable from the front."""
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        if not block.is_pose_known():
            return False
        return float(block.pose[0]) > 0.0

    def postconditions(self, world: Scene) -> bool:
        """Return True iff the block is still visible after the approach.
        """
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        return block.is_pose_known()

    # ------------------------------------------------------------------
    # Convenience method (used by tests and task scripts)
    # ------------------------------------------------------------------

    def execute(self, scene: Scene) -> SkillResult:
        """Check preconditions and return a structured result.

        Motion to the viewpoint is not executed here.
        """
        if not self.preconditions(scene):
            return SkillResult.fail(FailureReason.PRECONDITION_NOT_MET)
        return SkillResult.ok()

    # ------------------------------------------------------------------
    # SingleSkill interface — stubs; wired to motion primitive
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