"""InspectForDefectsSkill: viewpoint planning + defect classification + world-model update."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from skillet.core.checked_skill import CheckedSkill, FailureReason, SkillResult
from skillet.core.skill import SkillStatusCodes
from skillet.perception.inspection.defect_classifier import DefectClassifier
from skillet.scene.base import Scene
from skillet.scene.objects.inspectable_cube import InspectableCube

if TYPE_CHECKING:
    from skillet.perception.inspection.viewpoint_planner import InspectionViewpointPlanner


class InspectForDefectsSkill(CheckedSkill):
    """Chains viewpoint planning, defect classification, and world-model update.

    Preconditions (checked against the world model):
      - target is an InspectableCube with a known pose.
      - block has not yet been inspected (defective is None).

    Postconditions:
      - block.defective is no longer None (verdict has been written).

    The planner argument is optional; omitting it skips the reachability gate
    so the skill can be used in unit tests without a Pinocchio installation.
    """

    def __init__(
        self,
        scene: Scene,
        classifier: DefectClassifier,
        planner: InspectionViewpointPlanner | None = None,
        block_half_extents: np.ndarray | None = None,
    ) -> None:
        """Initialize.

        Args:
            scene: The world-model scene.
            classifier: Defect classifier (real or mock).
            planner: Optional viewpoint planner; enables reachability gating.
            block_half_extents: Half-extents [hx, hy, hz] of the blocks in metres.
                Required when planner is provided.

        """
        self._scene = scene
        self._classifier = classifier
        self._planner = planner
        self._block_half_extents = block_half_extents
        self._target_block_id: int | None = None
        self._status: int = SkillStatusCodes.UNINITIATED

    def set_target(self, block_id: int) -> None:
        """Set the target block id before calling preconditions or execute."""
        self._target_block_id = block_id

    # ------------------------------------------------------------------
    # CheckedSkill contract
    # ------------------------------------------------------------------

    def preconditions(self, world: Scene) -> bool:
        """Return True iff the target block exists, has a known pose, and is uninspected."""
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        if not isinstance(block, InspectableCube):
            return False
        if not block.is_pose_known():
            return False
        return block.defective is None

    def postconditions(self, world: Scene) -> bool:
        """Return True iff the defect verdict has been written to the block."""
        if self._target_block_id is None:
            return False
        try:
            block = world.get_objects_from_id([self._target_block_id])[0]
        except (ValueError, IndexError):
            return False
        if not isinstance(block, InspectableCube):
            return False
        return block.defective is not None

    # ------------------------------------------------------------------
    # Convenience method (used by tests and task scripts)
    # ------------------------------------------------------------------

    def execute(self, scene: Scene, image: np.ndarray) -> SkillResult:
        """Run the full inspection pipeline and return a structured result.

        Steps:
          1. Precondition check.
          2. Viewpoint reachability gate (if planner provided).
          3. Defect classification.
          4. Write verdict to block.defective.
          5. Postcondition check.

        Args:
            scene: Current world-model scene.
            image: HxWxC uint8 BGR wrist-camera image.

        """
        if not self.preconditions(scene):
            return SkillResult.fail(FailureReason.PRECONDITION_NOT_MET)

        block = scene.get_objects_from_id([self._target_block_id])[0]

        if self._planner is not None and self._block_half_extents is not None:
            import pinocchio as pin

            pos = block.pose[:3].cpu().numpy()
            block_se3 = pin.SE3(np.eye(3), pos)
            vp_result = self._planner.plan(block_se3, self._block_half_extents)
            if not vp_result.reachable:
                return SkillResult.fail(FailureReason.IK_FAILURE, vp_result.failure_reason)

        object_id_str = block.name if block.name else str(self._target_block_id)
        verdict = self._classifier.classify(image, object_id_str)
        block.defective = verdict.defective

        if not self.postconditions(scene):
            return SkillResult.fail(FailureReason.POSTCONDITION_NOT_MET)
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
