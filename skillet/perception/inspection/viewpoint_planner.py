"""Inspection viewpoint planner: single top-face viewpoint per block"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class ViewpointPlanResult:
    """Structured outcome of viewpoint planning for a single block."""

    viewpoints: tuple[pin.SE3, ...]
    reachable: bool
    failure_reason: str | None = None


class InspectionViewpointPlanner:
    """Computes a single top-face inspection viewpoint for a block.

    Only viewpoints in the robot's forward half-space (base frame x > 0) are valid.
    Reachability is validated via a standalone Pinocchio kinematic model;
    collision is validated via CollisionProximityMonitor.
    """

    _EE_FRAME: str = "end_effector_link"
    # EE rotation for top-face inspection: z-axis pointing down
    _R_EE_DOWN: np.ndarray = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    _COLLISION_DISTANCE_THRESHOLD: float = 0.02  # metres

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str | None = None,
        package_dirs: list[str] | None = None,
        standoff_m: float = 0.20,
    ) -> None:
        self._standoff_m = standoff_m
        self._kin_model, self._col_model, _ = pin.buildModelsFromUrdf(
            urdf_path, list(package_dirs) if package_dirs else [],
        )
        self._kin_data = self._kin_model.createData()
        self._col_model.addAllCollisionPairs()
        if srdf_path:
            pin.removeCollisionPairs(self._kin_model, self._col_model, srdf_path)
        self._col_data = pin.GeometryData(self._col_model)
        self._ee_frame_id = self._kin_model.getFrameId(self._EE_FRAME)

    def plan(
        self,
        block_pose: pin.SE3,
        block_half_extents: np.ndarray,
    ) -> ViewpointPlanResult:
        """Return a single reachable top-face viewpoint or a structured failure.

        Args:
            block_pose: SE3 pose of the block centre in the robot base frame.
            block_half_extents: Half-lengths [hx, hy, hz] of the block in metres.

        Returns:
            ViewpointPlanResult with one viewpoint on success or a failure_reason on failure.
        """
        viewpoint = self._top_face_viewpoint(block_pose, block_half_extents)

        if viewpoint.translation[0] <= 0.0:
            return ViewpointPlanResult(
                viewpoints=(), reachable=False, failure_reason="not_in_workspace",
            )

        q = self._solve_ik(viewpoint)
        if q is None:
            return ViewpointPlanResult(viewpoints=(), reachable=False, failure_reason="ik_failed")

        if not self._is_collision_free(q):
            return ViewpointPlanResult(viewpoints=(), reachable=False, failure_reason="collision")

        return ViewpointPlanResult(viewpoints=(viewpoint,), reachable=True)

    def _top_face_viewpoint(self, block_pose: pin.SE3, half_extents: np.ndarray) -> pin.SE3:
        pos = block_pose.translation.copy()
        pos[2] += float(half_extents[2]) + self._standoff_m
        return pin.SE3(self._R_EE_DOWN.copy(), pos)

    def _solve_ik(
        self,
        target: pin.SE3,
        max_iter: int = 300,
        tol: float = 1e-4,
        step: float = 0.5,
        lambda_val: float = 0.01,
    ) -> np.ndarray | None:
        """Damped-least-squares IK in world-aligned end-effector space with random restarts.

        Error and Jacobian are both expressed in LOCAL_WORLD_ALIGNED (world-frame axes,
        EE origin) — the same convention used by kortex_env._update_pinocchio_state.
        Returns full model q on convergence, or None if all restarts fail.
        """
        starts: list[np.ndarray] = [pin.neutral(self._kin_model)]
        for _ in range(7):
            starts.append(pin.randomConfiguration(self._kin_model))

        for q0 in starts:
            q = q0.copy()
            for _ in range(max_iter):
                pin.forwardKinematics(self._kin_model, self._kin_data, q)
                pin.computeJointJacobians(self._kin_model, self._kin_data, q)
                pin.updateFramePlacements(self._kin_model, self._kin_data)
                current = self._kin_data.oMf[self._ee_frame_id]

                # Position and orientation error in world frame
                err = np.empty(6)
                err[:3] = target.translation - current.translation
                err[3:] = pin.log3(target.rotation @ current.rotation.T)

                if np.linalg.norm(err) < tol:
                    return q

                J = pin.getFrameJacobian(
                    self._kin_model, self._kin_data, self._ee_frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                JJt = J @ J.T
                dq = J.T @ np.linalg.solve(JJt + lambda_val**2 * np.eye(6), err)
                q = pin.integrate(self._kin_model, q, step * dq)

        return None

    def _is_collision_free(self, q: np.ndarray) -> bool:
        pin.computeDistances(
            self._kin_model, self._kin_data, self._col_model, self._col_data, q,
        )
        return all(
            r.min_distance > self._COLLISION_DISTANCE_THRESHOLD
            for r in self._col_data.distanceResults
        )
