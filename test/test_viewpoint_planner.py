"""Tests for InspectionViewpointPlanner"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pinocchio = pytest.importorskip("pinocchio")
pytest.importorskip("hppfcl")

import pinocchio as pin  # noqa: E402

from skillet.perception.inspection.viewpoint_planner import InspectionViewpointPlanner, ViewpointPlanResult  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).parents[1]
_URDF = _REPO_ROOT / "skillet_tasks/assets/kortex/kinova_gen3/gen3_2f85.urdf"
_SRDF = _REPO_ROOT / "skillet_tasks/assets/kortex/kinova_gen3/gen3_2f85.srdf"
_PKG_DIRS = [str(_REPO_ROOT / "skillet_tasks/assets/kortex/kinova_gen3/")]

pytestmark = pytest.mark.skipif(
    not _URDF.exists(),
    reason="Gen3 URDF not found; skipping viewpoint planner tests",
)

_HALF_EXTENTS = np.array([0.025, 0.025, 0.025])  # 5 cm cube
_BLOCK_FORWARD = pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.1]))
_BLOCK_BEHIND = pin.SE3(np.eye(3), np.array([-0.4, 0.0, 0.1]))


@pytest.fixture(scope="module")
def planner() -> InspectionViewpointPlanner:
    return InspectionViewpointPlanner(
        urdf_path=str(_URDF),
        srdf_path=str(_SRDF),
        package_dirs=_PKG_DIRS,
        standoff_m=0.20,
    )


def test_viewpoint_pose_geometry(planner: InspectionViewpointPlanner) -> None:
    """Viewpoint is placed directly above the top face at the configured standoff."""
    result = planner.plan(_BLOCK_FORWARD, _HALF_EXTENTS)

    assert result.reachable
    assert len(result.viewpoints) == 1

    vp = result.viewpoints[0]
    assert vp.translation[0] == pytest.approx(0.4, abs=1e-6)
    assert vp.translation[1] == pytest.approx(0.0, abs=1e-6)
    assert vp.translation[2] == pytest.approx(0.1 + _HALF_EXTENTS[2] + 0.20, abs=1e-6)


def test_standoff_within_tolerance(planner: InspectionViewpointPlanner) -> None:
    """Standoff distance between viewpoint and block top face matches configuration."""
    result = planner.plan(_BLOCK_FORWARD, _HALF_EXTENTS)
    assert result.reachable

    block_top_z = _BLOCK_FORWARD.translation[2] + _HALF_EXTENTS[2]
    actual_standoff = result.viewpoints[0].translation[2] - block_top_z
    assert actual_standoff == pytest.approx(0.20, abs=1e-3)


def test_viewpoint_deterministic(planner: InspectionViewpointPlanner) -> None:
    """Repeated calls on the same input return the same viewpoint pose."""
    r1 = planner.plan(_BLOCK_FORWARD, _HALF_EXTENTS)
    r2 = planner.plan(_BLOCK_FORWARD, _HALF_EXTENTS)

    assert r1.reachable == r2.reachable
    if r1.reachable:
        np.testing.assert_allclose(r1.viewpoints[0].translation, r2.viewpoints[0].translation)
        np.testing.assert_allclose(r1.viewpoints[0].rotation, r2.viewpoints[0].rotation)


def test_block_behind_robot_rejected(planner: InspectionViewpointPlanner) -> None:
    """Block behind the robot (x <= 0) is rejected before IK."""
    result = planner.plan(_BLOCK_BEHIND, _HALF_EXTENTS)

    assert not result.reachable
    assert result.failure_reason == "not_in_workspace"
