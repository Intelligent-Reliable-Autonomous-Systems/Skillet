"""Inspection perception: binary defect classification and viewpoint planning."""

from skillet.perception.inspection.defect_classifier import DefectClassifier, DefectResult
from skillet.perception.inspection.mock_defect_classifier import MockDefectClassifier
from skillet.perception.inspection.viewpoint_planner import InspectionViewpointPlanner, ViewpointPlanResult

__all__ = [
    "DefectClassifier",
    "DefectResult",
    "MockDefectClassifier",
    "InspectionViewpointPlanner",
    "ViewpointPlanResult",
]
