"""Inspection perception: binary defect classification."""

from skillet.perception.inspection.defect_classifier import DefectClassifier, DefectResult
from skillet.perception.inspection.mock_defect_classifier import MockDefectClassifier

__all__ = [
    "DefectClassifier",
    "DefectResult",
    "MockDefectClassifier",
]
