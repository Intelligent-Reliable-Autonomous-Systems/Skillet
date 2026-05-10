"""MockDefectClassifier for sim runs without a VLM"""

from __future__ import annotations

import numpy as np

from skillet.perception.inspection.defect_classifier import DefectClassifier, DefectResult


class MockDefectClassifier(DefectClassifier):
    """Returns canned results from a fixture dict keyed by object_id"""

    def __init__(self, results: dict[str, DefectResult]) -> None:
        self._results = results

    def classify(self, image: np.ndarray, object_id: str) -> DefectResult:
        """Return the pre-set result for object_id."""
        if object_id not in self._results:
            raise KeyError(f"MockDefectClassifier has no fixture for object_id={object_id!r}")
        return self._results[object_id]
