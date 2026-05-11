"""SkillEventLogger: append-only JSON-lines event log for skill execution traces."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skillet.core.checked_skill import SkillResult
from skillet.scene.base import Scene


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SkillEventLogger:
    """Appends structured events to a JSON-lines file.

    Each line is a self-contained JSON object with at minimum:
      - ``event``: one of ``skill_start | skill_end | planner_decision |
        classifier_verdict | world_model_snapshot``
      - ``run_id``: shared identifier for a single task execution.
      - ``ts``: ISO-8601 timestamp in UTC.
    """

    def __init__(self, path: str | Path, run_id: str | None = None) -> None:
        """Open the log file for appending.

        Args:
            path: Destination file (created if absent; parent dir must exist).
            run_id: Shared identifier for this execution run.  Defaults to
                the UTC timestamp at construction time.

        """
        self._path = Path(path)
        self._run_id = run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._lock = threading.Lock()
        self._fh = self._path.open("a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Shared run identifier written into every event."""
        return self._run_id

    def log_skill_start(self, skill_name: str, params: Any = None) -> None:
        """Record that a skill has been initiated.

        Args:
            skill_name: Skill class name or symbolic action string.
            params: Skill parameters (must be JSON-serialisable or None).

        """
        self._write({
            "event": "skill_start",
            "skill": skill_name,
            "params": params,
        })

    def log_skill_end(self, skill_name: str, result: SkillResult) -> None:
        """Record that a skill has terminated.

        Args:
            skill_name: Same name used in ``log_skill_start``.
            result: Structured outcome from the skill.

        """
        self._write({
            "event": "skill_end",
            "skill": skill_name,
            "success": result.success,
            "failure_reason": result.failure_reason.name if result.failure_reason else None,
            "message": result.message,
        })

    def log_planner_decision(self, action: str, parameters: list[str]) -> None:
        """Record a PDDL planner action selection.

        Args:
            action: Grounded action name (e.g. ``"inspect-for-defects"``).
            parameters: Symbolic parameter list (e.g. ``["block_0", "table"]``).

        """
        self._write({
            "event": "planner_decision",
            "action": action,
            "parameters": parameters,
        })

    def log_classifier_verdict(
        self,
        object_id: str,
        defective: bool,
        confidence: float,
    ) -> None:
        """Record a defect classifier result.

        Args:
            object_id: Symbolic id of the block that was inspected.
            defective: Binary verdict from the classifier.
            confidence: Confidence score in [0, 1].

        """
        self._write({
            "event": "classifier_verdict",
            "object_id": object_id,
            "defective": defective,
            "confidence": confidence,
        })

    def log_world_model_snapshot(self, scene: Scene) -> None:
        """Record a snapshot of the current scene state.

        Args:
            scene: The world-model scene to snapshot.

        """
        objects = []
        for obj in scene.objects:
            entry: dict[str, Any] = {
                "id": obj.object_id,
                "name": obj.name,
                "type": obj.type_name,
            }
            if obj.is_pose_known():
                try:
                    entry["position"] = obj.pose[:3].cpu().tolist()
                except Exception:  # noqa: BLE001
                    pass
            objects.append(entry)
        self._write({"event": "world_model_snapshot", "objects": objects})

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> SkillEventLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


    def _write(self, payload: dict[str, Any]) -> None:
        payload["run_id"] = self._run_id
        payload["ts"] = _now_iso()
        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
