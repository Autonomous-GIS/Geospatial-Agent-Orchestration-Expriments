"""Common Execution Ledger for GAS Benchmark.

Instruments the common GAS service boundary to record every actual worker service
invocation across C0 (Magentic-One) and C1-C5 (GAS Orchestrator) in a condition-neutral manner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread-local storage for the active execution ledger
_local = threading.local()


def _compute_file_sha256(path: Path | str | None) -> str | None:
    """Compute SHA-256 hash of a file if it exists."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.debug("Could not hash file %s: %s", p, e)
        return None


@dataclass
class ExecutionEvent:
    """A structured record of one worker service invocation."""

    call_id: str
    trajectory_id: str
    agent_id: str
    operation: str
    query: str
    timestamp_start: float
    timestamp_end: float
    duration_seconds: float
    input_artifact_paths: List[str] = field(default_factory=list)
    input_artifact_hashes: Dict[str, Optional[str]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "completed"  # 'completed', 'failed', 'error_503', 'rejected'
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    output_artifact_paths: List[str] = field(default_factory=list)
    output_artifact_hashes: Dict[str, Optional[str]] = field(default_factory=dict)
    summary: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionLedger:
    """Context manager and tracker for worker invocations within a trajectory."""

    def __init__(self, trajectory_id: str):
        self.trajectory_id = trajectory_id
        self.events: List[ExecutionEvent] = []
        self._active = False
        self._prev_ledger: Optional[ExecutionLedger] = None

    def __enter__(self) -> ExecutionLedger:
        self._prev_ledger = getattr(_local, "active_ledger", None)
        _local.active_ledger = self
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _local.active_ledger = self._prev_ledger
        self._active = False

    def record_event(self, event: ExecutionEvent):
        """Append an execution event to the ledger."""
        self.events.append(event)

    def get_events(self) -> List[ExecutionEvent]:
        """Return a copy of all recorded events."""
        return list(self.events)

    def to_list(self) -> List[Dict[str, Any]]:
        """Return all events as dictionaries."""
        return [e.to_dict() for e in self.events]


def get_active_ledger() -> Optional[ExecutionLedger]:
    """Retrieve the currently active thread-local execution ledger, if any."""
    return getattr(_local, "active_ledger", None)


def record_service_call(
    agent_id: str,
    query: str,
    input_dataset_paths: List[str] | None,
    parameters: Dict[str, Any] | None,
    start_time: float,
    end_time: float,
    result: Dict[str, Any],
) -> Optional[ExecutionEvent]:
    """Record a worker service invocation in the active execution ledger if one exists."""
    ledger = get_active_ledger()
    if ledger is None:
        return None

    paths = list(input_dataset_paths or [])
    input_hashes = {p: _compute_file_sha256(p) for p in paths if Path(p).is_file()}

    output_paths = list(result.get("artifacts", []) or [])
    output_hashes = {p: _compute_file_sha256(p) for p in output_paths if Path(p).is_file()}

    error_msg = result.get("error")
    error_code = None
    status = "completed"

    if error_msg:
        status = "failed"
        if "503" in str(error_msg):
            status = "error_503"
            error_code = "E_SERVICE_UNAVAILABLE"
        elif "E_CRS_MISMATCH" in str(error_msg):
            error_code = "E_CRS_MISMATCH"
        elif "E_INVALID_GEOMETRY" in str(error_msg):
            error_code = "E_INVALID_GEOMETRY"
        elif "E_GRID_MISALIGNED" in str(error_msg):
            error_code = "E_GRID_MISALIGNED"
        elif "E_FORMAT_UNSUPPORTED" in str(error_msg):
            error_code = "E_FORMAT_UNSUPPORTED"
        else:
            error_code = "E_GENERAL"

    params = dict(parameters or {})
    operation = str(params.get("operation") or params.get("action") or agent_id)

    event = ExecutionEvent(
        call_id=str(uuid.uuid4()),
        trajectory_id=ledger.trajectory_id,
        agent_id=agent_id,
        operation=operation,
        query=query,
        timestamp_start=start_time,
        timestamp_end=end_time,
        duration_seconds=round(max(0.0, end_time - start_time), 4),
        input_artifact_paths=paths,
        input_artifact_hashes=input_hashes,
        parameters=copy.deepcopy(params),
        status=status,
        error_code=error_code,
        error_message=error_msg,
        output_artifact_paths=output_paths,
        output_artifact_hashes=output_hashes,
        summary=str(result.get("summary") or ""),
        raw_response={k: v for k, v in result.items() if k != "artifacts"},
    )

    ledger.record_event(event)
    return event
