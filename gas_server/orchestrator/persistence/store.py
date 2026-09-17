"""Workflow persistence protocol and concurrency errors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from gas_server.orchestrator.core.models import ExecutionEvent, WorkflowState


class WorkflowNotFound(KeyError):
    pass


class StateVersionConflict(RuntimeError):
    pass


class LeaseConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionLease:
    workflow_id: str
    owner_id: str
    expires_at: datetime


class WorkflowStore(Protocol):
    def create(self, state: WorkflowState) -> WorkflowState: ...
    def load(self, workflow_id: str) -> WorkflowState: ...
    def save(self, state: WorkflowState, expected_version: int) -> WorkflowState: ...
    def append_event(self, workflow_id: str, event: ExecutionEvent) -> None: ...
    def acquire_execution_lease(
        self, workflow_id: str, owner_id: str, ttl_seconds: int
    ) -> ExecutionLease: ...
    def release_execution_lease(self, lease: ExecutionLease) -> None: ...
