"""Transactional SQLite implementation of the durable workflow store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gas_server.orchestrator.core.models import ExecutionEvent, WorkflowState, utc_now
from gas_server.orchestrator.persistence.store import (
    ExecutionLease,
    LeaseConflict,
    StateVersionConflict,
    WorkflowNotFound,
)


class SQLiteWorkflowStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    state_version INTEGER NOT NULL,
                    schema_version TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workflow_events (
                    event_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id)
                );
                CREATE TABLE IF NOT EXISTS execution_leases (
                    workflow_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id)
                );
                """
            )

    @staticmethod
    def _state_json(state: WorkflowState) -> str:
        return state.model_dump_json(by_alias=True)

    def create(self, state: WorkflowState) -> WorkflowState:
        candidate = state.model_copy(deep=True)
        candidate.state_version = 0
        candidate.updated_at = utc_now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO workflows(
                        workflow_id, state_version, schema_version, state_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.workflow_id,
                        candidate.state_version,
                        candidate.schema_version,
                        self._state_json(candidate),
                        candidate.created_at.isoformat(),
                        candidate.updated_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise StateVersionConflict(
                f"Workflow {candidate.workflow_id!r} already exists."
            ) from exc
        return candidate

    def load(self, workflow_id: str) -> WorkflowState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
        if row is None:
            raise WorkflowNotFound(workflow_id)
        return WorkflowState.model_validate_json(row["state_json"])

    def save(self, state: WorkflowState, expected_version: int) -> WorkflowState:
        candidate = state.model_copy(deep=True)
        candidate.state_version = expected_version + 1
        candidate.updated_at = utc_now()
        payload = self._state_json(candidate)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workflows
                SET state_version = ?, schema_version = ?, state_json = ?, updated_at = ?
                WHERE workflow_id = ? AND state_version = ?
                """,
                (
                    candidate.state_version,
                    candidate.schema_version,
                    payload,
                    candidate.updated_at.isoformat(),
                    candidate.workflow_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?",
                    (candidate.workflow_id,),
                ).fetchone()
                if exists is None:
                    raise WorkflowNotFound(candidate.workflow_id)
                raise StateVersionConflict(
                    f"Expected workflow version {expected_version} is stale."
                )
        return candidate

    def append_event(self, workflow_id: str, event: ExecutionEvent) -> None:
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO workflow_events(event_id, workflow_id, event_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        workflow_id,
                        event.model_dump_json(),
                        event.created_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise WorkflowNotFound(workflow_id) from exc

    def acquire_execution_lease(
        self, workflow_id: str, owner_id: str, ttl_seconds: int
    ) -> ExecutionLease:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=max(1, ttl_seconds))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            workflow = connection.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            row = connection.execute(
                "SELECT owner_id, expires_at FROM execution_leases WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
            if row is not None:
                existing_expiry = datetime.fromisoformat(row["expires_at"])
                if existing_expiry > now and row["owner_id"] != owner_id:
                    raise LeaseConflict(
                        f"Workflow {workflow_id!r} is leased by another executor."
                    )
            connection.execute(
                """
                INSERT INTO execution_leases(workflow_id, owner_id, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    expires_at = excluded.expires_at
                """,
                (workflow_id, owner_id, expires_at.isoformat()),
            )
        return ExecutionLease(workflow_id, owner_id, expires_at)

    def release_execution_lease(self, lease: ExecutionLease) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM execution_leases WHERE workflow_id = ? AND owner_id = ?",
                (lease.workflow_id, lease.owner_id),
            )

    def delete_expired_terminal_workflows(self, older_than: datetime) -> int:
        """Retention helper; callers must invoke it explicitly outside execution."""

        threshold = older_than.astimezone(timezone.utc).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT workflow_id, state_json FROM workflows WHERE updated_at < ?",
                (threshold,),
            ).fetchall()
            terminal_ids = [
                row["workflow_id"]
                for row in rows
                if WorkflowState.model_validate_json(row["state_json"]).terminal_state
                is not None
            ]
            for workflow_id in terminal_ids:
                connection.execute(
                    "DELETE FROM workflow_events WHERE workflow_id = ?", (workflow_id,)
                )
                connection.execute(
                    "DELETE FROM execution_leases WHERE workflow_id = ?", (workflow_id,)
                )
                connection.execute(
                    "DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,)
                )
        return len(terminal_ids)
