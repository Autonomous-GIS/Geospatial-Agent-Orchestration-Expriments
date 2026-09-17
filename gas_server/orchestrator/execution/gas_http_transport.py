"""Canonical HTTP GAS worker transport."""

from __future__ import annotations

from typing import Any

import requests

from gas_server.orchestrator.core.models import TaskBindingState, TaskState
from gas_server.orchestrator.execution.normalization import (
    NormalizedWorkerResult,
    normalize_worker_response,
)


class GasHttpTransport:
    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        headers: dict[str, str] | None = None,
        session: requests.Session | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.headers = dict(headers or {})
        self.session = session or requests.Session()

    def execute(
        self,
        task: TaskState,
        binding: TaskBindingState,
        input_paths: list[str],
    ) -> NormalizedWorkerResult:
        if not binding.endpoint:
            raise ValueError(f"Binding {binding.binding_id!r} has no GAS endpoint.")
        payload: dict[str, Any] = {
            "task": {"instructions": binding.instruction},
            "inputs": {"input_datasets": input_paths},
            "parameters": dict(task.parameters),
            "metadata": {
                "task_id": task.task_id,
                "binding_id": binding.binding_id,
                "operation": task.operation,
                "attempt": binding.attempt,
            },
        }
        response = self.session.post(
            binding.endpoint,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return normalize_worker_response(response.json())
