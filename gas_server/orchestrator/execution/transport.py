"""Worker transport protocol."""

from __future__ import annotations

from typing import Protocol

from gas_server.orchestrator.core.models import TaskBindingState, TaskState
from gas_server.orchestrator.execution.normalization import NormalizedWorkerResult


class WorkerTransport(Protocol):
    def execute(
        self,
        task: TaskState,
        binding: TaskBindingState,
        input_paths: list[str],
    ) -> NormalizedWorkerResult: ...
