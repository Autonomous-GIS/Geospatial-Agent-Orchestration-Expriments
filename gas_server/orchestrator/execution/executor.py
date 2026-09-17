"""Role-aware worker execution through a normalized transport."""

from __future__ import annotations

from time import monotonic

from gas_server.orchestrator.core.budgets import reserve_worker_call
from gas_server.orchestrator.core.models import (
    StructuredError,
    TaskBindingState,
    TaskState,
    WorkflowState,
    utc_now,
)
from gas_server.orchestrator.execution.normalization import NormalizedWorkerResult
from gas_server.orchestrator.execution.transport import WorkerTransport


class WorkerExecutionError(RuntimeError):
    def __init__(self, result: NormalizedWorkerResult):
        self.result = result
        message = (
            result.structured_error.message
            if result.structured_error
            else "Worker failed."
        )
        super().__init__(message)


class WorkerExecutor:
    def __init__(self, transport: WorkerTransport):
        self.transport = transport

    def execute(
        self,
        state: WorkflowState,
        task: TaskState,
        binding: TaskBindingState,
    ) -> NormalizedWorkerResult:
        reserve_worker_call(state)
        ordered = sorted(binding.input_bindings, key=lambda item: item.binding_order)
        input_paths = [state.artifacts[item.artifact_id].location for item in ordered]
        task.attempts += 1
        binding.started_at = utc_now()
        started = monotonic()
        try:
            result = self.transport.execute(task, binding, input_paths)
        except Exception as exc:
            result = NormalizedWorkerResult(
                status="failed",
                structured_error=StructuredError(
                    code="E_WORKER_TRANSPORT",
                    message=str(exc),
                    retryable=any(
                        marker in str(exc).lower()
                        for marker in ("timeout", "connection", "temporarily unavailable")
                    ),
                ),
            )
        binding.completed_at = utc_now()
        binding.duration_seconds = max(0.0, monotonic() - started)
        result.timing["worker_execution_seconds"] = binding.duration_seconds
        binding.invocation_id = result.invocation_id
        binding.status = result.status
        binding.messages = list(result.messages)
        binding.structured_error = result.structured_error
        state.record_event(
            "worker_execution_completed",
            actor="worker_executor",
            task_id=task.task_id,
            reason=(
                result.structured_error.message
                if result.structured_error
                else "Worker returned successfully."
            ),
            metadata={
                "binding_id": binding.binding_id,
                "agent_id": binding.agent_id,
                "operation": binding.operation,
                "attempt": task.attempts,
                "status": result.status,
                "duration_seconds": binding.duration_seconds,
            },
        )
        if result.status == "failed":
            task.last_error = result.structured_error
            raise WorkerExecutionError(result)
        return result
