"""Local ServiceRegistry transport used by development and deterministic workers."""

from __future__ import annotations

from gas_server.core.service_registry import get_service_registration
from gas_server.orchestrator.core.models import TaskBindingState, TaskState
from gas_server.orchestrator.execution.normalization import (
    NormalizedWorkerResult,
    normalize_worker_response,
)


class LocalRegistryTransport:
    def execute(
        self,
        task: TaskState,
        binding: TaskBindingState,
        input_paths: list[str],
    ) -> NormalizedWorkerResult:
        registration = get_service_registration(binding.agent_id)
        worker = registration.build_agent()
        if hasattr(worker, "set_request_parameters"):
            worker.set_request_parameters(dict(task.parameters))
        response = worker.run(
            query=binding.instruction,
            input_dataset_paths=input_paths,
        )
        return normalize_worker_response(response)
