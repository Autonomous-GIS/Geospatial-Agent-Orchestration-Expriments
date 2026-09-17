from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from gas_server.core.llm_client import InfrastructureError


@dataclass
class ProgressiveExecutionResult:
    status: str
    summary: str
    output_artifacts: list[str]
    executed_steps: list[dict[str, Any]]
    workflow_plan: dict[str, Any]
    workflow_state: dict[str, Any]
    produced_artifacts: list[str]
    execution_events: list[dict[str, Any]]
    worker_execution_time: float = 0.0
    qc_evaluations: int = 0
    qc_blocks: int = 0
    qc_interventions: int = 0
    effective_blocks: int = 0
    effective_interventions: int = 0
    qc_policy_overrides: int = 0
    replanning_iterations: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "artifacts": self.output_artifacts,
            "executed_steps": self.executed_steps,
            "workflow_plan": self.workflow_plan,
            "workflow_state": self.workflow_state,
            "produced_artifacts": self.produced_artifacts,
            "execution_events": self.execution_events,
            "worker_execution_time": self.worker_execution_time,
            "qc_evaluations": self.qc_evaluations,
            "qc_blocks": self.qc_blocks,
            "qc_interventions": self.qc_interventions,
            "effective_blocks": self.effective_blocks,
            "effective_interventions": self.effective_interventions,
            "qc_policy_overrides": self.qc_policy_overrides,
            "replanning_iterations": self.replanning_iterations,
            "error": self.error,
        }


class ProgressiveExecutionKernel:
    """Compatibility name for a production-loop invocation adapter."""

    def __init__(self, orchestrator: Any, condition: str, max_replan_rounds: int = 2):
        self.orchestrator = orchestrator
        self.condition = condition
        self.max_replan_rounds = max_replan_rounds

    def run_pipeline(
        self,
        query: str,
        initial_input_paths: list[str],
        trajectory_id: str,
    ) -> ProgressiveExecutionResult:
        parameters = dict(getattr(self.orchestrator, "request_parameters", {}) or {})
        parameters.update(
            {
                "mode": "execute",
                "condition": self.condition,
                "trajectory_id": trajectory_id,
            }
        )
        if "workflow_store_path" not in parameters:
            from gas_server.core.config import DATA_DIR

            parameters["workflow_store_path"] = str(
                DATA_DIR / "orchestrator" / "benchmark_stores" / f"{trajectory_id}.sqlite3"
            )
        self.orchestrator.set_request_parameters(parameters)
        try:
            result = self.orchestrator.run(
                query=query,
                input_dataset_paths=initial_input_paths,
                progress_callback=None,
            )
        except InfrastructureError:
            raise
        except Exception as exc:
            return ProgressiveExecutionResult(
                status="failed",
                summary="Authoritative orchestrator execution failed.",
                output_artifacts=[],
                executed_steps=[],
                workflow_plan={},
                workflow_state={},
                produced_artifacts=[],
                execution_events=[],
                error=str(exc),
            )

        outputs = result.get("outputs", {})
        state = result.get("workflow_state", {})
        terminal = outputs.get("terminal_state")
        status_map = {
            "completed": "successful",
            "human_intervention_required": "human_intervention_required",
        }
        status = status_map.get(terminal, "failed")
        tasks = state.get("tasks", {}) if isinstance(state, dict) else {}
        executed_steps = [
            {
                "step_index": index,
                "step_id": task_id,
                "agent_id": (
                    state.get("bindings", {})
                    .get(task.get("active_binding_id") or "", {})
                    .get("agent_id")
                ),
                "title": task.get("title", ""),
                "instructions": task.get("instructions", ""),
                "output_artifacts": [
                    state.get("artifacts", {}).get(item, {}).get("location")
                    for item in task.get("output_artifact_ids", [])
                    if item in state.get("artifacts", {})
                ],
                "status": task.get("status"),
                "error": task.get("last_error"),
            }
            for index, (task_id, task) in enumerate(tasks.items(), start=1)
        ]
        qc_history = state.get("qc_history", []) if isinstance(state, dict) else []
        component_calls = state.get("component_calls", []) if isinstance(state, dict) else []
        produced_artifacts = [
            artifact.get("location")
            for artifact in state.get("artifacts", {}).values()
            if artifact.get("producer_task_id") and artifact.get("location")
        ]
        execution_events = _bridge_execution_events(state, trajectory_id)
        return ProgressiveExecutionResult(
            status=status,
            summary=str(outputs.get("text") or "Workflow state updated."),
            output_artifacts=list(outputs.get("artifacts") or []),
            executed_steps=executed_steps,
            workflow_plan=dict(outputs.get("workflow_plan") or {}),
            workflow_state=state,
            produced_artifacts=produced_artifacts,
            execution_events=execution_events,
            worker_execution_time=sum(
                float(item.get("duration_seconds", 0.0) or 0.0)
                for item in execution_events
            ),
            qc_evaluations=sum(
                1 for item in component_calls if item.get("component") == "qc"
            ),
            qc_blocks=sum(
                1
                for item in qc_history
                if (item.get("proposed_verdict") or item.get("verdict")) == "BLOCK"
            ),
            qc_interventions=sum(
                1
                for item in qc_history
                if (item.get("proposed_verdict") or item.get("verdict")) == "BLOCK"
                and item.get("action_class")
                in {"LOCAL_RECOVERY", "STRUCTURAL_REPLAN", "HUMAN_CLARIFICATION"}
            ),
            effective_blocks=sum(
                1 for item in qc_history if item.get("verdict") == "BLOCK"
            ),
            effective_interventions=sum(
                1
                for item in qc_history
                if item.get("verdict") == "BLOCK"
                and (item.get("effective_action") or item.get("action_class"))
                in {"LOCAL_RECOVERY", "STRUCTURAL_REPLAN", "HUMAN_CLARIFICATION"}
            ),
            qc_policy_overrides=sum(
                1
                for item in qc_history
                if item.get("effective_action_source")
                in {
                    "deterministic_invariant_guardrail",
                    "deterministic_recovery_policy",
                    "condition_policy",
                }
                and item.get("effective_action") != item.get("action_class")
            ),
            replanning_iterations=int(
                result.get("metrics", {}).get("replans", 0) or 0
            ),
            error=None if status != "failed" else str(outputs.get("text") or terminal),
        )


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    return 0.0


def _bridge_execution_events(
    state: dict[str, Any], trajectory_id: str
) -> list[dict[str, Any]]:
    """Project neutral production binding telemetry into benchmark ledger events."""
    artifacts = state.get("artifacts", {})
    tasks = state.get("tasks", {})
    events: list[dict[str, Any]] = []
    for binding_id, binding in state.get("bindings", {}).items():
        if not binding.get("started_at"):
            continue
        task = tasks.get(binding.get("task_id"), {})
        input_ids = [item.get("artifact_id") for item in binding.get("input_bindings", [])]
        output_ids = list(binding.get("output_artifact_ids", []))
        input_paths = [
            artifacts[item]["location"]
            for item in input_ids
            if item in artifacts and artifacts[item].get("location")
        ]
        output_paths = [
            artifacts[item]["location"]
            for item in output_ids
            if item in artifacts and artifacts[item].get("location")
        ]
        error = binding.get("structured_error") or {}
        status = "completed" if binding.get("status") == "successful" else "failed"
        events.append(
            {
                "call_id": binding.get("invocation_id") or binding_id,
                "trajectory_id": trajectory_id,
                "agent_id": binding.get("agent_id") or "unknown_worker",
                "operation": binding.get("operation") or task.get("operation") or "unknown",
                "query": binding.get("instruction") or task.get("instructions") or "",
                "timestamp_start": _timestamp(binding.get("started_at")),
                "timestamp_end": _timestamp(binding.get("completed_at")),
                "duration_seconds": float(binding.get("duration_seconds", 0.0) or 0.0),
                "input_artifact_paths": input_paths,
                "input_artifact_hashes": {
                    artifacts[item]["location"]: artifacts[item].get("content_hash")
                    for item in input_ids
                    if item in artifacts and artifacts[item].get("location")
                },
                "parameters": dict(task.get("parameters", {})),
                "status": status,
                "error_code": error.get("code"),
                "error_message": error.get("message"),
                "output_artifact_paths": output_paths,
                "output_artifact_hashes": {
                    artifacts[item]["location"]: artifacts[item].get("content_hash")
                    for item in output_ids
                    if item in artifacts and artifacts[item].get("location")
                },
                "summary": " ".join(binding.get("messages", [])),
                "raw_response": {
                    "binding_id": binding_id,
                    "task_id": binding.get("task_id"),
                    "attempt": binding.get("attempt"),
                },
            }
        )
    return events
