"""Bounded non-structural recovery operations.

The public recovery interface is unchanged.  This implementation is deliberately
strict about *changed action*: a local recovery is only accepted when it can
actually alter the next attempt (except a genuinely retryable unchanged retry).
It also normalizes a few semantically equivalent recovery labels so QC/Replanner
wording cannot accidentally make a supported recovery look unsupported.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from gas_server.orchestrator.core.budgets import reserve_local_recovery
from gas_server.orchestrator.core.models import QCDecision, TaskStatus, ValidationStatus, WorkflowState
from gas_server.orchestrator.core.state_machine import transition_task


class LocalRecoveryError(RuntimeError):
    pass


class LocalRecoveryManager:
    ALLOWED = {
        "retry",
        "revise_invocation",
        "rebind_worker",
        "alternative_candidate",
    }

    _ALIASES = {
        "retry_changed_action": "revise_invocation",
        "changed_retry": "revise_invocation",
        "alternative_retrieval": "alternative_candidate",
        "retrieve_alternative": "alternative_candidate",
        "retrieve_another_dataset": "alternative_candidate",
        "rebind": "rebind_worker",
    }

    @classmethod
    def _canonical_recovery(cls, value: Any) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        return cls._ALIASES.get(normalized, normalized)

    @staticmethod
    def _changed_invocation_payload(decision: QCDecision) -> tuple[dict[str, Any], str | None]:
        params = dict(getattr(decision, "recovery_parameters", {}) or {})
        parameters = (
            params.get("parameters")
            or params.get("parameter_updates")
            or params.get("updated_parameters")
            or {}
        )
        if isinstance(parameters, dict):
            parameters = dict(parameters)
        else:
            parameters = {}

        # QC often reports schema-grounding repairs as evidence records instead of
        # a ready-to-apply `parameters` dict.  Convert those mappings into concrete
        # task invocation edits so LOCAL_RECOVERY actually changes the next worker
        # call instead of failing as "no applicable recovery".
        schema_mappings = params.get("schema_mappings") or params.get("field_mappings") or []
        if isinstance(schema_mappings, list):
            for mapping in schema_mappings:
                if not isinstance(mapping, dict):
                    continue
                parameter = mapping.get("parameter")
                actual_field = mapping.get("actual_field") or mapping.get("field")
                if parameter and actual_field:
                    parameters[str(parameter)] = actual_field

                downstream = mapping.get("downstream_parameters") or []
                if isinstance(downstream, dict):
                    downstream = [downstream]
                for item in downstream:
                    if not isinstance(item, dict):
                        continue
                    downstream_parameter = item.get("parameter")
                    if downstream_parameter and actual_field:
                        parameters[str(downstream_parameter)] = actual_field

        for key in (
            "vector_key", "table_key", "join_key", "field", "value_field",
            "group_by", "x_field", "y_field", "longitude_field", "latitude_field",
        ):
            if key in params and params.get(key) is not None:
                parameters[str(key)] = params[key]
        required_aliases = {
            "required_year": "year",
            "required_format": "format",
            "required_location": "location",
            "required_data_model": "data_model",
            "retrieval_request": "query",
        }
        for source, target in required_aliases.items():
            if source in params and params.get(source) is not None and target not in parameters:
                parameters[target] = params[source]
        if "required_temporal_coverage" in params and "temporal_coverage" not in parameters:
            parameters["temporal_coverage"] = params["required_temporal_coverage"]
        instructions = params.get("instructions")
        return (
            parameters,
            str(instructions) if instructions is not None and str(instructions).strip() else None,
        )

    @staticmethod
    def _alternative_ids(decision: QCDecision) -> list[str]:
        params = dict(getattr(decision, "recovery_parameters", {}) or {})
        values = (
            params.get("exclude_artifact_ids")
            or params.get("rejected_artifact_ids")
            or params.get("reject_artifact_ids")
            or params.get("exclude_artifact_id")
            or params.get("rejected_artifact_id")
            or params.get("reject_artifact_id")
            or []
        )
        if isinstance(values, str):
            values = [values]
        return [str(item) for item in values if str(item).strip()]

    @classmethod
    def _is_applicable(
        cls,
        state: WorkflowState,
        task_id: str,
        recovery: str,
        decision: QCDecision,
    ) -> bool:
        task = state.tasks[task_id]
        if recovery == "retry":
            return bool(task.last_error and task.last_error.retryable)
        if recovery == "revise_invocation":
            parameters, instructions = cls._changed_invocation_payload(decision)
            if instructions and instructions != task.instructions:
                return True
            return any(task.parameters.get(key) != value for key, value in parameters.items())
        if recovery == "rebind_worker":
            return bool(state.bindings.get(task.active_binding_id or ""))
        if recovery == "alternative_candidate":
            return bool(cls._alternative_ids(decision))
        return False

    @classmethod
    def _choose_recovery(
        cls,
        state: WorkflowState,
        task_id: str,
        decision: QCDecision,
    ) -> str:
        canonical = []
        for item in decision.allowed_recovery_types:
            recovery = cls._canonical_recovery(item)
            if recovery in cls.ALLOWED and recovery not in canonical:
                canonical.append(recovery)

        # A changed retry is sometimes emitted as revise_invocation without a concrete
        # edit.  If the worker explicitly marked the error retryable, keep ordinary
        # retry as a bounded fallback instead of failing the recovery router itself.
        if "revise_invocation" in canonical and "retry" not in canonical:
            task = state.tasks[task_id]
            if task.last_error and task.last_error.retryable:
                canonical.append("retry")

        for recovery in canonical:
            if cls._is_applicable(state, task_id, recovery, decision):
                return recovery
        if canonical:
            raise LocalRecoveryError(
                "QC authorized local recovery types, but none is currently applicable: "
                f"{canonical}."
            )
        raise LocalRecoveryError("QC did not authorize a supported local recovery.")

    def apply(
        self,
        state: WorkflowState,
        task_id: str,
        decision: QCDecision,
    ) -> WorkflowState:
        recovery = self._choose_recovery(state, task_id, decision)
        task = state.tasks[task_id]
        before = {
            "parameters": deepcopy(task.parameters),
            "instructions": task.instructions,
            "excluded": list(task.metadata.get("excluded_agent_ids", [])),
            "attempts": task.attempts,
        }

        reserve_local_recovery(state)
        if recovery == "retry":
            if not task.last_error or not task.last_error.retryable:
                raise LocalRecoveryError("Unchanged retry requires a retryable worker error.")

        elif recovery == "revise_invocation":
            parameters, instructions = self._changed_invocation_payload(decision)
            if parameters:
                task.parameters.update(parameters)
            if instructions:
                task.instructions = instructions

        elif recovery == "rebind_worker":
            binding = state.bindings.get(task.active_binding_id or "")
            if not binding:
                raise LocalRecoveryError("Rebinding requires a prior binding.")
            excluded = set(task.metadata.get("excluded_agent_ids", []))
            excluded.add(binding.agent_id)
            task.metadata["excluded_agent_ids"] = sorted(excluded)
            task.active_binding_id = None

        elif recovery == "alternative_candidate":
            rejected = self._alternative_ids(decision)
            if not rejected:
                raise LocalRecoveryError("Alternative retrieval requires rejected artifact IDs.")
            existing = set(task.parameters.get("exclude_artifact_ids", []))
            existing.update(rejected)

            # A rejected retrieval candidate must not remain an accepted artifact in
            # authoritative state.  Keep it for provenance, but make it ineligible for
            # downstream binding/final completion and record why it was rejected.
            for artifact_id in rejected:
                artifact = state.artifacts.get(artifact_id)
                if artifact is None:
                    continue
                path = str(getattr(artifact, "path", "") or "")
                if path:
                    from pathlib import Path

                    source_path = Path(path)
                    if source_path.name:
                        existing.add(source_path.name.lower())
                    if source_path.stem:
                        existing.add(source_path.stem.lower())
                    if source_path.stem.startswith("retrieved_"):
                        parts = source_path.stem.split("_", 2)
                        if len(parts) == 3:
                            existing.add(parts[2].lower())
                source_artifact_id = (
                    (getattr(artifact, "metadata", {}) or {}).get("source_artifact_id")
                    or (getattr(artifact, "metadata", {}) or {}).get("catalog_artifact_id")
                )
                if source_artifact_id:
                    existing.add(str(source_artifact_id).lower())
                artifact.validation_status = ValidationStatus.FAIL
                artifact.metadata = dict(artifact.metadata or {})
                artifact.metadata["candidate_rejected"] = True
                artifact.metadata["candidate_rejection_reason"] = str(decision.reason or "")
            task.parameters["exclude_artifact_ids"] = sorted(existing)

            # Keep a durable explanation for retrieval workers that consume generic
            # instructions rather than only structured exclusion parameters.
            goal = str(
                (getattr(decision, "recovery_parameters", {}) or {}).get("recovery_goal")
                or ""
            ).strip()
            if goal and goal not in task.instructions:
                task.instructions = (task.instructions.rstrip() + "\nRecovery constraint: " + goal).strip()

        after = {
            "parameters": deepcopy(task.parameters),
            "instructions": task.instructions,
            "excluded": list(task.metadata.get("excluded_agent_ids", [])),
            "attempts": task.attempts,
        }
        if before == after and recovery != "retry":
            raise LocalRecoveryError("Local recovery made no relevant change.")

        task.output_artifact_ids = []
        transition_task(
            state,
            task,
            TaskStatus.READY,
            actor="local_recovery",
            reason=f"Authorized local recovery: {recovery}.",
        )
        state.record_event(
            "local_recovery_applied",
            actor="local_recovery",
            task_id=task_id,
            reason=decision.reason,
            metadata={
                "recovery_type": recovery,
                "authorized_recovery_types": list(decision.allowed_recovery_types),
            },
        )
        return state
