"""Independent evidence-grounded QC component.

The LLM still interprets workflow context, but deterministic findings are treated as
hard facts.  The component also supplies a bounded recovery recommendation so a
confirmed problem cannot be acknowledged and then accidentally routed to NONE.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gas_server.orchestrator.components.common import ComponentClient, stable_hash
from gas_server.orchestrator.core.budgets import reserve_model_call
from gas_server.orchestrator.core.models import (
    ArtifactState,
    ComponentCallRecord,
    QCDecision,
    TaskState,
    ViolationRecord,
    WorkflowState,
)
from gas_server.orchestrator.execution.normalization import NormalizedWorkerResult


class QCComponent:
    component_name = "qc"

    _STRUCTURAL_RECOVERY_TYPES = {
        "METRIC_DISTANCE_UNSUPPORTED": ["reproject_artifact"],
        "CRS_INCOMPATIBLE": ["reproject_artifact"],
        "RASTER_GRID_INCOMPATIBLE": ["reproject_artifact", "align_raster_grid"],
        "INVALID_GEOMETRY": ["repair_geometry"],
        "SCHEMA_REQUIREMENT_UNRESOLVED": ["ground_runtime_schema"],
        "SCHEMA_PARAMETER_UNRESOLVED": ["ground_runtime_schema"],
        "COORDINATE_FIELDS_UNRESOLVED": ["spatialize_coordinates"],
        "TABLE_REQUIRES_SPATIALIZATION": ["spatialize_coordinates"],
        "DATA_MODEL_INCOMPATIBLE": ["convert_data_model"],
        "FORMAT_UNSUPPORTED": ["convert_artifact_format"],
        "ANALYTICAL_COMMITMENT_MISSING": ["establish_analytical_commitment"],
        "COMMITMENT_DISCLOSURE_MISSING": ["propagate_commitment_disclosure"],
        "COMMITMENT_MISSING": ["establish_required_commitment"],
    }
    _ALTERNATIVE_RETRIEVAL_CODES = {
        "TEMPORAL_REQUIREMENT_UNMET",
        "SPATIAL_COVERAGE_INSUFFICIENT",
        "RESOLUTION_REQUIREMENT_UNMET",
    }

    def __init__(self, client: ComponentClient, *, guidance_mode: str):
        if guidance_mode not in {"generic", "gis_aware"}:
            raise ValueError("QC guidance mode must be generic or gis_aware.")
        self.client = client
        self.guidance_mode = guidance_mode

    def _prompt(self) -> str:
        suffix = "gis" if self.guidance_mode == "gis_aware" else "generic"
        base = (
            Path(__file__).parent / "prompts" / f"qc_{suffix}.txt"
        ).read_text(encoding="utf-8")
        common = """

RUNTIME ENFORCEMENT ADDENDUM:
- Treat deterministic_violations as observed facts, not suggestions.
- Never return PASS for the current task while it has an active blocking deterministic violation.
- A worker failure must receive a concrete recovery class when bounded recovery is available; do not acknowledge an error and return NONE.
- Preserve valid completed work and prefer the smallest changed action that can make progress.
- When terminal_goal_guard.terminal_candidate is true, compare the ORIGINAL user_goal with the current workflow/evidence. Do not declare a terminal transition safe merely because the current DAG has no more steps. If a requested output or necessary transformation is absent, BLOCK and request structural replanning.
- Runtime schema evidence is authoritative for field names. Never invent a field that is not present in the supplied schema.
"""
        if self.guidance_mode != "gis_aware":
            return base + common
        gis = """

GIS-AWARE RECOVERY RULES:
- Missing consequential CRS with no trusted assertion/provenance requires HUMAN_CLARIFICATION; do not guess. assign_crs is allowed only to define genuinely unknown CRS metadata when the value comes from the user or trusted persisted evidence; it must never be used to relabel known coordinates instead of reprojecting them.
- Metric-distance/CRS incompatibility, invalid geometry, format conversion, raster-grid alignment, coordinate spatialization, and schema-grounding defects require a changed workflow/task structure when they cannot be fixed by revising the current invocation.
- Wrong-year, insufficient-coverage, or insufficient-resolution retrieval candidates should be rejected and another candidate requested when the retrieval task can be rerun.
- Service unavailability should first use an equivalent healthy worker when one exists.
- For multiple simultaneously observed blockers, choose a recovery that addresses all current blockers without erasing prior accepted work.
"""
        return base + common + gis

    @staticmethod
    def _status_value(task: TaskState) -> str:
        status = getattr(task, "status", "")
        return str(getattr(status, "value", status))

    @staticmethod
    def _latest_inspection_values(state: WorkflowState, artifact: ArtifactState) -> dict[str, Any]:
        for evidence_id in reversed(artifact.evidence_ids):
            if evidence_id not in state.evidence:
                continue
            record = state.evidence[evidence_id]
            if record.evidence_type == "artifact_inspection":
                return dict(record.values or {})
        return {}

    @staticmethod
    def _goal_explicitly_assigns_crs(user_goal: str, target_crs: str) -> bool:
        text = " ".join(str(user_goal or "").lower().replace("-", " ").split())
        target = str(target_crs or "").lower().strip()
        if not target or target not in str(user_goal or "").lower():
            return False
        return any(
            token in text
            for token in (
                "assign crs",
                "define crs",
                "set crs",
                "assign coordinate reference",
                "define coordinate reference",
            )
        )

    @classmethod
    def _trusted_crs_assertion(
        cls,
        state: WorkflowState,
        task: TaskState,
        target_crs: str,
    ) -> bool:
        target = str(target_crs or "").strip().lower()
        for item in state.commitments.values():
            if not getattr(item, "active", False):
                continue
            if str(getattr(item, "key", "")) != "user_asserted_crs":
                continue
            value = str(getattr(item, "value", "")).strip().lower()
            affected = set(getattr(item, "affected_task_ids", []) or [])
            if value == target and (not affected or task.task_id in affected):
                return True
        return cls._goal_explicitly_assigns_crs(state.user_goal, target_crs)

    @staticmethod
    def _task_input_artifacts(
        state: WorkflowState,
        task: TaskState,
    ) -> list[ArtifactState]:
        binding_id = getattr(task, "active_binding_id", None)
        binding = state.bindings.get(binding_id) if binding_id else None
        if binding is None:
            return []
        result: list[ArtifactState] = []
        for item in getattr(binding, "input_bindings", []) or []:
            artifact_id = getattr(item, "artifact_id", None)
            if artifact_id in state.artifacts:
                result.append(state.artifacts[artifact_id])
        return result

    @classmethod
    def _unsafe_crs_assignment(
        cls,
        state: WorkflowState,
        task: TaskState,
    ) -> dict[str, Any] | None:
        if str(task.operation or "").lower() != "assign_crs":
            return None
        target_crs = str((task.parameters or {}).get("target_crs") or "").strip()
        if not target_crs:
            return {
                "kind": "missing_target",
                "target_crs": None,
                "input_artifact_ids": [],
            }
        if cls._trusted_crs_assertion(state, task, target_crs):
            return None

        inputs = cls._task_input_artifacts(state, task)
        if not inputs:
            return None
        unknown = []
        known = []
        for artifact in inputs:
            values = cls._latest_inspection_values(state, artifact)
            observed = values.get("crs_authority") or values.get("crs")
            if observed:
                known.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "role": artifact.role,
                        "crs": str(observed),
                    }
                )
            else:
                unknown.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "role": artifact.role,
                    }
                )
        if unknown:
            return {
                "kind": "untrusted_missing_crs_assignment",
                "target_crs": target_crs,
                "unknown_inputs": unknown,
                "known_inputs": known,
            }
        if known:
            return {
                "kind": "known_crs_relabel",
                "target_crs": target_crs,
                "known_inputs": known,
                "unknown_inputs": [],
            }
        return None

    @staticmethod
    def _normalize_field_name(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    @classmethod
    def _candidate_fields(cls, requested: str, available: list[str]) -> list[str]:
        target = cls._normalize_field_name(requested)
        if not target:
            return available[:12]
        scored: list[tuple[int, str]] = []
        for field in available:
            normalized = cls._normalize_field_name(field)
            score = 0
            if normalized == target:
                score = 100
            elif target in normalized or normalized in target:
                score = 70
            else:
                target_tokens = set(re.findall(r"[a-z]+|\d+", str(requested).lower()))
                field_tokens = set(re.findall(r"[a-z]+|\d+", str(field).lower()))
                score = 10 * len(target_tokens & field_tokens)
            if score:
                scored.append((score, field))
        return [field for _, field in sorted(scored, key=lambda x: (-x[0], x[1]))[:12]]

    @staticmethod
    def _field_parameter_keys(parameters: dict[str, Any]) -> set[str]:
        keys: set[str] = set()
        for key in parameters:
            lowered = str(key).lower()
            if (
                lowered in {
                    "field", "fields", "columns", "group_by", "groupby",
                    "vector_key", "table_key", "join_key", "date_field",
                    "time_field", "value_field", "category_field",
                    "x_field", "y_field", "longitude_field", "latitude_field",
                }
                or lowered.endswith("_field")
                or lowered.endswith("_column")
                or lowered.endswith("_key")
            ):
                keys.add(str(key))
        return keys

    def _schema_grounding_targets(
        self,
        state: WorkflowState,
        tasks: list[TaskState],
    ) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        operation_defaults = {
            "attribute_join": {"vector_key", "table_key"},
            "thematic_mapping": {"field"},
        }
        for candidate_task in tasks:
            relevant_roles = {item.role for item in candidate_task.input_requirements}
            schema_by_artifact: list[dict[str, Any]] = []
            available_fields: list[str] = []
            for artifact in state.artifacts.values():
                if relevant_roles and artifact.role not in relevant_roles:
                    continue
                values = self._latest_inspection_values(state, artifact)
                schema = list((values.get("schema") or {}).keys())
                if not schema:
                    continue
                schema_by_artifact.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "role": artifact.role,
                        "fields": schema,
                    }
                )
                available_fields.extend(schema)
            available_fields = sorted(set(available_fields))
            if not available_fields:
                continue
            parameters = dict(candidate_task.parameters or {})
            parameter_keys = self._field_parameter_keys(parameters)
            parameter_keys.update(operation_defaults.get(candidate_task.operation, set()))
            for parameter in sorted(parameter_keys):
                raw_value = parameters.get(parameter)
                values_to_check = raw_value if isinstance(raw_value, list) else [raw_value]
                nonblank = [str(item).strip() for item in values_to_check if str(item or "").strip()]
                if not nonblank:
                    targets.append(
                        {
                            "task_id": candidate_task.task_id,
                            "operation": candidate_task.operation,
                            "parameter": parameter,
                            "current_value": raw_value,
                            "problem": "parameter_unset",
                            "available_schema": schema_by_artifact,
                            "candidate_fields": available_fields[:20],
                            "instruction": "Ground this parameter to an exact runtime field before execution.",
                        }
                    )
                    continue
                missing = [item for item in nonblank if item not in available_fields]
                if not missing:
                    continue
                requested = missing[0]
                targets.append(
                    {
                        "task_id": candidate_task.task_id,
                        "operation": candidate_task.operation,
                        "parameter": parameter,
                        "current_value": raw_value,
                        "problem": "planned_field_not_in_runtime_schema",
                        "missing_fields": missing,
                        "available_schema": schema_by_artifact,
                        "candidate_fields": self._candidate_fields(requested, available_fields),
                        "instruction": (
                            "The planned field is absent from runtime evidence. Replace it only with "
                            "an exact observed field justified by the user goal and schema evidence."
                        ),
                    }
                )
        return targets

    @staticmethod
    def _worker_error_payload(result: NormalizedWorkerResult) -> dict[str, Any] | None:
        error = getattr(result, "structured_error", None)
        if error is None:
            return None
        if hasattr(error, "model_dump"):
            return error.model_dump(mode="json")
        if isinstance(error, dict):
            return dict(error)
        return {"message": str(error)}

    @classmethod
    def _worker_failure_recovery(cls, result: NormalizedWorkerResult):
        if result.status == "successful":
            return None
        payload = cls._worker_error_payload(result) or {}
        text = " ".join(str(value) for value in payload.values()).lower()
        unavailable_tokens = (
            "503", "service unavailable", "unavailable", "connection refused",
            "connection error", "timeout", "timed out", "temporarily unavailable",
            "endpoint unavailable", "server error",
        )
        if any(token in text for token in unavailable_tokens):
            return (
                "LOCAL_RECOVERY",
                ["rebind_worker", "retry"],
                {"recovery_goal": "Use an equivalent healthy worker or a changed retry."},
            )
        return (
            "LOCAL_RECOVERY",
            ["revise_invocation", "retry"],
            {"recovery_goal": "Retry only after changing a relevant parameter, input, or invocation."},
        )

    @staticmethod
    def _looks_like_retrieval_task(task: TaskState) -> bool:
        operation = str(task.operation or "").lower().replace("-", "_")
        text = " ".join(
            [
                operation,
                str(getattr(task, "title", "") or "").lower(),
                str(getattr(task, "purpose", "") or "").lower(),
                str(getattr(task, "instructions", "") or "").lower(),
            ]
        )
        return any(
            token in text
            for token in (
                "retrieve", "retrieval", "dataset_search", "search_dataset",
                "fetch_dataset", "download_dataset", "select_dataset",
                "data retrieval", "find dataset",
            )
        )

    @classmethod
    def _producer_candidate_blockers(
        cls,
        task: TaskState,
        artifacts: list[ArtifactState],
        downstream_blockers: list[ViolationRecord],
    ) -> list[ViolationRecord]:
        """Return downstream candidate-fitness blockers caused by this producer output.

        Candidate selection is most effective when the retrieval task itself is rerun
        with the rejected artifact excluded.  Routing these findings only to a later
        consumer loses the ability to ask the retrieval worker for the next candidate.
        """
        if not cls._looks_like_retrieval_task(task):
            return []
        output_ids = {artifact.artifact_id for artifact in artifacts}
        if not output_ids:
            return []
        return [
            item
            for item in downstream_blockers
            if item.code in cls._ALTERNATIVE_RETRIEVAL_CODES
            and bool(output_ids & set(item.artifact_ids))
        ]

    def _deterministic_recovery(
        self,
        state: WorkflowState,
        task: TaskState,
        blockers: list[ViolationRecord],
        schema_grounding_targets: list[dict[str, Any]] | None = None,
    ):
        codes = {item.code for item in blockers}
        if not codes:
            return None

        # Preserve the experimental ablation boundary: generic QC receives factual
        # deterministic blockers but not GIS-specific repair recipes.
        if self.guidance_mode != "gis_aware":
            if all(getattr(item, "computationally_resolvable", False) for item in blockers):
                return (
                    "STRUCTURAL_REPLAN",
                    [],
                    {
                        "blocking_codes": sorted(codes),
                        "recovery_goal": "Apply a changed workflow/task action that resolves the confirmed blockers.",
                    },
                )
            return (
                "HUMAN_CLARIFICATION",
                [],
                {
                    "blocking_codes": sorted(codes),
                    "recovery_goal": "The confirmed blocker cannot be resolved safely from current evidence.",
                },
            )

        # Unknown consequential CRS is the epistemic boundary: no automatic repair
        # is allowed unless a trustworthy assertion/provenance has already resolved it.
        if "MISSING_CRS" in codes:
            return (
                "HUMAN_CLARIFICATION",
                [],
                {
                    "blocking_codes": sorted(codes),
                    "recovery_goal": "Obtain trustworthy CRS information; do not guess.",
                },
            )

        # If an unsuitable retrieval candidate is the only problem, rerun the same
        # retrieval with that concrete artifact excluded.  LocalRecoveryManager uses
        # the canonical `alternative_candidate` name.
        if codes.issubset(self._ALTERNATIVE_RETRIEVAL_CODES):
            rejected = sorted({
                artifact_id
                for item in blockers
                for artifact_id in item.artifact_ids
            })
            return (
                "LOCAL_RECOVERY",
                ["alternative_candidate"],
                {
                    "blocking_codes": sorted(codes),
                    "exclude_artifact_ids": rejected,
                    "recovery_goal": "Reject the unsuitable candidate and retrieve another candidate satisfying the observed requirement.",
                },
            )

        # Runtime schema drift can often be corrected without changing the graph.
        # Only do this deterministically when there is one unambiguous observed field.
        if codes.issubset({"SCHEMA_PARAMETER_UNRESOLVED", "SCHEMA_REQUIREMENT_UNRESOLVED"}):
            updates: dict[str, Any] = {}
            for target in schema_grounding_targets or []:
                if target.get("task_id") != task.task_id:
                    continue
                candidates = list(target.get("candidate_fields") or [])
                if len(candidates) == 1:
                    updates[str(target["parameter"])] = candidates[0]
            if updates:
                merged = dict(task.parameters or {})
                merged.update(updates)
                return (
                    "LOCAL_RECOVERY",
                    ["revise_invocation"],
                    {
                        "blocking_codes": sorted(codes),
                        "parameters": updates,
                        "recovery_goal": "Replace stale planned field names with the unique exact fields observed in runtime schema evidence.",
                    },
                )

        recovery_types: list[str] = []
        for code in sorted(codes):
            recovery_types.extend(self._STRUCTURAL_RECOVERY_TYPES.get(code, []))
        recovery_types = list(dict.fromkeys(recovery_types))
        if any(getattr(item, "computationally_resolvable", False) for item in blockers):
            return (
                "STRUCTURAL_REPLAN",
                recovery_types,
                {
                    "blocking_codes": sorted(codes),
                    "blocking_artifact_ids": sorted({
                        artifact_id
                        for item in blockers
                        for artifact_id in item.artifact_ids
                    }),
                    "recovery_goal": "Apply the smallest structural/task patch that resolves every current deterministic blocker while preserving accepted work.",
                },
            )
        return (
            "HUMAN_CLARIFICATION",
            [],
            {
                "blocking_codes": sorted(codes),
                "recovery_goal": "The blocking decision is not computationally justified from current evidence.",
            },
        )

    def _terminal_goal_guard(
        self,
        state: WorkflowState,
        task: TaskState,
        downstream_tasks: list[TaskState],
        current_artifacts: list[ArtifactState],
    ) -> dict[str, Any]:
        unresolved_blocking_requirements = [
            {
                "requirement_id": item.requirement_id,
                "requirement_type": item.requirement_type,
                "status": item.status,
            }
            for item in state.requirements.values()
            if getattr(item, "blocking", False)
            and getattr(item, "status", None) != "satisfied"
            and not getattr(item, "superseded_by", None)
        ]
        active_blocking_violations = [
            {
                "violation_id": item.violation_id,
                "task_id": item.task_id,
                "code": item.code,
            }
            for item in state.violations.values()
            if item.status == "active" and item.blocking
        ]
        incomplete_tasks = [
            {
                "task_id": item.task_id,
                "operation": item.operation,
                "status": self._status_value(item),
                "depends_on": list(item.depends_on),
            }
            for item in state.tasks.values()
            if self._status_value(item) not in {"successful", "superseded"}
            and item.task_id != task.task_id
        ]
        active_commitments = [
            {
                "key": item.key,
                "value": item.value,
                "disclosed": bool(getattr(item, "disclosed", False)),
            }
            for item in state.commitments.values()
            if item.active
        ]
        current_roles = {item.role for item in current_artifacts}
        accepted_roles = {
            artifact.role
            for artifact in state.artifacts.values()
            if str(getattr(getattr(artifact, "validation_status", None), "value", getattr(artifact, "validation_status", ""))).lower() == "pass"
        }
        required_terminal_roles = list(getattr(state, "required_terminal_roles", []) or [])
        terminal_roles_present = sorted((current_roles | accepted_roles) & set(required_terminal_roles))
        missing_terminal_roles = sorted(set(required_terminal_roles) - (current_roles | accepted_roles))
        terminal_candidate = not downstream_tasks and not incomplete_tasks
        return {
            "terminal_candidate": terminal_candidate,
            "required_terminal_roles": required_terminal_roles,
            "terminal_roles_present": terminal_roles_present,
            "missing_terminal_roles": missing_terminal_roles,
            "original_user_goal": state.user_goal,
            "current_task_id": task.task_id,
            "current_operation": task.operation,
            "workflow_operations": [
                {
                    "task_id": item.task_id,
                    "operation": item.operation,
                    "status": self._status_value(item),
                }
                for item in list(state.tasks.values())[:80]
            ],
            "artifact_roles_present": sorted({artifact.role for artifact in state.artifacts.values()}),
            "unresolved_blocking_requirements": unresolved_blocking_requirements[:50],
            "active_blocking_violations": active_blocking_violations[:50],
            "other_incomplete_tasks": incomplete_tasks[:50],
            "active_commitments": active_commitments[:30],
            "instruction": (
                "If this is a terminal candidate, verify that the produced evidence answers the original goal, "
                "not merely that the currently planned graph has ended."
            ),
        }

    def review(
        self,
        state: WorkflowState,
        task: TaskState,
        artifacts: list[ArtifactState],
        result: NormalizedWorkerResult,
    ) -> QCDecision:
        reserve_model_call(state)
        violations: list[ViolationRecord] = [
            item
            for item in state.violations.values()
            if item.task_id == task.task_id and item.status == "active"
        ]
        relevant_evidence_ids = {
            evidence_id for artifact in artifacts for evidence_id in artifact.evidence_ids
        }
        relevant_evidence_ids.update(
            evidence_id for violation in violations for evidence_id in violation.evidence_ids
        )
        prompt = self._prompt()
        requirements = [
            state.requirements[item]
            for item in task.requirement_ids
            if item in state.requirements
        ]
        downstream_tasks = [
            item
            for item in state.tasks.values()
            if task.task_id in item.depends_on
            and self._status_value(item) not in {"successful", "superseded"}
        ]
        downstream_roles = {
            requirement.role
            for downstream in downstream_tasks
            for requirement in downstream.input_requirements
        }
        for artifact in state.artifacts.values():
            if artifact.role not in downstream_roles:
                continue
            relevant_evidence_ids.update(artifact.evidence_ids)
        downstream_task_ids = {item.task_id for item in downstream_tasks}
        schema_grounding_targets = self._schema_grounding_targets(
            state, [task, *downstream_tasks]
        )
        downstream_violations = [
            item
            for item in state.violations.values()
            if item.task_id in downstream_task_ids and item.status == "active"
        ]
        blockers = [item for item in violations if item.blocking and item.deterministic]
        downstream_blockers = [
            item
            for item in downstream_violations
            if item.blocking and item.deterministic
        ]
        producer_candidate_blockers = self._producer_candidate_blockers(
            task, artifacts, downstream_blockers
        )
        deterministic_recovery = self._deterministic_recovery(
            state, task, blockers, schema_grounding_targets
        )
        worker_failure_recovery = self._worker_failure_recovery(result)
        unsafe_crs_assignment = (
            self._unsafe_crs_assignment(state, task)
            if self.guidance_mode == "gis_aware"
            else None
        )

        input_payload = {
            "user_goal": state.user_goal,
            "task": task.model_dump(mode="json"),
            "downstream_tasks": [item.model_dump(mode="json") for item in downstream_tasks],
            "requirement_status_summary": [
                {
                    "requirement_id": item.requirement_id,
                    "requirement_type": item.requirement_type,
                    "status": item.status,
                    "is_active_blocker": bool(
                        item.blocking
                        and item.status != "satisfied"
                        and not item.superseded_by
                    ),
                }
                for item in requirements
            ],
            "unresolved_requirements": [
                item.model_dump(mode="json")
                for item in requirements
                if item.status != "satisfied" and not item.superseded_by
            ],
            "runtime_evidence": [
                state.evidence[item].model_dump(mode="json")
                for item in sorted(relevant_evidence_ids)
                if item in state.evidence
            ],
            "deterministic_violations": [item.model_dump(mode="json") for item in violations],
            "downstream_deterministic_violations": [
                item.model_dump(mode="json") for item in downstream_violations
            ],
            "producer_candidate_fitness_blockers": [
                item.model_dump(mode="json") for item in producer_candidate_blockers
            ],
            "deterministic_recovery_recommendation": (
                {
                    "action_class": deterministic_recovery[0],
                    "allowed_recovery_types": deterministic_recovery[1],
                    "recovery_parameters": deterministic_recovery[2],
                }
                if deterministic_recovery
                else None
            ),
            "required_schema_grounding_targets": schema_grounding_targets,
            "terminal_goal_guard": self._terminal_goal_guard(
                state, task, downstream_tasks, artifacts
            ),
            "active_commitments": [
                item.model_dump(mode="json")
                for item in state.commitments.values()
                if item.active
                and (
                    task.task_id in item.affected_task_ids
                    or bool(downstream_task_ids & set(item.affected_task_ids))
                )
            ],
            "epistemic_crs_assignment_guard": unsafe_crs_assignment,
            "worker_status": result.status,
            "worker_error": self._worker_error_payload(result),
            "worker_failure_recovery_recommendation": (
                {
                    "action_class": worker_failure_recovery[0],
                    "allowed_recovery_types": worker_failure_recovery[1],
                    "recovery_parameters": worker_failure_recovery[2],
                }
                if worker_failure_recovery
                else None
            ),
        }
        response = self.client.complete(
            component=self.component_name,
            system_prompt=prompt,
            input_payload=input_payload,
            output_schema=QCDecision.model_json_schema(),
            retry_limit=state.budgets.qc_retry_limit,
        )
        # Controller-owned enforcement fields are not accepted from the QC LLM.
        # Strip them before Pydantic validation so a model-emitted alias or
        # out-of-enum value cannot crash an otherwise valid QC proposal.
        qc_payload = response.payload
        if isinstance(qc_payload, dict):
            qc_payload = dict(qc_payload)
            for controller_field in (
                "proposed_verdict",
                "effective_action",
                "effective_action_source",
                "forced_by_invariant",
            ):
                qc_payload.pop(controller_field, None)
        decision = QCDecision.model_validate(qc_payload)
        if decision.task_id != task.task_id:
            raise ValueError("QC decision references the wrong task.")

        # These fields describe controller enforcement, not the LLM proposal.
        decision.proposed_verdict = decision.verdict
        decision.effective_action = None
        decision.effective_action_source = None
        decision.forced_by_invariant = False

        asserted_crs_values = {
            str(item.value)
            for item in state.commitments.values()
            if item.active
            and item.key == "user_asserted_crs"
            and (
                task.task_id in item.affected_task_ids
                or bool(downstream_task_ids & set(item.affected_task_ids))
            )
        }
        inspected_output_crs = {
            str(state.evidence[evidence_id].values.get("crs"))
            for artifact in artifacts
            for evidence_id in artifact.evidence_ids
            if evidence_id in state.evidence
            and state.evidence[evidence_id].evidence_type == "artifact_inspection"
            and state.evidence[evidence_id].values.get("crs")
        }

        # Suppress stale clarification after a user assertion has already been
        # materialized and no objective blocker remains.
        if (
            decision.verdict == "BLOCK"
            and decision.action_class in {"HUMAN_CLARIFICATION", "STRUCTURAL_REPLAN"}
            and result.status == "successful"
            and not blockers
            and not downstream_blockers
            and bool(asserted_crs_values & inspected_output_crs)
        ):
            proposed = decision
            decision = QCDecision(
                task_id=task.task_id,
                verdict="PASS",
                proposed_verdict="BLOCK",
                reason=(
                    "The successful output evidence contains the user-asserted CRS "
                    "and no active deterministic blocker remains."
                ),
                action_class="NONE",
                preserve_completed_work=proposed.preserve_completed_work,
                forced_by_invariant=True,
            )
            state.record_event(
                "qc_stale_clarification_suppressed",
                actor="qc_component",
                task_id=task.task_id,
                reason=decision.reason,
                metadata={
                    "qc_action": proposed.action_class,
                    "asserted_crs_values": sorted(asserted_crs_values),
                    "inspected_output_crs": sorted(inspected_output_crs),
                },
            )

        # Epistemic CRS guard.  Even if an unsafe assign_crs worker call happened,
        # its output remains provisional and cannot be accepted.  Unknown CRS without
        # a trusted assertion must pause; relabelling known coordinates must be
        # replaced by an actual coordinate transformation when a different CRS is
        # required.
        if unsafe_crs_assignment:
            proposed = decision
            kind = unsafe_crs_assignment.get("kind")
            if kind == "known_crs_relabel":
                action_class = "STRUCTURAL_REPLAN"
                recovery_types = ["reproject_artifact"]
                recovery_parameters = {
                    "blocking_codes": ["CRS_INCOMPATIBLE"],
                    "target_crs": unsafe_crs_assignment.get("target_crs"),
                    "recovery_goal": (
                        "Replace CRS relabelling with coordinate transformation; preserve "
                        "the original coordinates only through a true reprojection."
                    ),
                }
                reason = (
                    "assign_crs was used on input coordinates whose CRS was already known. "
                    "Metadata relabelling cannot substitute for reprojection."
                )
            else:
                action_class = "HUMAN_CLARIFICATION"
                recovery_types = []
                recovery_parameters = {
                    "blocking_codes": ["MISSING_CRS"],
                    "target_crs": unsafe_crs_assignment.get("target_crs"),
                    "recovery_goal": "Obtain an explicit trustworthy source CRS; do not guess or infer it from another layer.",
                }
                reason = (
                    "assign_crs attempted to define an unknown source CRS without an "
                    "explicit user/trusted CRS assertion."
                )
            decision = QCDecision(
                task_id=task.task_id,
                verdict="BLOCK",
                proposed_verdict=proposed.verdict,
                reason=reason,
                acknowledged_finding_ids=list(proposed.acknowledged_finding_ids),
                action_class=action_class,
                allowed_recovery_types=recovery_types,
                recovery_parameters=recovery_parameters,
                preserve_completed_work=True,
                forced_by_invariant=True,
                effective_action=action_class,
                effective_action_source="deterministic_invariant_guardrail",
            )
            state.record_event(
                "qc_unsafe_crs_assignment_blocked",
                actor="qc_component",
                task_id=task.task_id,
                reason=reason,
                metadata=unsafe_crs_assignment,
            )

        # The deterministic layer is the authority for objective GIS blockers.
        # If a worker transition succeeded and neither the current task nor its
        # downstream consumers have active deterministic blockers, do not let an
        # unsupported epistemic concern from QC pause the workflow.  Terminal-goal
        # coverage is still enforced below by the deterministic terminal guard.
        if (
            decision.verdict == "BLOCK"
            and result.status == "successful"
            and not blockers
            and not downstream_blockers
            and not producer_candidate_blockers
            and not unsafe_crs_assignment
            and decision.action_class in {"HUMAN_CLARIFICATION", "STRUCTURAL_REPLAN"}
        ):
            proposed = decision
            decision = QCDecision(
                task_id=task.task_id,
                verdict="PASS",
                proposed_verdict=proposed.verdict,
                reason=(
                    "The worker transition succeeded and no active deterministic "
                    "current/downstream blocker supports pausing or replanning this task."
                ),
                acknowledged_finding_ids=list(proposed.acknowledged_finding_ids),
                action_class="NONE",
                preserve_completed_work=proposed.preserve_completed_work,
                forced_by_invariant=True,
                effective_action="NONE",
                effective_action_source="deterministic_recovery_policy",
            )
            state.record_event(
                "qc_unsupported_successful_transition_block_suppressed",
                actor="qc_component",
                task_id=task.task_id,
                reason=decision.reason,
                metadata={"qc_action": proposed.action_class, "qc_reason": proposed.reason},
            )

        # Candidate-fitness violations belong to the retrieval transition when the
        # current producer created the rejected artifact.  Rerun that producer with
        # the artifact excluded instead of waiting for a downstream analytical task
        # to fail.  This turns temporal/coverage/resolution evidence into an actual
        # alternative-candidate retrieval.
        if (
            result.status == "successful"
            and not blockers
            and producer_candidate_blockers
        ):
            proposed = decision
            rejected = sorted({
                artifact_id
                for item in producer_candidate_blockers
                for artifact_id in item.artifact_ids
                if artifact_id in {artifact.artifact_id for artifact in artifacts}
            })
            decision = QCDecision(
                task_id=task.task_id,
                verdict="BLOCK",
                proposed_verdict=proposed.verdict,
                reason=(
                    "The retrieval transition produced a candidate that violates a confirmed "
                    "downstream temporal/coverage/resolution requirement; request another candidate."
                ),
                acknowledged_finding_ids=[item.violation_id for item in producer_candidate_blockers],
                action_class="LOCAL_RECOVERY",
                allowed_recovery_types=["alternative_candidate"],
                recovery_parameters={
                    "exclude_artifact_ids": rejected,
                    "blocking_codes": sorted({item.code for item in producer_candidate_blockers}),
                    "recovery_goal": (
                        "Retrieve another candidate satisfying the explicit downstream fitness requirement; "
                        "do not return any excluded artifact."
                    ),
                },
                preserve_completed_work=True,
                forced_by_invariant=True,
                effective_action="LOCAL_RECOVERY",
                effective_action_source="deterministic_recovery_policy",
            )
            state.record_event(
                "qc_candidate_retrieval_rerouted_to_producer",
                actor="qc_component",
                task_id=task.task_id,
                reason=decision.reason,
                metadata={
                    "rejected_artifact_ids": rejected,
                    "blocking_codes": decision.recovery_parameters.get("blocking_codes", []),
                    "downstream_violation_ids": [
                        item.violation_id for item in producer_candidate_blockers
                    ],
                },
            )

        # A producer can be accepted while a downstream consumer remains blocked;
        # recovery stays attached to the task whose contract is actually violated.
        if (
            decision.verdict == "BLOCK"
            and result.status == "successful"
            and not blockers
            and downstream_blockers
            and not producer_candidate_blockers
        ):
            proposed = decision
            decision = QCDecision(
                task_id=task.task_id,
                verdict="PASS",
                proposed_verdict="BLOCK",
                reason=(
                    "The current producer transition satisfies its own contract; "
                    "the objective defect remains active on the downstream task where recovery will be routed."
                ),
                action_class="NONE",
                preserve_completed_work=proposed.preserve_completed_work,
                forced_by_invariant=True,
            )
            state.record_event(
                "qc_downstream_scope_routed",
                actor="qc_component",
                task_id=task.task_id,
                reason=decision.reason,
                metadata={
                    "qc_verdict": "BLOCK",
                    "qc_action": proposed.action_class,
                    "downstream_violation_ids": [item.violation_id for item in downstream_blockers],
                },
            )

        # Hard enforcement: a confirmed current-task blocker receives both BLOCK
        # and a concrete recovery class.  This fixes the previous failure mode in
        # which a deterministic alarm could be retained while action_class stayed NONE.
        if blockers:
            proposed_verdict = decision.verdict
            recovery = deterministic_recovery or self._deterministic_recovery(
                state, task, blockers, schema_grounding_targets
            )
            assert recovery is not None
            action_class, recovery_types, recovery_parameters = recovery
            decision.verdict = "BLOCK"
            decision.proposed_verdict = proposed_verdict
            decision.action_class = action_class
            decision.allowed_recovery_types = list(recovery_types)
            merged_parameters = dict(getattr(decision, "recovery_parameters", {}) or {})
            merged_parameters.update(recovery_parameters)
            decision.recovery_parameters = merged_parameters
            decision.acknowledged_finding_ids = [item.violation_id for item in blockers]
            decision.forced_by_invariant = True
            decision.effective_action = action_class
            decision.effective_action_source = "deterministic_invariant_guardrail"
            if proposed_verdict == "PASS" or action_class != (response.payload.get("action_class") if isinstance(response.payload, dict) else None):
                state.record_event(
                    "qc_deterministic_recovery_enforced",
                    actor="qc_component",
                    task_id=task.task_id,
                    reason="Active deterministic blockers require a concrete recovery action.",
                    metadata={
                        "blocking_codes": sorted({item.code for item in blockers}),
                        "effective_action": action_class,
                        "allowed_recovery_types": list(recovery_types),
                    },
                )

        # Generic worker failures are also not allowed to terminate with NONE when
        # bounded retry/rebind is available.
        elif worker_failure_recovery and decision.action_class == "NONE":
            action_class, recovery_types, recovery_parameters = worker_failure_recovery
            decision.verdict = "BLOCK"
            decision.action_class = action_class
            decision.allowed_recovery_types = list(recovery_types)
            decision.recovery_parameters = dict(recovery_parameters)
            decision.forced_by_invariant = True
            decision.effective_action = action_class
            decision.effective_action_source = "deterministic_recovery_policy"
            state.record_event(
                "qc_worker_failure_recovery_enforced",
                actor="qc_component",
                task_id=task.task_id,
                reason="Worker failure has a bounded changed-action recovery path.",
                metadata={"effective_action": action_class, "allowed_recovery_types": list(recovery_types)},
            )

        # A terminal graph cannot be accepted when the Planner-declared final
        # artifact contract is not satisfied.  This is deterministic and directly
        # targets false completion without prescribing one preferred workflow.
        terminal_guard = self._terminal_goal_guard(
            state, task, downstream_tasks, artifacts
        )
        if (
            terminal_guard.get("terminal_candidate")
            and terminal_guard.get("missing_terminal_roles")
            and decision.verdict == "PASS"
        ):
            proposed = decision
            decision = QCDecision(
                task_id=task.task_id,
                verdict="BLOCK",
                proposed_verdict=proposed.verdict,
                reason=(
                    "The workflow reached a terminal candidate without producing all "
                    "Planner-declared required terminal artifact roles: "
                    f"{terminal_guard['missing_terminal_roles']}."
                ),
                action_class="STRUCTURAL_REPLAN",
                allowed_recovery_types=["restore_goal_coverage"],
                recovery_parameters={
                    "missing_terminal_roles": terminal_guard["missing_terminal_roles"],
                    "recovery_goal": "Restore only the missing end-to-end work required to produce the declared terminal outputs.",
                },
                preserve_completed_work=True,
                forced_by_invariant=True,
                effective_action="STRUCTURAL_REPLAN",
                effective_action_source="deterministic_invariant_guardrail",
            )
            state.record_event(
                "qc_false_completion_guard",
                actor="qc_component",
                task_id=task.task_id,
                reason=decision.reason,
                metadata={
                    "required_terminal_roles": terminal_guard["required_terminal_roles"],
                    "missing_terminal_roles": terminal_guard["missing_terminal_roles"],
                },
            )

        state.component_calls.append(
            ComponentCallRecord(
                component="qc",
                prompt_hash=stable_hash(prompt),
                tool_schema_hash=stable_hash(QCDecision.model_json_schema()),
                model=response.model,
                provider=response.provider,
                attempts=response.attempts,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_seconds=response.duration_seconds,
                disposition=response.disposition,
            )
        )
        state.budgets.total_tokens += response.input_tokens + response.output_tokens
        return decision
