"""The single authoritative progressive workflow execution loop."""

from __future__ import annotations

from time import monotonic
from typing import Protocol
from uuid import uuid4

from gas_server.orchestrator.artifacts.registry import ArtifactRegistry
from gas_server.orchestrator.artifacts.evidence import EvidenceEngine, InvariantEngine
from gas_server.orchestrator.binding.binder import Binder, BindingError
from gas_server.orchestrator.components.replanner import ReplannerComponent
from gas_server.orchestrator.config import OrchestratorFeatures
from gas_server.orchestrator.core.completion import finalize_if_complete
from gas_server.orchestrator.core.graph import derive_edges, ready_task_ids, validate_task_graph
from gas_server.orchestrator.core.models import (
    ArtifactState,
    QCDecision,
    HumanClarificationState,
    StructuredError,
    TaskState,
    TaskStatus,
    TerminalState,
    ValidationStatus,
    WorkflowState,
)
from gas_server.orchestrator.core.state_machine import transition_task
from gas_server.orchestrator.execution.executor import WorkerExecutionError, WorkerExecutor
from gas_server.orchestrator.execution.normalization import NormalizedWorkerResult
from gas_server.orchestrator.persistence.store import WorkflowStore
from gas_server.orchestrator.recovery.local import LocalRecoveryError, LocalRecoveryManager


class PlannerComponent(Protocol):
    def plan(self, state: WorkflowState) -> WorkflowState: ...


class TransitionReviewer(Protocol):
    def review(
        self,
        state: WorkflowState,
        task: TaskState,
        artifacts: list[ArtifactState],
        result: NormalizedWorkerResult,
    ) -> QCDecision: ...


class LoopController:
    """Schedule, execute, observe, review, persist, and repeat.

    Recovery and structural replanning are attached in later gated phases.  A
    reviewer is mandatory so worker success can never become accepted success
    without an explicit validation decision.
    """

    def __init__(
        self,
        *,
        store: WorkflowStore,
        binder: Binder,
        executor: WorkerExecutor,
        reviewer: TransitionReviewer,
        planner: PlannerComponent | None = None,
        artifact_registry: ArtifactRegistry | None = None,
        evidence_engine: EvidenceEngine | None = None,
        invariant_engine: InvariantEngine | None = None,
        local_recovery: LocalRecoveryManager | None = None,
        replanner: ReplannerComponent | None = None,
        max_loop_iterations: int = 100,
    ):
        self.store = store
        self.binder = binder
        self.executor = executor
        self.reviewer = reviewer
        self.planner = planner
        self.artifact_registry = artifact_registry or ArtifactRegistry()
        self.evidence_engine = evidence_engine or EvidenceEngine()
        self.invariant_engine = invariant_engine or InvariantEngine()
        self.local_recovery = local_recovery
        self.replanner = replanner
        self.max_loop_iterations = max_loop_iterations

    def _save(self, state: WorkflowState) -> WorkflowState:
        return self.store.save(state, expected_version=state.state_version)

    def _transition(
        self,
        state: WorkflowState,
        task_id: str,
        status: TaskStatus,
        reason: str,
    ) -> WorkflowState:
        transition_task(
            state,
            state.tasks[task_id],
            status,
            actor="loop_controller",
            reason=reason,
        )
        return self._save(state)

    def run(self, workflow_id: str) -> WorkflowState:
        state = self.store.load(workflow_id)
        owner_id = f"loop-{uuid4()}"
        lease = self.store.acquire_execution_lease(
            workflow_id,
            owner_id,
            ttl_seconds=max(30, int(state.budgets.max_runtime_seconds) + 30),
        )
        started = monotonic()
        try:
            if not state.tasks:
                if self.planner is None:
                    raise ValueError("An unplanned workflow requires a Planner component.")
                state = self.planner.plan(state)
                if state.current_plan_version < 1:
                    raise ValueError("Planner must commit plan version 1.")
                state.edges = derive_edges(state.tasks)
                validate_task_graph(state)
                state = self._save(state)
            else:
                if not state.edges:
                    state.edges = derive_edges(state.tasks)
                validate_task_graph(state)

            for _ in range(self.max_loop_iterations):
                if state.terminal_state is not None:
                    return state
                if monotonic() - started > state.budgets.max_runtime_seconds:
                    state.terminal_state = TerminalState.BUDGET_EXHAUSTED
                    state.terminal_reason = "Workflow runtime budget expired."
                    state.record_event(
                        "budget_exhausted",
                        actor="loop_controller",
                        reason=state.terminal_reason,
                    )
                    return self._save(state)

                if finalize_if_complete(state):
                    return self._save(state)

                ready = ready_task_ids(state)
                if not ready:
                    state.terminal_state = TerminalState.FAILED_UNRECOVERABLE
                    state.terminal_reason = "No task is ready and completion requirements are unmet."
                    state.record_event(
                        "workflow_deadlock",
                        actor="loop_controller",
                        reason=state.terminal_reason,
                    )
                    return self._save(state)

                task_id = ready[0]
                current = state.tasks[task_id].status
                if current in {TaskStatus.PROPOSED, TaskStatus.BLOCKED}:
                    state = self._transition(
                        state, task_id, TaskStatus.READY, "Dependencies are accepted."
                    )
                state = self._transition(state, task_id, TaskStatus.BINDING, "Binding task.")
                try:
                    binding = self.binder.bind(state, state.tasks[task_id])
                except BindingError as exc:
                    state.tasks[task_id].last_error = {
                        "code": "E_BINDING",
                        "message": str(exc),
                        "retryable": False,
                        "details": {},
                    }
                    state = self._transition(
                        state, task_id, TaskStatus.FAILED, "Worker binding failed."
                    )
                    state.terminal_state = TerminalState.FAILED_UNRECOVERABLE
                    state.terminal_reason = str(exc)
                    return self._save(state)
                state = self._transition(state, task_id, TaskStatus.BOUND, "Worker bound.")
                features = OrchestratorFeatures(**state.feature_config)
                binding = state.bindings[state.tasks[task_id].active_binding_id]
                requirements = [
                    state.requirements[item]
                    for item in state.tasks[task_id].requirement_ids
                    if item in state.requirements
                ]
                if features.gis_finding_routing:
                    preflight_role_map = {
                        item.role: state.artifacts[item.artifact_id]
                        for item in binding.input_bindings
                        if item.artifact_id in state.artifacts
                    }
                    evaluable = self.invariant_engine.evaluable_requirements(
                        state,
                        requirements,
                        preflight_role_map,
                    )
                    preflight_violations = self.invariant_engine.evaluate(
                        state,
                        task_id,
                        evaluable,
                        preflight_role_map,
                    )
                    blocking_preconditions = [
                        item
                        for item in preflight_violations
                        if item.status == "active" and item.blocking
                    ]
                    state.record_event(
                        "pre_execution_requirements_evaluated",
                        actor="invariant_engine",
                        task_id=task_id,
                        reason=(
                            "Known blocking preconditions failed."
                            if blocking_preconditions
                            else "All currently evaluable preconditions are satisfied."
                        ),
                        metadata={
                            "evaluated_requirement_ids": [
                                item.requirement_id for item in evaluable
                            ],
                            "blocking_violation_ids": [
                                item.violation_id for item in blocking_preconditions
                            ],
                        },
                    )
                    state = self._save(state)
                    if blocking_preconditions:
                        state = self._transition(
                            state,
                            task_id,
                            TaskStatus.VALIDATING,
                            "Reviewing known blocking preconditions before execution.",
                        )
                        precondition_result = NormalizedWorkerResult(
                            status="precondition_blocked",
                            structured_error=StructuredError(
                                code="E_PRECONDITION_BLOCKED",
                                message=(
                                    "One or more deterministic task preconditions "
                                    "are not satisfied by the bound inputs."
                                ),
                                retryable=False,
                                details={
                                    "violation_ids": [
                                        item.violation_id
                                        for item in blocking_preconditions
                                    ]
                                },
                            ),
                        )
                        binding = state.bindings[
                            state.tasks[task_id].active_binding_id
                        ]
                        binding.status = "precondition_blocked"
                        binding.structured_error = precondition_result.structured_error
                        input_artifacts = list(preflight_role_map.values())
                        if features.qc_mode != "none":
                            decision = self.reviewer.review(
                                state,
                                state.tasks[task_id],
                                input_artifacts,
                                precondition_result,
                            )
                        elif features.structural_patching and self.replanner is not None:
                            decision = QCDecision(
                                task_id=task_id,
                                verdict="BLOCK",
                                reason="A deterministic execution precondition failed.",
                                action_class="STRUCTURAL_REPLAN",
                            )
                        else:
                            decision = QCDecision(
                                task_id=task_id,
                                verdict="BLOCK",
                                reason="A deterministic execution precondition failed.",
                                action_class="NONE",
                            )
                        decision = self._enforce_recovery_route(
                            state, task_id, decision, features
                        )
                        state.qc_history.append(decision)
                        state = self._transition(
                            state,
                            task_id,
                            TaskStatus.REJECTED,
                            decision.reason,
                        )
                        recovered = self._recover(
                            state, task_id, decision, features
                        )
                        if recovered is not None:
                            state = self._save(recovered)
                            continue
                        state.terminal_state = TerminalState.FAILED_UNRECOVERABLE
                        state.terminal_reason = (
                            "No authorized recovery satisfied the known precondition: "
                            + decision.reason
                        )
                        return self._save(state)
                state = self._transition(state, task_id, TaskStatus.RUNNING, "Worker started.")

                binding = state.bindings[state.tasks[task_id].active_binding_id]
                try:
                    result = self.executor.execute(state, state.tasks[task_id], binding)
                except WorkerExecutionError as exc:
                    state = self._transition(
                        state, task_id, TaskStatus.FAILED, "Worker execution failed."
                    )
                    failed_requirements = [
                        state.requirements[item]
                        for item in state.tasks[task_id].requirement_ids
                        if item in state.requirements
                    ]
                    if features.gis_finding_routing:
                        failed_role_map = {
                            item.role: state.artifacts[item.artifact_id]
                            for item in binding.input_bindings
                            if item.artifact_id in state.artifacts
                        }
                        self.invariant_engine.evaluate(
                            state,
                            task_id,
                            failed_requirements,
                            failed_role_map,
                        )
                    if features.qc_mode != "none":
                        failed_binding = state.bindings.get(
                            state.tasks[task_id].active_binding_id or ""
                        )
                        input_artifacts = (
                            [
                                state.artifacts[item.artifact_id]
                                for item in failed_binding.input_bindings
                                if item.artifact_id in state.artifacts
                            ]
                            if failed_binding
                            else []
                        )
                        decision = self.reviewer.review(
                            state, state.tasks[task_id], input_artifacts, exc.result
                        )
                    elif features.structural_patching and self.replanner is not None:
                        decision = QCDecision(
                            task_id=task_id,
                            verdict="BLOCK",
                            reason=f"Explicit worker failure: {exc}",
                            action_class="STRUCTURAL_REPLAN",
                        )
                    else:
                        decision = QCDecision(
                            task_id=task_id,
                            verdict="BLOCK",
                            reason=f"Explicit worker failure: {exc}",
                            action_class="NONE",
                        )
                    decision = self._enforce_recovery_route(
                        state, task_id, decision, features
                    )
                    state.qc_history.append(decision)
                    recovered = self._recover(state, task_id, decision, features)
                    if recovered is not None:
                        state = self._save(recovered)
                        continue
                    state.terminal_state = TerminalState.FAILED_UNRECOVERABLE
                    state.terminal_reason = decision.reason
                    return self._save(state)

                state = self._transition(
                    state,
                    task_id,
                    TaskStatus.PROVISIONAL_SUCCESS,
                    "Worker returned a provisional result.",
                )
                binding = state.bindings[state.tasks[task_id].active_binding_id]
                artifacts = self.artifact_registry.register(
                    state, state.tasks[task_id], binding, result
                )
                self.evidence_engine.inspect_artifacts(state, artifacts)
                role_map = {
                    item.role: state.artifacts[item.artifact_id]
                    for item in binding.input_bindings
                }
                role_map.update({item.role: item for item in artifacts})
                requirements = [
                    state.requirements[item]
                    for item in state.tasks[task_id].requirement_ids
                    if item in state.requirements
                ]
                if features.gis_finding_routing:
                    self.invariant_engine.evaluate(
                        state,
                        task_id,
                        requirements,
                        role_map,
                    )
                    produced_ids = {item.artifact_id for item in artifacts}
                    for downstream in state.tasks.values():
                        if (
                            task_id not in downstream.depends_on
                            or downstream.status
                            in {TaskStatus.SUCCESSFUL, TaskStatus.SUPERSEDED}
                        ):
                            continue
                        available = [
                            artifact
                            for artifact in state.artifacts.values()
                            if (
                                artifact.producer_task_id is None
                                or artifact.producer_task_id in downstream.depends_on
                            )
                            and (
                                artifact.validation_status is ValidationStatus.PASS
                                or artifact.artifact_id in produced_ids
                            )
                        ]
                        downstream_role_map = {
                            artifact.role: artifact
                            for artifact in sorted(
                                available, key=lambda item: item.created_at
                            )
                        }
                        downstream_requirements = [
                            state.requirements[item]
                            for item in downstream.requirement_ids
                            if item in state.requirements
                        ]
                        early_violations = self.invariant_engine.evaluate(
                            state,
                            downstream.task_id,
                            downstream_requirements,
                            downstream_role_map,
                        )
                        if early_violations:
                            state.record_event(
                                "downstream_requirement_pre_evaluated",
                                actor="invariant_engine",
                                task_id=downstream.task_id,
                                reason=(
                                    "New producer evidence established a downstream "
                                    "task precondition violation."
                                ),
                                metadata={
                                    "producer_task_id": task_id,
                                    "violation_ids": [
                                        item.violation_id for item in early_violations
                                    ],
                                },
                            )
                else:
                    # In generic/C4 paths, successful worker-local operation
                    # contracts are recorded without synthesizing hidden GIS QC.
                    for requirement in requirements:
                        if requirement.provenance.value in {
                            "OPERATION_CONTRACT",
                            "WORKER_CONTRACT",
                        }:
                            requirement.status = "satisfied"
                state = self._save(state)
                state = self._transition(
                    state, task_id, TaskStatus.VALIDATING, "Reviewing runtime evidence."
                )
                decision = self.reviewer.review(state, state.tasks[task_id], artifacts, result)
                decision = self._enforce_recovery_route(
                    state, task_id, decision, features
                )
                state.qc_history.append(decision)
                if decision.verdict == "PASS":
                    self._apply_schema_mappings(state, decision)
                    for artifact in artifacts:
                        state.artifacts[artifact.artifact_id].validation_status = ValidationStatus.PASS
                    state = self._transition(
                        state, task_id, TaskStatus.SUCCESSFUL, decision.reason
                    )
                    continue

                for artifact in artifacts:
                    state.artifacts[artifact.artifact_id].validation_status = ValidationStatus.FAIL
                state = self._transition(state, task_id, TaskStatus.REJECTED, decision.reason)
                recovered = self._recover(state, task_id, decision, features)
                if recovered is not None:
                    state = self._save(recovered)
                    continue
                state.terminal_state = TerminalState.FAILED_UNRECOVERABLE
                state.terminal_reason = "No authorized recovery succeeded: " + decision.reason
                return self._save(state)

            state.terminal_state = TerminalState.BUDGET_EXHAUSTED
            state.terminal_reason = "Maximum controller loop iterations reached."
            return self._save(state)
        finally:
            self.store.release_execution_lease(lease)

    def _recover(
        self,
        state: WorkflowState,
        task_id: str,
        decision: QCDecision,
        features: OrchestratorFeatures,
    ) -> WorkflowState | None:
        effective_action = decision.effective_action or decision.action_class
        if effective_action == "LOCAL_RECOVERY" and features.local_recovery:
            if self.local_recovery is None:
                return None
            try:
                return self.local_recovery.apply(state, task_id, decision)
            except LocalRecoveryError as exc:
                state.record_event(
                    "local_recovery_rejected",
                    actor="loop_controller",
                    task_id=task_id,
                    reason=str(exc),
                )
                return None
        if effective_action == "STRUCTURAL_REPLAN":
            if self.replanner is None or not features.structural_patching:
                return None
            return self.replanner.repair(
                state,
                decision,
                allow_structural_patching=features.structural_patching,
            )
        if effective_action == "HUMAN_CLARIFICATION":
            task = state.tasks[task_id]
            raw_question = decision.recovery_parameters.get("question")
            question = (
                raw_question.strip()
                if isinstance(raw_question, str) and raw_question.strip()
                else "Please provide the missing information required to continue."
            )
            raw_answer_schema = decision.recovery_parameters.get("answer_schema")
            answer_schema = (
                raw_answer_schema
                if isinstance(raw_answer_schema, dict)
                else {"type": "string", "minLength": 1}
            )
            if question != raw_question or answer_schema is not raw_answer_schema:
                state.record_event(
                    "qc_recovery_parameters_normalized",
                    actor="loop_controller",
                    task_id=task_id,
                    reason=(
                        "Malformed or missing human-clarification parameters were "
                        "normalized to the durable pause contract."
                    ),
                )
            transition_task(
                state,
                task,
                TaskStatus.WAITING_FOR_HUMAN,
                actor="loop_controller",
                reason=decision.reason,
            )
            state.human_clarification = HumanClarificationState(
                question=question,
                answer_schema=answer_schema,
                affected_task_id=task_id,
                requirement_ids=list(task.requirement_ids),
                requested_at_state_version=state.state_version,
                requested_at_plan_version=state.current_plan_version,
                reason=decision.reason,
            )
            state.terminal_state = TerminalState.HUMAN_INTERVENTION_REQUIRED
            state.terminal_reason = decision.reason
            return state
        return None

    def _enforce_recovery_route(
        self,
        state: WorkflowState,
        task_id: str,
        decision: QCDecision,
        features: OrchestratorFeatures,
    ) -> QCDecision:
        """Resolve controller action without rewriting the QC component's proposal."""
        if decision.proposed_verdict is None:
            decision.proposed_verdict = decision.verdict
        if decision.effective_action is None:
            decision.effective_action = decision.action_class
        if decision.effective_action_source is None:
            decision.effective_action_source = (
                "qc_component" if features.qc_mode != "none" else "condition_policy"
            )
        blockers = [
            item
            for item in state.violations.values()
            if item.task_id == task_id
            and item.status == "active"
            and item.blocking
            and item.deterministic
        ]
        if decision.verdict == "PASS" and blockers:
            decision.verdict = "BLOCK"
            decision.reason = (
                "QC PASS conflicted with objective blocking deterministic findings."
            )
            decision.forced_by_invariant = True
            decision.acknowledged_finding_ids = sorted(
                set(decision.acknowledged_finding_ids)
                | {item.violation_id for item in blockers}
            )
            state.record_event(
                "qc_conflict_with_invariant",
                actor="loop_controller",
                task_id=task_id,
                reason=decision.reason,
                metadata={
                    "qc_verdict": decision.proposed_verdict,
                    "effective_verdict": decision.verdict,
                    "qc_action": decision.action_class,
                    "violation_ids": [item.violation_id for item in blockers],
                },
            )
        missing_crs_blockers = [item for item in blockers if item.code == "MISSING_CRS"]
        asserted_crs = next(
            (
                item
                for item in state.commitments.values()
                if item.active
                and item.key == "user_asserted_crs"
                and task_id in item.affected_task_ids
            ),
            None,
        )
        if (
            decision.verdict == "BLOCK"
            and missing_crs_blockers
            and asserted_crs is not None
            and features.structural_patching
            and self.replanner is not None
            and decision.action_class != "STRUCTURAL_REPLAN"
        ):
            decision.effective_action = "STRUCTURAL_REPLAN"
            decision.effective_action_source = "deterministic_recovery_policy"
            decision.acknowledged_finding_ids = sorted(
                set(decision.acknowledged_finding_ids)
                | {item.violation_id for item in missing_crs_blockers}
            )
            decision.forced_by_invariant = True
            state.record_event(
                "qc_recovery_route_overridden",
                actor="loop_controller",
                task_id=task_id,
                reason=(
                    "The source CRS has now been user-asserted; executable CRS "
                    "assignment must precede the blocked analytical task."
                ),
                metadata={
                    "commitment_id": asserted_crs.commitment_id,
                    "qc_action": decision.action_class,
                    "effective_action": decision.effective_action,
                    "effective_action_source": decision.effective_action_source,
                },
            )
            return decision
        if (
            decision.verdict == "BLOCK"
            and decision.action_class == "NONE"
            and missing_crs_blockers
        ):
            decision.effective_action = "HUMAN_CLARIFICATION"
            decision.effective_action_source = "deterministic_recovery_policy"
            decision.recovery_parameters.setdefault(
                "question",
                "What source coordinate reference system should be assigned to the input data?",
            )
            decision.recovery_parameters.setdefault(
                "answer_schema", {"type": "string", "minLength": 1}
            )
            decision.acknowledged_finding_ids = sorted(
                set(decision.acknowledged_finding_ids)
                | {item.violation_id for item in missing_crs_blockers}
            )
            decision.forced_by_invariant = True
            state.record_event(
                "qc_recovery_route_overridden",
                actor="loop_controller",
                task_id=task_id,
                reason=(
                    "A deterministic missing-CRS blocker requires human "
                    "clarification rather than terminal action NONE."
                ),
                metadata={
                    "violation_ids": [
                        item.violation_id for item in missing_crs_blockers
                    ],
                    "qc_action": decision.action_class,
                    "effective_action": decision.effective_action,
                    "effective_action_source": decision.effective_action_source,
                },
            )
            return decision
        if (
            decision.verdict != "BLOCK"
            or decision.action_class == "STRUCTURAL_REPLAN"
            or not features.structural_patching
            or self.replanner is None
        ):
            return decision
        repairable_blockers = [
            item
            for item in blockers
            if item.computationally_resolvable is True and item.code != "MISSING_CRS"
        ]
        if not repairable_blockers:
            return decision
        decision.effective_action = "STRUCTURAL_REPLAN"
        decision.effective_action_source = "deterministic_recovery_policy"
        decision.acknowledged_finding_ids = sorted(
            set(decision.acknowledged_finding_ids)
            | {item.violation_id for item in repairable_blockers}
        )
        decision.forced_by_invariant = True
        state.record_event(
            "qc_recovery_route_overridden",
            actor="loop_controller",
            task_id=task_id,
            reason=(
                "A deterministic computationally resolvable blocker requires "
                "structural recovery rather than the QC component's proposed "
                f"action {decision.action_class}."
            ),
            metadata={
                "violation_ids": [item.violation_id for item in repairable_blockers],
                "qc_action": decision.action_class,
                "effective_action": decision.effective_action,
                "effective_action_source": decision.effective_action_source,
            },
        )
        return decision

    @staticmethod
    def _apply_schema_mappings(state: WorkflowState, decision: QCDecision) -> None:
        """Persist only schema mappings grounded in inspected artifact fields."""

        mappings = decision.recovery_parameters.get("schema_mappings", [])
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            artifact_id = str(mapping.get("artifact_id") or "")
            actual_field = str(mapping.get("actual_field") or "")
            semantic_name = str(mapping.get("semantic_name") or "")
            artifact = state.artifacts.get(artifact_id)
            if not artifact or not actual_field or not semantic_name:
                continue
            grounded = False
            for evidence_id in artifact.evidence_ids:
                evidence = state.evidence.get(evidence_id)
                schema = (evidence.values.get("schema") if evidence else None) or {}
                if actual_field in schema:
                    grounded = True
                    break
            if not grounded:
                state.record_event(
                    "schema_mapping_rejected",
                    actor="loop_controller",
                    reason=f"Field {actual_field!r} was not present in runtime schema evidence.",
                )
                continue
            key = f"{artifact_id}:{semantic_name}"
            normalized_mapping = dict(mapping)
            targets = mapping.get("downstream_parameters", [])
            if isinstance(targets, dict):
                targets = [targets]
                normalized_mapping["downstream_parameters"] = targets
            state.schema_mappings[key] = normalized_mapping
            for target in targets:
                if not isinstance(target, dict):
                    continue
                target_task = state.tasks.get(str(target.get("task_id") or ""))
                parameter_name = str(target.get("parameter") or "")
                if target_task and parameter_name and target_task.status not in {
                    TaskStatus.SUCCESSFUL,
                    TaskStatus.SUPERSEDED,
                }:
                    target_task.parameters[parameter_name] = actual_field
            state.record_event(
                "schema_mapping_grounded",
                actor="loop_controller",
                reason=f"Mapped {semantic_name!r} to runtime field {actual_field!r}.",
                metadata={"artifact_id": artifact_id, "mapping_key": key},
            )
