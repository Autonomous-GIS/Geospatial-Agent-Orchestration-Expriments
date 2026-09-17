"""Strict, serializable models for the authoritative orchestration state."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gas_server.orchestrator.config import STATE_SCHEMA_VERSION


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TaskStatus(StrEnum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"
    READY = "ready"
    BINDING = "binding"
    BOUND = "bound"
    RUNNING = "running"
    PROVISIONAL_SUCCESS = "provisional_success"
    VALIDATING = "validating"
    SUCCESSFUL = "successful"
    REJECTED = "rejected"
    REPAIRING = "repairing"
    FAILED = "failed"
    WAITING_FOR_HUMAN = "waiting_for_human"
    SUPERSEDED = "superseded"


class ValidationStatus(StrEnum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class TerminalState(StrEnum):
    COMPLETED = "completed"
    HUMAN_INTERVENTION_REQUIRED = "human_intervention_required"
    FAILED_UNRECOVERABLE = "failed_unrecoverable"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    INVALID_STATE = "invalid_state"


class RequirementProvenance(StrEnum):
    USER_EXPLICIT = "USER_EXPLICIT"
    OPERATION_CONTRACT = "OPERATION_CONTRACT"
    WORKER_CONTRACT = "WORKER_CONTRACT"
    PLANNER_INFERRED = "PLANNER_INFERRED"
    COMMITMENT_DERIVED = "COMMITMENT_DERIVED"
    DOWNSTREAM_CONSUMER = "DOWNSTREAM_CONSUMER"


class StructuredError(StrictModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactRoleRequirement(StrictModel):
    role: str = Field(min_length=1)
    data_model: str | None = None
    formats: list[str] = Field(default_factory=list)
    required: bool = True


class ArtifactBinding(StrictModel):
    role: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    source_task_id: str | None = None
    binding_order: int = Field(default=0, ge=0)


class TaskState(StrictModel):
    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    input_requirements: list[ArtifactRoleRequirement] = Field(default_factory=list)
    expected_outputs: list[ArtifactRoleRequirement] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PROPOSED
    attempts: int = Field(default=0, ge=0)
    active_binding_id: str | None = None
    output_artifact_ids: list[str] = Field(default_factory=list)
    last_error: StructuredError | None = None
    created_in_plan_version: int = Field(default=1, ge=1)
    superseded_in_plan_version: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> "TaskState":
        if self.task_id in self.depends_on:
            raise ValueError("A task cannot depend on itself.")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("Task dependencies must be unique.")
        input_roles = [item.role for item in self.input_requirements]
        output_roles = [item.role for item in self.expected_outputs]
        if len(input_roles) != len(set(input_roles)):
            raise ValueError("Task input roles must be unique.")
        if len(output_roles) != len(set(output_roles)):
            raise ValueError("Task output roles must be unique.")
        return self


class TaskBindingState(StrictModel):
    binding_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    input_bindings: list[ArtifactBinding] = Field(default_factory=list)
    agent_profile_snapshot: dict[str, Any] = Field(default_factory=dict)
    endpoint: str | None = None
    attempt: int = Field(default=1, ge=1)
    invocation_id: str | None = None
    status: str = "active"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = Field(default=0.0, ge=0)
    output_artifact_ids: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    structured_error: StructuredError | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactState(StrictModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    location: str = Field(min_length=1)
    content_hash: str | None = None
    format: str | None = None
    media_type: str | None = None
    data_model: str | None = None
    producer_task_id: str | None = None
    producer_binding_id: str | None = None
    role: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactDerivation(StrictModel):
    derivation_id: str = Field(default_factory=lambda: str(uuid4()))
    output_artifact_id: str = Field(min_length=1)
    source_artifact_ids: list[str] = Field(default_factory=list)
    producer_task_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    semantic_effect: str
    evidence_snapshot_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class Requirement(StrictModel):
    requirement_id: str = Field(default_factory=lambda: str(uuid4()))
    requirement_type: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    task_id: str | None = None
    artifact_roles: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    blocking: bool = True
    provenance: RequirementProvenance
    created_by: str = Field(min_length=1)
    created_in_plan_version: int = Field(default=1, ge=1)
    status: str = "pending"
    satisfied_by_evidence_ids: list[str] = Field(default_factory=list)
    superseded_by: str | None = None


class EvidenceRecord(StrictModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_type: str = Field(min_length=1)
    subject_ids: list[str] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(min_length=1)
    inspector_version: str = Field(min_length=1)
    observed_at: datetime = Field(default_factory=utc_now)
    complete: bool = True
    limitations: list[str] = Field(default_factory=list)


class ViolationRecord(StrictModel):
    violation_id: str = Field(default_factory=lambda: str(uuid4()))
    code: str = Field(min_length=1)
    layer: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    artifact_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    blocking: bool = True
    deterministic: bool = True
    computationally_resolvable: bool | None = None
    status: str = "active"
    created_in_plan_version: int = Field(default=1, ge=1)
    resolved_by_evidence_ids: list[str] = Field(default_factory=list)
    resolved_in_plan_version: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class AnalyticalCommitment(StrictModel):
    commitment_id: str = Field(default_factory=lambda: str(uuid4()))
    key: str = Field(min_length=1)
    value: Any
    rationale: str = Field(min_length=1)
    evidence_source: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    plan_version: int = Field(ge=1)
    active: bool = True
    disclosed: bool = False
    affected_task_ids: list[str] = Field(default_factory=list)
    revision_history: list[dict[str, Any]] = Field(default_factory=list)


class ExecutionEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    state_version: int = Field(ge=0)
    plan_version: int = Field(ge=0)
    task_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class QCDecision(StrictModel):
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    verdict: Literal["PASS", "BLOCK"]
    proposed_verdict: Literal["PASS", "BLOCK"] | None = None
    reason: str
    acknowledged_finding_ids: list[str] = Field(default_factory=list)
    action_class: Literal[
        "NONE", "LOCAL_RECOVERY", "STRUCTURAL_REPLAN", "HUMAN_CLARIFICATION"
    ] = "NONE"
    effective_action: Literal[
        "NONE", "LOCAL_RECOVERY", "STRUCTURAL_REPLAN", "HUMAN_CLARIFICATION"
    ] | None = None
    effective_action_source: Literal[
        "qc_component",
        "deterministic_invariant_guardrail",
        "deterministic_recovery_policy",
        "condition_policy",
    ] | None = None
    allowed_recovery_types: list[str] = Field(default_factory=list)
    recovery_parameters: dict[str, Any] = Field(default_factory=dict)
    preserve_completed_work: bool = True
    forced_by_invariant: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class ReplanDecision(StrictModel):
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    base_plan_version: int = Field(ge=1)
    rationale: str
    patch: dict[str, Any]
    accepted: bool
    created_at: datetime = Field(default_factory=utc_now)


class ComponentCallRecord(StrictModel):
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    component: Literal["planner", "qc", "replanner"]
    prompt_hash: str
    tool_schema_hash: str
    model: str
    provider: str
    attempts: int = Field(default=1, ge=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    disposition: str
    created_at: datetime = Field(default_factory=utc_now)


class PlanVersion(StrictModel):
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    reason: str
    task_ids: list[str]
    edges: list[tuple[str, str]]
    created_at: datetime = Field(default_factory=utc_now)


class HumanClarificationState(StrictModel):
    clarification_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer_schema: dict[str, Any]
    affected_task_id: str | None = None
    artifact_id: str | None = None
    requirement_ids: list[str] = Field(default_factory=list)
    requested_at_state_version: int = Field(ge=0)
    requested_at_plan_version: int = Field(ge=0)
    reason: str
    answered: bool = False
    answer: Any = None


class WorkflowBudgets(StrictModel):
    max_total_model_calls: int = Field(default=24, ge=0)
    max_total_tokens: int = Field(default=120_000, ge=0)
    max_worker_calls: int = Field(default=30, ge=1)
    max_local_recoveries: int = Field(default=6, ge=0)
    max_replans: int = Field(default=3, ge=0)
    max_runtime_seconds: float = Field(default=900.0, gt=0)
    planner_retry_limit: int = Field(default=2, ge=0)
    qc_retry_limit: int = Field(default=1, ge=0)
    replanner_retry_limit: int = Field(default=1, ge=0)
    total_model_calls: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    worker_calls: int = Field(default=0, ge=0)
    local_recoveries: int = Field(default=0, ge=0)
    replans: int = Field(default=0, ge=0)


class WorkflowState(StrictModel):
    schema_version: str = STATE_SCHEMA_VERSION
    workflow_id: str = Field(default_factory=lambda: str(uuid4()))
    state_version: int = Field(default=0, ge=0)
    user_goal: str = Field(min_length=1)
    feature_profile: str = "C2"
    feature_config: dict[str, Any] = Field(default_factory=dict)
    current_plan_version: int = Field(default=0, ge=0)
    tasks: dict[str, TaskState] = Field(default_factory=dict)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    bindings: dict[str, TaskBindingState] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactState] = Field(default_factory=dict)
    derivations: dict[str, ArtifactDerivation] = Field(default_factory=dict)
    requirements: dict[str, Requirement] = Field(default_factory=dict)
    commitments: dict[str, AnalyticalCommitment] = Field(default_factory=dict)
    schema_mappings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence: dict[str, EvidenceRecord] = Field(default_factory=dict)
    violations: dict[str, ViolationRecord] = Field(default_factory=dict)
    execution_history: list[ExecutionEvent] = Field(default_factory=list)
    qc_history: list[QCDecision] = Field(default_factory=list)
    replan_history: list[ReplanDecision] = Field(default_factory=list)
    component_calls: list[ComponentCallRecord] = Field(default_factory=list)
    plan_versions: list[PlanVersion] = Field(default_factory=list)
    budgets: WorkflowBudgets = Field(default_factory=WorkflowBudgets)
    terminal_state: TerminalState | None = None
    terminal_reason: str | None = None
    human_clarification: HumanClarificationState | None = None
    final_artifact_ids: list[str] = Field(default_factory=list)
    required_terminal_roles: list[str] = Field(default_factory=list)
    final_disclosure: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_identity_and_references(self) -> "WorkflowState":
        if any(key != value.task_id for key, value in self.tasks.items()):
            raise ValueError("Task dictionary keys must equal task_id.")
        if any(key != value.artifact_id for key, value in self.artifacts.items()):
            raise ValueError("Artifact dictionary keys must equal artifact_id.")
        task_ids = set(self.tasks)
        for task in self.tasks.values():
            missing = set(task.depends_on) - task_ids
            if missing:
                raise ValueError(f"Task {task.task_id!r} has unknown dependencies: {sorted(missing)}")
        return self

    def record_event(
        self,
        event_type: str,
        *,
        actor: str,
        task_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        event = ExecutionEvent(
            event_type=event_type,
            actor=actor,
            state_version=self.state_version,
            plan_version=self.current_plan_version,
            task_id=task_id,
            reason=reason,
            metadata=metadata or {},
        )
        self.execution_history.append(event)
        self.updated_at = utc_now()
        return event
