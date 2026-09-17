"""Deterministic objective-completion predicate."""

from __future__ import annotations

from gas_server.orchestrator.core.graph import validate_task_graph
from gas_server.orchestrator.core.models import (
    RequirementProvenance,
    TaskStatus,
    TerminalState,
    ValidationStatus,
    WorkflowState,
)


def completion_failures(state: WorkflowState) -> list[str]:
    failures: list[str] = []
    try:
        validate_task_graph(state)
    except Exception as exc:
        failures.append(f"invalid_graph:{exc}")

    active_tasks = [
        task for task in state.tasks.values() if task.status is not TaskStatus.SUPERSEDED
    ]
    if any(task.status is not TaskStatus.SUCCESSFUL for task in active_tasks):
        failures.append("active_tasks_not_successful")

    gis_finding_routing = bool(state.feature_config.get("gis_finding_routing", True))
    blocking_requirements = [
        requirement
        for requirement in state.requirements.values()
        if requirement.blocking and not requirement.superseded_by
        and (
            gis_finding_routing
            or requirement.provenance
            in {
                RequirementProvenance.USER_EXPLICIT,
                RequirementProvenance.OPERATION_CONTRACT,
                RequirementProvenance.WORKER_CONTRACT,
            }
        )
    ]
    def _latest_inspection_schema(artifact) -> set[str]:
        for evidence_id in reversed(list(getattr(artifact, "evidence_ids", []) or [])):
            evidence = state.evidence.get(evidence_id)
            if evidence is None or evidence.evidence_type != "artifact_inspection":
                continue
            schema = (evidence.values or {}).get("schema") or {}
            if isinstance(schema, dict):
                return set(str(key) for key in schema.keys())
        return set()

    def _requirement_satisfied(requirement) -> bool:
        if requirement.status == "satisfied":
            return True
        # The evidence engine can attach satisfying evidence before the status
        # field is materialized.  Completion should not deadlock when a blocking
        # requirement has already been objectively satisfied by accepted evidence.
        if bool(getattr(requirement, "satisfied_by_evidence_ids", []) or []):
            return True
        if str(getattr(requirement, "requirement_type", "")) == "required_fields":
            params = dict(getattr(requirement, "parameters", {}) or {})
            required = {str(item) for item in (params.get("fields") or []) if str(item).strip()}
            if not required:
                return True
            schemas: list[set[str]] = []
            roles = set(getattr(requirement, "artifact_roles", []) or [])
            for artifact in state.artifacts.values():
                if artifact.validation_status is not ValidationStatus.PASS:
                    continue
                if roles and artifact.role not in roles:
                    continue
                schemas.append(_latest_inspection_schema(artifact))
            union = set().union(*schemas) if schemas else set()
            return not bool(required - union)
        return False

    # Pending requirements are not automatically failures at completion time:
    # some are planner/QC bookkeeping requirements that may be unevaluable from
    # the current task's role bindings even though the terminal artifact has
    # already passed and no active deterministic violation remains.  Active
    # violations and missing terminal roles are enforced separately below.
    if any(
        str(getattr(requirement, "status", "")) == "violated"
        and not _requirement_satisfied(requirement)
        for requirement in blocking_requirements
    ):
        failures.append("blocking_requirements_unsatisfied")

    if any(
        violation.blocking and violation.status == "active"
        for violation in state.violations.values()
    ):
        failures.append("active_blocking_violations")

    accepted_roles = {
        artifact.role
        for artifact in state.artifacts.values()
        if artifact.validation_status is ValidationStatus.PASS
    }
    missing_roles = sorted(set(state.required_terminal_roles) - accepted_roles)
    if missing_roles:
        failures.append(f"missing_terminal_roles:{','.join(missing_roles)}")

    if state.human_clarification and not state.human_clarification.answered:
        failures.append("waiting_for_human")
    return failures


def finalize_if_complete(state: WorkflowState) -> bool:
    if completion_failures(state):
        return False
    non_leaf_ids = {source for source, _ in state.edges}
    leaf_ids = set(state.tasks) - non_leaf_ids
    state.final_artifact_ids = sorted(
        artifact.artifact_id
        for artifact in state.artifacts.values()
        if artifact.producer_task_id in leaf_ids
        and artifact.validation_status is ValidationStatus.PASS
    )
    commitment_lines = [
        f"{item.key}={item.value}"
        for item in state.commitments.values()
        if item.active
    ]
    state.final_disclosure = (
        "Workflow completed with accepted terminal artifacts."
        + (" Active commitments: " + "; ".join(commitment_lines) if commitment_lines else "")
    )
    for commitment in state.commitments.values():
        if commitment.active:
            commitment.disclosed = True
    state.terminal_state = TerminalState.COMPLETED
    state.terminal_reason = "All active tasks, blocking requirements, and terminal outputs are satisfied."
    state.record_event(
        "workflow_completed",
        actor="completion_policy",
        reason=state.terminal_reason,
        metadata={"final_artifact_ids": state.final_artifact_ids},
    )
    return True
