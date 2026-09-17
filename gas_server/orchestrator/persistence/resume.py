"""Validate clarification answers and resume the same durable workflow."""

from __future__ import annotations

from gas_server.orchestrator.core.models import (
    AnalyticalCommitment,
    EvidenceRecord,
    TaskStatus,
    TerminalState,
)
from gas_server.orchestrator.core.state_machine import transition_task
from gas_server.orchestrator.persistence.store import StateVersionConflict, WorkflowStore


class ClarificationError(ValueError):
    pass


def _validate_answer(answer, schema: dict):
    expected_type = schema.get("type")
    if expected_type == "string":
        if not isinstance(answer, str) or len(answer) < int(schema.get("minLength", 0)):
            raise ClarificationError("Clarification answer must be a non-empty string.")
    if "enum" in schema and answer not in schema["enum"]:
        raise ClarificationError("Clarification answer is not an allowed value.")


class ResumeService:
    def __init__(self, store: WorkflowStore):
        self.store = store

    def answer(
        self,
        *,
        workflow_id: str,
        expected_state_version: int,
        clarification_id: str,
        answer,
    ):
        state = self.store.load(workflow_id)
        if state.state_version != expected_state_version:
            raise StateVersionConflict(
                f"Expected state version {expected_state_version}; current version is "
                f"{state.state_version}."
            )
        clarification = state.human_clarification
        if not clarification or clarification.clarification_id != clarification_id:
            raise ClarificationError("Clarification ID does not match the active pause.")
        if clarification.answered:
            raise ClarificationError("Clarification has already been answered.")
        _validate_answer(answer, clarification.answer_schema)
        clarification.answer = answer
        clarification.answered = True
        evidence = EvidenceRecord(
            evidence_type="human_clarification",
            subject_ids=[
                item
                for item in (clarification.affected_task_id, clarification.artifact_id)
                if item
            ],
            values={"answer": answer, "clarification_id": clarification_id},
            source="user_assertion",
            inspector_version="not_applicable",
            complete=True,
        )
        state.evidence[evidence.evidence_id] = evidence
        if "crs" in (clarification.reason + " " + clarification.question).lower():
            commitment = AnalyticalCommitment(
                key="user_asserted_crs",
                value=answer,
                rationale=clarification.reason,
                evidence_source=evidence.evidence_id,
                created_by="human_clarification",
                plan_version=state.current_plan_version,
                affected_task_ids=[clarification.affected_task_id]
                if clarification.affected_task_id
                else [],
            )
            state.commitments[commitment.commitment_id] = commitment
        task_id = clarification.affected_task_id
        if task_id:
            transition_task(
                state,
                state.tasks[task_id],
                TaskStatus.READY,
                actor="resume_service",
                reason="Validated human clarification received.",
            )
        state.terminal_state = None
        state.terminal_reason = None
        state.record_event(
            "workflow_resumed",
            actor="resume_service",
            task_id=task_id,
            reason="Validated clarification stored as user-asserted evidence.",
        )
        return self.store.save(state, expected_version=expected_state_version)
