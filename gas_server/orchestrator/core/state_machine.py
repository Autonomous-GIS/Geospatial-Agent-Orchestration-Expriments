"""Deterministically enforced task-state transitions."""

from __future__ import annotations

from gas_server.orchestrator.core.models import TaskState, TaskStatus, WorkflowState


class IllegalTaskTransition(ValueError):
    pass


LEGAL_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PROPOSED: {TaskStatus.BLOCKED, TaskStatus.READY},
    TaskStatus.BLOCKED: {
        TaskStatus.READY,
        TaskStatus.WAITING_FOR_HUMAN,
        TaskStatus.SUPERSEDED,
    },
    TaskStatus.READY: {TaskStatus.BINDING},
    TaskStatus.BINDING: {TaskStatus.BOUND, TaskStatus.FAILED},
    TaskStatus.BOUND: {TaskStatus.RUNNING, TaskStatus.VALIDATING},
    TaskStatus.RUNNING: {TaskStatus.PROVISIONAL_SUCCESS, TaskStatus.FAILED},
    TaskStatus.PROVISIONAL_SUCCESS: {TaskStatus.VALIDATING},
    TaskStatus.VALIDATING: {
        TaskStatus.SUCCESSFUL,
        TaskStatus.REJECTED,
        TaskStatus.WAITING_FOR_HUMAN,
    },
    TaskStatus.REJECTED: {
        TaskStatus.READY,
        TaskStatus.REPAIRING,
        TaskStatus.WAITING_FOR_HUMAN,
        TaskStatus.FAILED,
    },
    TaskStatus.REPAIRING: {
        TaskStatus.READY,
        TaskStatus.BLOCKED,
        TaskStatus.SUPERSEDED,
        TaskStatus.WAITING_FOR_HUMAN,
    },
    TaskStatus.FAILED: {
        TaskStatus.READY,
        TaskStatus.REPAIRING,
        TaskStatus.WAITING_FOR_HUMAN,
    },
    TaskStatus.WAITING_FOR_HUMAN: {TaskStatus.READY, TaskStatus.BLOCKED},
    TaskStatus.SUCCESSFUL: {TaskStatus.SUPERSEDED},
    TaskStatus.SUPERSEDED: set(),
}


def transition_task(
    state: WorkflowState,
    task: TaskState,
    next_status: TaskStatus,
    *,
    actor: str,
    reason: str,
) -> None:
    allowed = LEGAL_TRANSITIONS[task.status]
    if next_status not in allowed:
        raise IllegalTaskTransition(
            f"Illegal task transition {task.task_id}: {task.status.value} -> {next_status.value}"
        )
    previous = task.status
    task.status = next_status
    state.record_event(
        "task_transition",
        actor=actor,
        task_id=task.task_id,
        reason=reason,
        metadata={"from": previous.value, "to": next_status.value},
    )
