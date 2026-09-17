"""Central budget reservation and exhaustion behavior."""

from __future__ import annotations

from gas_server.orchestrator.core.models import TerminalState, WorkflowState


class BudgetExhausted(RuntimeError):
    pass


def _exhaust(state: WorkflowState, reason: str) -> None:
    state.terminal_state = TerminalState.BUDGET_EXHAUSTED
    state.terminal_reason = reason
    state.record_event("budget_exhausted", actor="budget_manager", reason=reason)
    raise BudgetExhausted(reason)


def reserve_worker_call(state: WorkflowState) -> int:
    if state.budgets.worker_calls >= state.budgets.max_worker_calls:
        _exhaust(state, "Maximum worker-call budget reached.")
    state.budgets.worker_calls += 1
    return state.budgets.worker_calls


def reserve_model_call(state: WorkflowState, estimated_tokens: int = 0) -> int:
    if state.budgets.total_model_calls >= state.budgets.max_total_model_calls:
        _exhaust(state, "Maximum model-call budget reached.")
    if state.budgets.total_tokens + max(0, estimated_tokens) > state.budgets.max_total_tokens:
        _exhaust(state, "Maximum model-token budget reached.")
    state.budgets.total_model_calls += 1
    state.budgets.total_tokens += max(0, estimated_tokens)
    return state.budgets.total_model_calls


def reserve_local_recovery(state: WorkflowState) -> int:
    if state.budgets.local_recoveries >= state.budgets.max_local_recoveries:
        _exhaust(state, "Maximum local-recovery budget reached.")
    state.budgets.local_recoveries += 1
    return state.budgets.local_recoveries


def reserve_replan(state: WorkflowState) -> int:
    if state.budgets.replans >= state.budgets.max_replans:
        _exhaust(state, "Maximum structural-replan budget reached.")
    state.budgets.replans += 1
    return state.budgets.replans
