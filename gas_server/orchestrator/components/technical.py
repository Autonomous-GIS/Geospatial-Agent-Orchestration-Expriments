"""Shared domain-neutral technical transition reviewer used by the QC ablation.

This reviewer intentionally contains no GIS-specific checks.  It only ensures that
technical failures receive a bounded changed-action recovery instead of being
silently routed to NONE, preserving the C4 ablation boundary.
"""

from __future__ import annotations

from gas_server.orchestrator.core.models import QCDecision


class TechnicalTransitionReviewer:
    @staticmethod
    def _error_text(result) -> str:
        error = getattr(result, "structured_error", None)
        if error is None:
            return ""
        if hasattr(error, "model_dump"):
            try:
                payload = error.model_dump(mode="json")
                return " ".join(str(value) for value in payload.values()).lower()
            except Exception:
                pass
        if isinstance(error, dict):
            return " ".join(str(value) for value in error.values()).lower()
        return str(error).lower()

    def review(self, state, task, artifacts, result):
        del state
        if result.status != "successful":
            error_text = self._error_text(result)
            unavailable_tokens = (
                "503", "service unavailable", "unavailable", "connection refused",
                "connection error", "timeout", "timed out", "temporarily unavailable",
            )
            if any(token in error_text for token in unavailable_tokens):
                return QCDecision(
                    task_id=task.task_id,
                    verdict="BLOCK",
                    reason=(
                        "Worker did not return a successful technical status and the "
                        "failure is consistent with service unavailability."
                    ),
                    action_class="LOCAL_RECOVERY",
                    allowed_recovery_types=["rebind_worker", "retry"],
                    recovery_parameters={
                        "recovery_goal": "Use an equivalent healthy worker or a changed retry."
                    },
                )
            return QCDecision(
                task_id=task.task_id,
                verdict="BLOCK",
                reason="Worker did not return a successful technical status.",
                action_class="LOCAL_RECOVERY",
                allowed_recovery_types=["revise_invocation", "retry"],
                recovery_parameters={
                    "recovery_goal": "Retry only after changing a relevant input, parameter, or invocation."
                },
            )
        expected_roles = {item.role for item in task.expected_outputs if item.required}
        actual_roles = {item.role for item in artifacts}
        missing = sorted(expected_roles - actual_roles)
        if missing:
            return QCDecision(
                task_id=task.task_id,
                verdict="BLOCK",
                reason=f"Required output roles are missing: {missing}.",
                action_class="LOCAL_RECOVERY",
                allowed_recovery_types=["revise_invocation", "retry"],
                recovery_parameters={
                    "missing_output_roles": missing,
                    "recovery_goal": "Correct the invocation so every required output role is produced."
                },
            )
        return QCDecision(
            task_id=task.task_id,
            verdict="PASS",
            reason="Shared technical response and output-role checks passed.",
        )
