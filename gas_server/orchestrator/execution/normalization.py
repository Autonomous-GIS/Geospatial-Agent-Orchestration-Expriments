"""Normalize legacy flat and canonical GAS worker responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from gas_server.orchestrator.core.models import StrictModel, StructuredError


class NormalizedArtifactDescriptor(StrictModel):
    location: str = Field(min_length=1)
    role: str | None = None
    format: str | None = None
    media_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedWorkerResult(StrictModel):
    status: Literal["successful", "failed", "precondition_blocked"]
    invocation_id: str | None = None
    artifacts: list[NormalizedArtifactDescriptor] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    structured_error: StructuredError | None = None
    raw_response_reference: str | None = None
    timing: dict[str, float] = Field(default_factory=dict)


def _artifact_descriptor(item: Any) -> NormalizedArtifactDescriptor:
    if isinstance(item, str):
        return NormalizedArtifactDescriptor(
            location=item,
            format=Path(item).suffix.lower().lstrip(".") or None,
        )
    if not isinstance(item, dict):
        raise ValueError("Worker artifact descriptors must be strings or objects.")
    location = (
        item.get("local_path")
        or item.get("path")
        or item.get("file_path")
        or item.get("url")
    )
    if not location:
        raise ValueError("Worker artifact descriptor has no usable location.")
    return NormalizedArtifactDescriptor(
        location=str(location),
        role=item.get("role"),
        format=item.get("format") or Path(str(location)).suffix.lower().lstrip(".") or None,
        media_type=item.get("mime_type") or item.get("media_type"),
        metadata={
            key: value
            for key, value in item.items()
            if key
            not in {"local_path", "path", "file_path", "url", "role", "format", "mime_type", "media_type"}
        },
    )


def normalize_worker_response(response: dict[str, Any]) -> NormalizedWorkerResult:
    """Accept deterministic-worker and canonical GAS response shapes."""

    if not isinstance(response, dict):
        raise ValueError("Worker response must be an object.")

    canonical_task = response.get("task") if isinstance(response.get("task"), dict) else {}
    canonical_outputs = response.get("outputs") if isinstance(response.get("outputs"), dict) else {}
    error_value = response.get("error")
    if not error_value and canonical_task.get("status") in {"failed", "rejected", "canceled"}:
        error_value = canonical_task.get("error") or canonical_task.get("message") or "Worker failed."

    raw_artifacts = canonical_outputs.get("artifacts")
    if raw_artifacts is None:
        raw_artifacts = response.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise ValueError("Worker artifacts must be a list.")

    messages: list[str] = []
    for candidate in (
        response.get("summary"),
        response.get("text"),
        canonical_outputs.get("text"),
    ):
        if candidate:
            messages.append(str(candidate))

    if error_value:
        text = str(error_value)
        code, _, message = text.partition(":")
        if not message:
            code, message = "E_WORKER_FAILURE", text
        code = {
            "E_GRID_MISALIGNMENT": "E_GRID_MISALIGNED",
        }.get(code.strip(), code.strip())
        return NormalizedWorkerResult(
            status="failed",
            invocation_id=str(canonical_task.get("id") or "") or None,
            messages=messages,
            structured_error=StructuredError(
                code=code,
                message=message.strip(),
                retryable=code in {"E_TIMEOUT", "E_SERVICE_UNAVAILABLE"},
            ),
        )

    status = str(canonical_task.get("status") or "successful").lower()
    if status not in {"successful", "succeeded", "completed"}:
        return NormalizedWorkerResult(
            status="failed",
            invocation_id=str(canonical_task.get("id") or "") or None,
            messages=messages,
            structured_error=StructuredError(
                code="E_WORKER_STATUS",
                message=f"Worker returned non-success status {status!r}.",
            ),
        )

    return NormalizedWorkerResult(
        status="successful",
        invocation_id=str(canonical_task.get("id") or "") or None,
        artifacts=[_artifact_descriptor(item) for item in raw_artifacts],
        messages=messages,
    )
