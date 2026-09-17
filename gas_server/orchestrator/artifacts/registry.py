"""Register normalized worker artifacts without interpreting analytical fitness."""

from __future__ import annotations

import hashlib
from pathlib import Path

from gas_server.orchestrator.core.models import (
    ArtifactDerivation,
    ArtifactState,
    TaskBindingState,
    TaskState,
    WorkflowState,
)
from gas_server.orchestrator.execution.normalization import NormalizedWorkerResult


class ArtifactRegistrationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_data_model(format_name: str | None) -> str | None:
    value = (format_name or "").lower().lstrip(".")
    if value in {"gpkg", "geojson", "shp", "fgb", "kml"}:
        return "vector"
    if value in {"tif", "tiff", "geotiff", "cog"}:
        return "raster"
    if value in {"csv", "tsv", "parquet", "xlsx"}:
        return "table"
    return "file" if value else None


class ArtifactRegistry:
    def register(
        self,
        state: WorkflowState,
        task: TaskState,
        binding: TaskBindingState,
        result: NormalizedWorkerResult,
    ) -> list[ArtifactState]:
        registered: list[ArtifactState] = []
        expected_roles = [item.role for item in task.expected_outputs]
        source_ids = [item.artifact_id for item in binding.input_bindings]
        for index, descriptor in enumerate(result.artifacts):
            path = Path(descriptor.location)
            if not path.is_file() or path.is_symlink():
                raise ArtifactRegistrationError(
                    f"Worker artifact is not a readable regular file: {descriptor.location}"
                )
            role = descriptor.role
            if not role and index < len(expected_roles):
                role = expected_roles[index]
            role = role or f"output_{index + 1}"
            artifact = ArtifactState(
                location=str(path.resolve()),
                content_hash=_sha256(path),
                format=descriptor.format or path.suffix.lower().lstrip(".") or None,
                media_type=descriptor.media_type,
                data_model=_infer_data_model(descriptor.format or path.suffix),
                producer_task_id=task.task_id,
                producer_binding_id=binding.binding_id,
                role=role,
                metadata=descriptor.metadata,
            )
            state.artifacts[artifact.artifact_id] = artifact
            task.output_artifact_ids.append(artifact.artifact_id)
            binding.output_artifact_ids.append(artifact.artifact_id)
            derivation = ArtifactDerivation(
                output_artifact_id=artifact.artifact_id,
                source_artifact_ids=source_ids,
                producer_task_id=task.task_id,
                operation=task.operation,
                parameters=dict(task.parameters),
                semantic_effect=f"Produced role {role!r} using operation {task.operation!r}.",
            )
            state.derivations[derivation.derivation_id] = derivation
            state.record_event(
                "artifact_registered",
                actor="artifact_registry",
                task_id=task.task_id,
                metadata={
                    "artifact_id": artifact.artifact_id,
                    "role": role,
                    "content_hash": artifact.content_hash,
                },
            )
            registered.append(artifact)
        return registered
