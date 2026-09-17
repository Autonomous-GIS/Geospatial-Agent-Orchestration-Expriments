"""Bind task operations and artifact roles to eligible workers."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from gas_server.orchestrator.core.models import (
    ArtifactBinding,
    StrictModel,
    TaskBindingState,
    TaskState,
    ValidationStatus,
    WorkflowState,
)


class BindingError(RuntimeError):
    pass


def _canonical_format(value: str | None) -> str:
    normalized = str(value or "").lower().lstrip(".")
    aliases = {
        "tif": "geotiff",
        "tiff": "geotiff",
        "geotif": "geotiff",
        "cog": "geotiff",
        "geopackage": "gpkg",
        "json": "geojson",
    }
    return aliases.get(normalized, normalized)


def capability_supports_expected_outputs(
    profile_snapshot: dict[str, Any], task: TaskState
) -> bool:
    """Check declared task outputs against one worker's advertised outputs."""
    advertised = [
        item for item in profile_snapshot.get("outputs", []) if isinstance(item, dict)
    ]
    if not advertised or not task.expected_outputs:
        return True
    for requirement in task.expected_outputs:
        if not requirement.required:
            continue
        compatible = False
        for output in advertised:
            output_type = str(output.get("type") or "").lower()
            required_model = str(requirement.data_model or "").lower()
            if (
                required_model
                and required_model != "dataset"
                and output_type not in {"", "dataset", required_model}
            ):
                continue
            required_formats = {
                _canonical_format(value) for value in requirement.formats
            }
            output_formats = {
                _canonical_format(value) for value in output.get("formats", [])
            }
            if required_formats and output_formats and not (
                required_formats & output_formats
            ):
                continue
            compatible = True
            break
        if not compatible:
            return False
    return True


class WorkerCapability(StrictModel):
    agent_id: str
    endpoint: str | None = None
    operations: list[str] = Field(default_factory=list)
    input_data_models: list[str] = Field(default_factory=list)
    input_formats: list[str] = Field(default_factory=list)
    available: bool = True
    credential_available: bool = True
    profile_snapshot: dict[str, Any] = Field(default_factory=dict)


class Binder:
    def __init__(self, capabilities: list[WorkerCapability]):
        self.capabilities = {item.agent_id: item for item in capabilities}

    @staticmethod
    def _dependency_artifacts(state: WorkflowState, task: TaskState):
        dependency_ids = set(task.depends_on)
        return [
            artifact
            for artifact in state.artifacts.values()
            if (artifact.producer_task_id in dependency_ids or artifact.producer_task_id is None)
            and artifact.validation_status is ValidationStatus.PASS
        ]

    def bind(self, state: WorkflowState, task: TaskState) -> TaskBindingState:
        excluded = set(task.metadata.get("excluded_agent_ids", []))
        preferred = task.metadata.get("preferred_agent_id")
        candidates = []
        required_capabilities = set(task.required_capabilities)
        for capability in self.capabilities.values():
            if capability.agent_id in excluded:
                continue
            if preferred and capability.agent_id != preferred:
                continue
            if not capability.available or not capability.credential_available:
                continue
            if task.operation not in capability.operations and "*" not in capability.operations:
                continue
            supported_capabilities = set(capability.operations) | {capability.agent_id}
            if required_capabilities and not required_capabilities.issubset(
                supported_capabilities
            ):
                continue
            if not capability_supports_expected_outputs(
                capability.profile_snapshot, task
            ):
                continue
            candidates.append(capability)
        if not candidates:
            requirement = (
                f" and required capabilities {sorted(required_capabilities)!r}"
                if required_capabilities
                else ""
            )
            raise BindingError(
                f"No eligible worker supports operation {task.operation!r}{requirement}."
            )
        selected = sorted(candidates, key=lambda item: item.agent_id)[0]

        artifacts = self._dependency_artifacts(state, task)
        by_role: dict[str, list] = {}
        for artifact in artifacts:
            by_role.setdefault(artifact.role, []).append(artifact)
        bindings: list[ArtifactBinding] = []
        for order, requirement in enumerate(task.input_requirements):
            matches = by_role.get(requirement.role, [])
            if requirement.required and not matches:
                raise BindingError(
                    f"Task {task.task_id!r} is missing accepted artifact role "
                    f"{requirement.role!r}."
                )
            if matches:
                artifact = sorted(matches, key=lambda item: item.created_at)[-1]
                if (
                    requirement.data_model
                    and requirement.data_model != "dataset"
                    and artifact.data_model not in {None, requirement.data_model}
                ):
                    raise BindingError(
                        f"Role {requirement.role!r} requires data model "
                        f"{requirement.data_model!r}."
                    )
                allowed_formats = {
                    _canonical_format(item) for item in requirement.formats
                }
                if allowed_formats and _canonical_format(artifact.format) not in allowed_formats:
                    raise BindingError(
                        f"Role {requirement.role!r} requires one of {requirement.formats}."
                    )
                bindings.append(
                    ArtifactBinding(
                        role=requirement.role,
                        artifact_id=artifact.artifact_id,
                        source_task_id=artifact.producer_task_id,
                        binding_order=order,
                    )
                )

        prior_attempts = [
            binding.attempt
            for binding in state.bindings.values()
            if binding.task_id == task.task_id
        ]
        binding = TaskBindingState(
            task_id=task.task_id,
            agent_id=selected.agent_id,
            operation=task.operation,
            instruction=task.instructions,
            input_bindings=bindings,
            agent_profile_snapshot=selected.profile_snapshot,
            endpoint=selected.endpoint,
            attempt=max(prior_attempts, default=0) + 1,
        )
        state.bindings[binding.binding_id] = binding
        task.active_binding_id = binding.binding_id
        return binding
