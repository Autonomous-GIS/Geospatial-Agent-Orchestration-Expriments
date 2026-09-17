"""Task-graph validation, scheduling, and atomic versioned patches."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import Field, field_validator

from gas_server.orchestrator.core.models import (
    ArtifactRoleRequirement,
    PlanVersion,
    StructuredError,
    StrictModel,
    TaskState,
    TaskStatus,
    WorkflowState,
)
from gas_server.orchestrator.requirements.derivation import derive_operation_requirements


class GraphValidationError(ValueError):
    pass


class TaskUpdate(StrictModel):
    """Strict partial update merged into an existing task by task-map key."""

    task_id: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    purpose: str | None = Field(default=None, min_length=1)
    operation: str | None = Field(default=None, min_length=1)
    instructions: str | None = Field(default=None, min_length=1)
    parameters: dict[str, Any] | None = None
    required_capabilities: list[str] | None = None
    depends_on: list[str] | None = None
    input_requirements: list[ArtifactRoleRequirement] | None = None
    expected_outputs: list[ArtifactRoleRequirement] | None = None
    requirement_ids: list[str] | None = None
    status: TaskStatus | None = None
    attempts: int | None = Field(default=None, ge=0)
    active_binding_id: str | None = None
    output_artifact_ids: list[str] | None = None
    last_error: StructuredError | None = None
    created_in_plan_version: int | None = Field(default=None, ge=1)
    superseded_in_plan_version: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] | None = None


class GraphPatch(StrictModel):
    base_plan_version: int = Field(ge=1)
    rationale: str = Field(min_length=1)
    insert_tasks: list[TaskState] = Field(default_factory=list)
    remove_task_ids: list[str] = Field(default_factory=list)
    supersede_task_ids: list[str] = Field(default_factory=list)
    update_tasks: dict[str, TaskUpdate] = Field(default_factory=dict)
    add_dependencies: list[tuple[str, str]] = Field(default_factory=list)
    remove_dependencies: list[tuple[str, str]] = Field(default_factory=list)
    reuse_artifact_ids: list[str] = Field(default_factory=list)
    reset_task_ids: list[str] = Field(default_factory=list)
    commitment_revisions: list[dict] = Field(default_factory=list)

    @field_validator("update_tasks", mode="before")
    @classmethod
    def _normalize_full_task_updates(cls, value):
        if not isinstance(value, dict):
            return value
        return {
            key: item.model_dump(mode="python") if isinstance(item, TaskState) else item
            for key, item in value.items()
        }

    def changes_structure(self) -> bool:
        return bool(
            self.insert_tasks
            or self.remove_task_ids
            or self.supersede_task_ids
            or self.update_tasks
            or self.add_dependencies
            or self.remove_dependencies
            or self.commitment_revisions
        )


def derive_edges(tasks: dict[str, TaskState]) -> list[tuple[str, str]]:
    return sorted(
        (dependency_id, task.task_id)
        for task in tasks.values()
        for dependency_id in task.depends_on
    )


def topological_order(tasks: dict[str, TaskState]) -> list[str]:
    """Validate references and return a deterministic Kahn topological order."""

    if not tasks:
        raise GraphValidationError("A workflow must contain at least one task.")
    task_ids = set(tasks)
    incoming: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {task_id: set() for task_id in tasks}
    for task_id, task in tasks.items():
        if task_id != task.task_id:
            raise GraphValidationError(f"Task key {task_id!r} differs from task_id.")
        missing = set(task.depends_on) - task_ids
        if missing:
            raise GraphValidationError(
                f"Task {task_id!r} references unknown dependencies: {sorted(missing)}"
            )
        incoming[task_id] = set(task.depends_on)
        for parent in task.depends_on:
            children[parent].add(task_id)

    ready = sorted(task_id for task_id, parents in incoming.items() if not parents)
    ordered: list[str] = []
    while ready:
        task_id = ready.pop(0)
        ordered.append(task_id)
        for child in sorted(children[task_id]):
            incoming[child].discard(task_id)
            if not incoming[child] and child not in ordered and child not in ready:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(tasks):
        cyclic = sorted(set(tasks) - set(ordered))
        raise GraphValidationError(f"Task graph contains a cycle involving: {cyclic}")
    return ordered


def validate_task_graph(state: WorkflowState) -> None:
    topological_order(state.tasks)
    expected = derive_edges(state.tasks)
    if sorted(state.edges) != expected:
        raise GraphValidationError("Explicit edges must exactly match task dependencies.")

    for task in state.tasks.values():
        requirement_ids = set(task.requirement_ids)
        missing_requirements = requirement_ids - set(state.requirements)
        if missing_requirements:
            raise GraphValidationError(
                f"Task {task.task_id!r} references unknown requirements: "
                f"{sorted(missing_requirements)}"
            )
        foreign_requirements = {
            requirement_id
            for requirement_id in requirement_ids
            if state.requirements[requirement_id].task_id
            not in {None, task.task_id}
        }
        if foreign_requirements:
            raise GraphValidationError(
                f"Task {task.task_id!r} references requirements owned by another "
                f"task: {sorted(foreign_requirements)}"
            )
        initial_roles = {
            artifact.role
            for artifact in state.artifacts.values()
            if artifact.producer_task_id is None
            and artifact.validation_status.value == "pass"
        }
        dependency_roles = {
            output.role
            for dependency_id in task.depends_on
            for output in state.tasks[dependency_id].expected_outputs
        }
        missing_roles = {
            requirement.role
            for requirement in task.input_requirements
            if requirement.required
            and requirement.role not in initial_roles | dependency_roles
        }
        if missing_roles:
            raise GraphValidationError(
                f"Task {task.task_id!r} requires artifact roles with no initial or "
                f"direct-dependency producer: {sorted(missing_roles)}"
            )


def ready_task_ids(state: WorkflowState) -> list[str]:
    successful = {TaskStatus.SUCCESSFUL, TaskStatus.SUPERSEDED}
    order = topological_order(state.tasks)
    return [
        task_id
        for task_id in order
        if state.tasks[task_id].status in {
            TaskStatus.PROPOSED,
            TaskStatus.BLOCKED,
            TaskStatus.READY,
        }
        and all(state.tasks[parent].status in successful for parent in state.tasks[task_id].depends_on)
    ]


def downstream_task_ids(state: WorkflowState, changed_task_ids: set[str]) -> set[str]:
    downstream: set[str] = set()
    frontier = list(changed_task_ids)
    while frontier:
        parent = frontier.pop(0)
        children = [child for source, child in state.edges if source == parent]
        for child in children:
            if child not in downstream:
                downstream.add(child)
                frontier.append(child)
    return downstream


def apply_graph_patch(
    state: WorkflowState,
    patch: GraphPatch,
    *,
    allow_structural_patching: bool,
) -> WorkflowState:
    """Apply a patch to a clone, validate it fully, then return the candidate."""

    if patch.base_plan_version != state.current_plan_version:
        raise GraphValidationError(
            f"Stale patch for plan {patch.base_plan_version}; current plan is "
            f"{state.current_plan_version}."
        )
    if patch.changes_structure() and not allow_structural_patching:
        raise GraphValidationError("Structural graph mutation is disabled by this feature profile.")

    candidate = state.model_copy(deep=True)
    original_ids = set(candidate.tasks)

    unknown_removed = set(patch.remove_task_ids) - original_ids
    unknown_superseded = set(patch.supersede_task_ids) - original_ids
    if unknown_removed or unknown_superseded:
        raise GraphValidationError(
            f"Patch references unknown tasks: {sorted(unknown_removed | unknown_superseded)}"
        )
    inserted_ids = [task.task_id for task in patch.insert_tasks]
    if len(inserted_ids) != len(set(inserted_ids)) or set(inserted_ids) & original_ids:
        raise GraphValidationError("Inserted task IDs must be unique and previously unused.")

    for task_id in patch.remove_task_ids:
        del candidate.tasks[task_id]
    for task_id in patch.supersede_task_ids:
        candidate.tasks[task_id].status = TaskStatus.SUPERSEDED
        candidate.tasks[task_id].superseded_in_plan_version = state.current_plan_version + 1
    for task_id, replacement in patch.update_tasks.items():
        if task_id not in candidate.tasks:
            raise GraphValidationError("Updated tasks must reference an existing task ID.")
        if replacement.task_id is not None and replacement.task_id != task_id:
            raise GraphValidationError("Updated tasks must retain their task-map key ID.")
        changes = replacement.model_dump(mode="python", exclude_unset=True)
        if changes.get("task_id") is None:
            changes.pop("task_id", None)
        original = candidate.tasks[task_id]
        merged = original.model_dump(mode="python")
        merged.update(changes)
        merged["task_id"] = task_id
        updated = TaskState.model_validate(merged)
        updated.requirement_ids = list(
            dict.fromkeys(
                original.requirement_ids + updated.requirement_ids
            )
        )
        if updated.status in {TaskStatus.REJECTED, TaskStatus.FAILED}:
            updated.status = TaskStatus.BLOCKED
            updated.active_binding_id = None
            updated.output_artifact_ids = []
            updated.last_error = None
        candidate.tasks[task_id] = updated
    for task in patch.insert_tasks:
        inserted = task.model_copy(deep=True)
        inserted.created_in_plan_version = state.current_plan_version + 1
        candidate.tasks[inserted.task_id] = inserted

    next_version = state.current_plan_version + 1
    for task_id in set(inserted_ids) | set(patch.update_tasks):
        task = candidate.tasks[task_id]
        referenced = [
            candidate.requirements[requirement_id]
            for requirement_id in task.requirement_ids
            if requirement_id in candidate.requirements
        ]
        for requirement in derive_operation_requirements(task):
            already_present = any(
                item.requirement_type == requirement.requirement_type
                and item.artifact_roles == requirement.artifact_roles
                and item.task_id == task.task_id
                for item in referenced
            )
            if already_present:
                continue
            requirement.created_in_plan_version = next_version
            candidate.requirements[requirement.requirement_id] = requirement
            task.requirement_ids.append(requirement.requirement_id)
            referenced.append(requirement)

    for parent, child in patch.remove_dependencies:
        if child not in candidate.tasks:
            raise GraphValidationError(f"Cannot edit dependencies of unknown task {child!r}.")
        candidate.tasks[child].depends_on = [
            item for item in candidate.tasks[child].depends_on if item != parent
        ]
    for parent, child in patch.add_dependencies:
        if parent not in candidate.tasks or child not in candidate.tasks:
            raise GraphValidationError("Added dependency endpoints must both exist.")
        if parent not in candidate.tasks[child].depends_on:
            candidate.tasks[child].depends_on.append(parent)

    removed = set(patch.remove_task_ids)
    for task in candidate.tasks.values():
        dangling = removed & set(task.depends_on)
        if dangling:
            raise GraphValidationError(
                f"Task {task.task_id!r} still depends on removed tasks: {sorted(dangling)}"
            )

    candidate.edges = derive_edges(candidate.tasks)
    validate_task_graph(candidate)

    affected = (
        set(patch.remove_task_ids)
        | set(patch.supersede_task_ids)
        | set(patch.update_tasks)
        | set(inserted_ids)
    )
    stale = downstream_task_ids(candidate, affected) | set(patch.reset_task_ids)
    for task_id in stale:
        task = candidate.tasks.get(task_id)
        if task and task.status not in {TaskStatus.SUPERSEDED, TaskStatus.PROPOSED}:
            task.status = TaskStatus.BLOCKED
            task.output_artifact_ids = []
            task.active_binding_id = None
    reusable = set(patch.reuse_artifact_ids)
    for artifact in candidate.artifacts.values():
        if artifact.producer_task_id in stale and artifact.artifact_id not in reusable:
            artifact.validation_status = "pending"

    candidate.current_plan_version = next_version
    candidate.plan_versions.append(
        PlanVersion(
            version=next_version,
            parent_version=state.current_plan_version,
            reason=patch.rationale,
            task_ids=sorted(candidate.tasks),
            edges=deepcopy(candidate.edges),
        )
    )
    candidate.record_event(
        "graph_patch_committed",
        actor="graph_controller",
        reason=patch.rationale,
        metadata={
            "inserted": inserted_ids,
            "removed": patch.remove_task_ids,
            "superseded": patch.supersede_task_ids,
            "stale": sorted(stale),
        },
    )
    return candidate
