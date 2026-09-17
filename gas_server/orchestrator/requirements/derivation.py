"""Derive operation-contract requirements available in every condition."""

from __future__ import annotations

from gas_server.orchestrator.core.models import (
    Requirement,
    RequirementProvenance,
    TaskState,
)


SUPPORTED_REQUIREMENT_TYPES = frozenset(
    {
        "metric_distance",
        "compatible_crs",
        "aligned_raster_grid",
        "valid_geometry",
        "required_fields",
        "temporal_range",
        "spatial_coverage",
        "max_resolution",
        "format_compatible",
        "crs_present",
    }
)


def derive_operation_requirements(task: TaskState) -> list[Requirement]:
    operation = task.operation.lower()
    roles = [item.role for item in task.input_requirements]
    requirements: list[Requirement] = []
    if operation == "buffer":
        requirements.append(
            Requirement(
                requirement_type="metric_distance",
                scope="task_transition",
                task_id=task.task_id,
                artifact_roles=roles[:1],
                provenance=RequirementProvenance.OPERATION_CONTRACT,
                created_by="operation_contract_catalog",
                created_in_plan_version=task.created_in_plan_version,
            )
        )
    if operation in {"intersection", "overlay", "spatial_join", "zonal_statistics"}:
        requirements.append(
            Requirement(
                requirement_type="compatible_crs",
                scope="task_transition",
                task_id=task.task_id,
                artifact_roles=roles,
                provenance=RequirementProvenance.OPERATION_CONTRACT,
                created_by="operation_contract_catalog",
                created_in_plan_version=task.created_in_plan_version,
            )
        )
    if operation in {
        "raster_difference",
        "cellwise_difference",
        "raster_calculator",
        "difference",
        "absolute_difference",
    }:
        requirements.append(
            Requirement(
                requirement_type="aligned_raster_grid",
                scope="task_transition",
                task_id=task.task_id,
                artifact_roles=roles,
                provenance=RequirementProvenance.OPERATION_CONTRACT,
                created_by="operation_contract_catalog",
                created_in_plan_version=task.created_in_plan_version,
            )
        )
    if operation in {"intersection", "overlay", "clip", "spatial_join"}:
        geometry_roles = [
            item.role
            for item in task.input_requirements
            # Retrieval workers may advertise their output as the generic
            # ``dataset`` model even though the bound artifact is later
            # inspected as vector data.  Keep such roles in the geometry
            # contract so runtime evidence can evaluate them.  Explicit
            # raster/table inputs remain outside this requirement.
            if item.data_model in {None, "dataset", "vector"}
        ]
        requirements.append(
            Requirement(
                requirement_type="valid_geometry",
                scope="task_transition",
                task_id=task.task_id,
                artifact_roles=geometry_roles,
                provenance=RequirementProvenance.OPERATION_CONTRACT,
                created_by="operation_contract_catalog",
                created_in_plan_version=task.created_in_plan_version,
            )
        )
    return requirements
