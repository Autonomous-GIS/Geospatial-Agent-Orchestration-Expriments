"""Independent minimal structural Replanner component."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from gas_server.orchestrator.binding.binder import capability_supports_expected_outputs
from gas_server.orchestrator.components.common import ComponentClient, stable_hash
from gas_server.orchestrator.core.budgets import reserve_model_call, reserve_replan
from gas_server.orchestrator.core.graph import GraphPatch, apply_graph_patch
from gas_server.orchestrator.core.models import (
    ComponentCallRecord,
    QCDecision,
    ReplanDecision,
    WorkflowState,
)


class ReplannerComponent:
    component_name = "replanner"

    def __init__(
        self,
        client: ComponentClient,
        *,
        guidance_mode: str,
        capability_snapshots: list[dict[str, Any]] | None = None,
    ):
        if guidance_mode not in {"generic", "gis_aware"}:
            raise ValueError("Replanner guidance mode must be generic or gis_aware.")
        self.client = client
        self.guidance_mode = guidance_mode
        self.capability_snapshots = list(capability_snapshots or [])

    def _prompt(self) -> str:
        suffix = "gis" if self.guidance_mode == "gis_aware" else "generic"
        prompt = (
            Path(__file__).parent / "prompts" / f"replanner_{suffix}.txt"
        ).read_text(encoding="utf-8")
        if self.guidance_mode != "gis_aware":
            return prompt
        return prompt + r"""

EVIDENCE-TO-REPAIR ADDENDUM (GIS-AWARE MODE)
- The supplied deterministic violations are facts. Repair those facts; do not invent a different problem.
- Produce the smallest patch that makes the blocked operation executable while preserving every unaffected successful task, accepted artifact, analytical commitment, and required terminal output role.
- Never repair by deleting a requested final output or shortening the original goal. A patch is invalid if it makes required_terminal_roles unreachable.
- Prefer these general GIS repair motifs when they match the observed violation and an advertised worker supports the operation:
  * metric-distance on non-metric coordinates -> insert a justified reprojection before the distance operation;
  * CRS incompatibility -> reproject one necessary input to a compatible CRS chosen from observed/downstream evidence;
  * invalid geometry -> insert geometry_repair before the geometry-consuming task;
  * coordinate table used as vector -> insert the advertised table/coordinate-to-point conversion using observed coordinate fields;
  * unsupported format/data model -> insert an advertised conversion that satisfies the blocked worker contract;
  * runtime schema parameter mismatch -> update the consuming task to an exact observed field; never invent a column;
  * raster grid incompatibility -> reproject/resample as needed so the cellwise inputs share CRS, resolution, origin/transform and dimensions required by the operation.
- Wrong temporal vintage, insufficient coverage, and insufficient resolution are usually candidate-selection defects; if QC routed them here rather than local alternative retrieval, preserve the analytical graph and modify only what is necessary to obtain/use a suitable candidate.
- Missing consequential CRS is not a structural-replan problem unless trustworthy CRS evidence/user assertion is already present. Never guess a CRS.
- CRS semantics are strict: assign_crs only defines metadata when the source CRS is unknown and a trusted user/asserted value exists. If source CRS is known and coordinates must change, use reproject. For metric-distance repair, the target must be a projected CRS with metre units.
- For a single-artifact metric-distance or invalid-geometry blocker, prefer one adapter task only. Do not build a multi-node repair subgraph when one reprojection or one geometry repair is sufficient.
- Inserted repair tasks must flow repair -> blocked consumer. They must never depend on the blocked consumer or any descendant of that consumer.
- Reuse accepted upstream artifacts and keep role identity stable across role-preserving transformations.
"""

    @staticmethod
    def _active_commitment_signature(state: WorkflowState) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for item in state.commitments.values():
            if not item.active:
                continue
            payload = item.model_dump(mode="json")
            # `disclosed` is presentation state and may legitimately advance later;
            # the analytical choice itself must not silently change during repair.
            payload.pop("disclosed", None)
            result[item.commitment_id] = payload
        return result

    @staticmethod
    def _terminal_role_producers(state: WorkflowState) -> dict[str, list[str]]:
        roles = list(getattr(state, "required_terminal_roles", []) or [])
        return {
            role: [
                task.task_id
                for task in state.tasks.values()
                if any(output.role == role for output in task.expected_outputs)
                and str(getattr(task.status, "value", task.status)) != "superseded"
            ]
            for role in roles
        }

    @classmethod
    def _validate_goal_preservation(
        cls,
        original: WorkflowState,
        revised: WorkflowState,
    ) -> None:
        original_roles = list(getattr(original, "required_terminal_roles", []) or [])
        revised_roles = list(getattr(revised, "required_terminal_roles", []) or [])
        if revised_roles != original_roles:
            raise ValueError(
                "Structural patch changed the required terminal goal contract; required_terminal_roles must be preserved."
            )
        missing = [
            role
            for role, producers in cls._terminal_role_producers(revised).items()
            if not producers
        ]
        if missing:
            raise ValueError(
                "Structural patch made required terminal output roles unreachable: "
                + ", ".join(sorted(missing))
            )

        before_commitments = cls._active_commitment_signature(original)
        after_commitments = cls._active_commitment_signature(revised)
        for commitment_id, payload in before_commitments.items():
            if commitment_id not in after_commitments:
                raise ValueError(
                    f"Structural patch removed/deactivated active analytical commitment {commitment_id!r}."
                )
            if after_commitments[commitment_id] != payload:
                raise ValueError(
                    f"Structural patch silently changed active analytical commitment {commitment_id!r}."
                )

        # A minimal repair may stale descendants, but it must not delete previously
        # completed work from the authoritative state.
        successful_before = {
            task.task_id
            for task in original.tasks.values()
            if str(getattr(task.status, "value", task.status)) == "successful"
        }
        missing_completed = sorted(successful_before - set(revised.tasks))
        if missing_completed:
            raise ValueError(
                "Structural patch deleted previously successful task(s): "
                + ", ".join(missing_completed)
            )

    @staticmethod
    def _repair_contract(codes: set[str]) -> list[dict[str, Any]]:
        strategies = {
            "METRIC_DISTANCE_UNSUPPORTED": ("reproject", "Establish a metric CRS before the distance operation."),
            "CRS_INCOMPATIBLE": ("reproject", "Harmonize only the incompatible input needed by the blocked operation."),
            "INVALID_GEOMETRY": ("geometry_repair", "Repair invalid geometry before the geometry-consuming operation."),
            "TABLE_REQUIRES_SPATIALIZATION": ("spatialize_coordinates", "Convert coordinate-bearing table rows to spatial point features using observed fields."),
            "COORDINATE_FIELDS_UNRESOLVED": ("ground_runtime_schema", "Ground coordinate parameters to exact observed longitude/latitude or x/y fields before spatialization."),
            "DATA_MODEL_INCOMPATIBLE": ("convert_data_model", "Use an advertised conversion producing the data model required by the consumer."),
            "FORMAT_UNSUPPORTED": ("convert_format", "Convert to a format explicitly accepted by the blocked worker."),
            "SCHEMA_PARAMETER_UNRESOLVED": ("ground_runtime_schema", "Use an exact observed runtime field in the consuming task parameters."),
            "SCHEMA_REQUIREMENT_UNRESOLVED": ("ground_runtime_schema", "Use exact observed runtime fields; never fabricate a column."),
            "RASTER_GRID_INCOMPATIBLE": ("align_raster_grid", "Make cellwise raster inputs compatible in CRS, resolution and grid geometry."),
            "TEMPORAL_REQUIREMENT_UNMET": ("alternative_candidate", "Obtain/use a candidate satisfying the explicit temporal requirement."),
            "SPATIAL_COVERAGE_INSUFFICIENT": ("alternative_candidate", "Obtain/use a candidate that covers the required target extent."),
            "RESOLUTION_REQUIREMENT_UNMET": ("alternative_candidate", "Obtain/use a candidate satisfying the explicit resolution threshold."),
        }
        return [
            {"violation_code": code, "preferred_strategy": strategies[code][0], "instruction": strategies[code][1]}
            for code in sorted(codes)
            if code in strategies
        ]

    @staticmethod
    def _latest_inspection_values(state: WorkflowState, artifact) -> dict[str, Any]:
        for evidence_id in reversed(getattr(artifact, "evidence_ids", []) or []):
            if evidence_id not in state.evidence:
                continue
            record = state.evidence[evidence_id]
            if getattr(record, "evidence_type", None) == "artifact_inspection":
                return dict(getattr(record, "values", {}) or {})
        return {}

    @staticmethod
    def _canonical_format(value: str | None) -> str:
        normalized = str(value or "").lower().lstrip(".")
        return {
            "tif": "geotiff",
            "tiff": "geotiff",
            "geopackage": "gpkg",
        }.get(normalized, normalized)

    @staticmethod
    def _goal_explicitly_assigns_crs(user_goal: str, target_crs: str) -> bool:
        text = " ".join(str(user_goal or "").lower().replace("-", " ").split())
        target = str(target_crs or "").lower().strip()
        if not target or target not in str(user_goal or "").lower():
            return False
        return any(
            token in text
            for token in (
                "assign crs",
                "define crs",
                "set crs",
                "assign coordinate reference",
                "define coordinate reference",
            )
        )

    @classmethod
    def _has_trusted_crs_assertion(
        cls,
        state: WorkflowState,
        *,
        target_crs: str | None = None,
    ) -> bool:
        target = str(target_crs or "").strip().lower()
        for commitment in state.commitments.values():
            if not getattr(commitment, "active", False):
                continue
            if str(getattr(commitment, "key", "")) != "user_asserted_crs":
                continue
            value = str(getattr(commitment, "value", "")).strip().lower()
            if not target or value == target:
                return True
        return bool(
            target
            and cls._goal_explicitly_assigns_crs(state.user_goal, target_crs or "")
        )

    @classmethod
    def _artifact_crs(cls, state: WorkflowState, artifact) -> str | None:
        values = cls._latest_inspection_values(state, artifact)
        value = values.get("crs_authority") or values.get("crs")
        return str(value).strip() if value else None

    @classmethod
    def _metric_target_is_safe(cls, target_crs: str | None) -> bool:
        if not target_crs:
            return False
        try:
            from pyproj import CRS

            crs = CRS.from_user_input(target_crs)
            if not crs.is_projected:
                return False
            axes = list(crs.axis_info or [])
            if not axes:
                return False
            return all(
                abs(float(getattr(axis, "unit_conversion_factor", 0.0)) - 1.0)
                <= 1e-9
                for axis in axes[:2]
            )
        except Exception:
            # If CRS parsing is unavailable, do not silently certify an unsafe
            # target.  The LLM receives the validation error and can choose a
            # parseable advertised CRS on the bounded correction attempt.
            return False

    @classmethod
    def _derive_local_metric_crs(
        cls,
        state: WorkflowState,
        artifact,
    ) -> str | None:
        """Derive a local UTM CRS from observed bounds when objectively possible."""
        values = cls._latest_inspection_values(state, artifact)
        source_crs = values.get("crs_authority") or values.get("crs")
        bounds = values.get("bounds") or values.get("extent")
        if not source_crs or not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
            return None
        try:
            from math import floor
            from pyproj import Transformer

            cx = (float(bounds[0]) + float(bounds[2])) / 2.0
            cy = (float(bounds[1]) + float(bounds[3])) / 2.0
            transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(cx, cy)
            if not (-180.0 <= lon <= 180.0 and -80.0 <= lat <= 84.0):
                return None
            zone = max(1, min(60, int(floor((lon + 180.0) / 6.0)) + 1))
            epsg = 32600 + zone if lat >= 0 else 32700 + zone
            return f"EPSG:{epsg}"
        except Exception:
            return None

    @staticmethod
    def _descendants(state: WorkflowState, task_id: str) -> set[str]:
        children: dict[str, set[str]] = {}
        for source, target in list(getattr(state, "edges", []) or []):
            children.setdefault(str(source), set()).add(str(target))
        seen: set[str] = set()
        stack = list(children.get(task_id, set()))
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            seen.add(item)
            stack.extend(children.get(item, set()) - seen)
        return seen

    @classmethod
    def _implicated_artifacts(
        cls,
        state: WorkflowState,
        blockers: list[Any],
    ) -> list[Any]:
        artifact_ids: list[str] = []
        for blocker in blockers:
            for artifact_id in getattr(blocker, "artifact_ids", []) or []:
                if artifact_id not in artifact_ids:
                    artifact_ids.append(artifact_id)
        return [state.artifacts[item] for item in artifact_ids if item in state.artifacts]

    @classmethod
    def _select_repair_source_artifact(
        cls,
        state: WorkflowState,
        blocked_task,
        task_payload: dict[str, Any],
        blockers: list[Any],
    ):
        inputs = task_payload.get("input_requirements") or []
        input_role = ""
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            input_role = str(inputs[0].get("role") or "")

        implicated = cls._implicated_artifacts(state, blockers)
        role_matches = [item for item in implicated if str(item.role) == input_role]
        if len(role_matches) == 1:
            return role_matches[0]
        if len(implicated) == 1:
            return implicated[0]

        # In GIS-aware mode the operation itself can sometimes identify the
        # affected artifact without any task/family-specific knowledge.
        operation = str(task_payload.get("operation") or "").lower()
        if operation == "geometry_repair":
            invalid = [
                item
                for item in implicated
                if cls._latest_inspection_values(state, item).get("geometry_valid")
                is False
            ]
            if len(invalid) == 1:
                return invalid[0]

        if operation in {"csv_to_points", "spatialize_coordinates"}:
            tables = [
                item
                for item in implicated
                if str(getattr(item, "data_model", "") or "").lower() == "table"
            ]
            if len(tables) == 1:
                return tables[0]

        if operation in {"reproject", "reproject_raster", "assign_crs"}:
            target = str((task_payload.get("parameters") or {}).get("target_crs") or "").strip()
            if target:
                different = [
                    item
                    for item in implicated
                    if (cls._artifact_crs(state, item) or "").lower()
                    and (cls._artifact_crs(state, item) or "").lower()
                    != target.lower()
                ]
                if len(different) == 1:
                    return different[0]

        if blocked_task is not None and operation in {
            "geojson_to_gpkg",
            "shapefile_to_gpkg",
            "table_to_csv",
            "convert",
            "convert_format",
        }:
            incompatible = []
            requirements = {
                str(req.role): req for req in getattr(blocked_task, "input_requirements", []) or []
            }
            for item in implicated:
                req = requirements.get(str(item.role))
                if req is None or not getattr(req, "formats", None):
                    continue
                actual = cls._canonical_format(getattr(item, "format", None))
                accepted = {cls._canonical_format(value) for value in req.formats}
                if actual and accepted and actual not in accepted:
                    incompatible.append(item)
            if len(incompatible) == 1:
                return incompatible[0]

        binding_id = getattr(blocked_task, "active_binding_id", None) if blocked_task else None
        binding = state.bindings.get(binding_id) if binding_id else None
        if binding is not None and input_role:
            matches = []
            for item in getattr(binding, "input_bindings", []) or []:
                if str(getattr(item, "role", "")) != input_role:
                    continue
                artifact_id = getattr(item, "artifact_id", None)
                if artifact_id in state.artifacts:
                    matches.append(state.artifacts[artifact_id])
            if len(matches) == 1:
                return matches[0]

        if input_role:
            matches = [
                artifact
                for artifact in state.artifacts.values()
                if str(artifact.role) == input_role
                and str(getattr(getattr(artifact, "validation_status", None), "value", getattr(artifact, "validation_status", ""))).lower()
                == "pass"
            ]
            if len(matches) == 1:
                return matches[0]
        return None

    @classmethod
    def _preferred_crs_for_incompatibility(
        cls,
        state: WorkflowState,
        source_artifact,
        blockers: list[Any],
    ) -> str | None:
        other = []
        source_id = getattr(source_artifact, "artifact_id", None)
        for artifact in cls._implicated_artifacts(state, blockers):
            if getattr(artifact, "artifact_id", None) == source_id:
                continue
            crs = cls._artifact_crs(state, artifact)
            if crs and crs not in other:
                other.append(crs)
        return other[0] if len(other) == 1 else None

    def _normalize_patch_payload(
        self,
        payload: Any,
        *,
        state: WorkflowState,
        decision: QCDecision,
        blockers: list[Any],
    ) -> tuple[Any, list[str]]:
        """Normalize an LLM patch into a safe, minimal graph transformation.

        The LLM still proposes the repair.  Deterministic code only enforces
        mechanics that have one safe interpretation: controller-owned fields,
        CRS-definition versus reprojection semantics, role preservation, and the
        direction of a one-step adapter around the blocked consumer.
        """
        if not isinstance(payload, dict):
            return payload, []
        normalized = deepcopy(payload)
        inserted_tasks = normalized.get("insert_tasks")
        if not isinstance(inserted_tasks, list):
            return normalized, []

        changes: list[str] = []
        controller_fields = {
            "requirement_ids",
            "status",
            "attempts",
            "active_binding_id",
            "output_artifact_ids",
            "last_error",
            "created_in_plan_version",
            "superseded_in_plan_version",
        }
        role_preserving_operations = {
            "assign_crs",
            "geometry_repair",
            "reproject",
            "reproject_raster",
            "resample",
        }
        adapter_operations = role_preserving_operations | {
            "convert",
            "convert_format",
            "spatialize_coordinates",
            "csv_to_points",
            "geojson_to_gpkg",
            "shapefile_to_gpkg",
            "table_to_csv",
        }
        blocked_task = state.tasks.get(decision.task_id)
        blocker_codes = {str(getattr(item, "code", "")) for item in blockers}
        blocker_codes.update(
            str(item)
            for item in (getattr(decision, "recovery_parameters", {}) or {}).get(
                "blocking_codes", []
            )
            if str(item)
        )

        # GIS-aware mode may normalize well-defined single-artifact repairs to a
        # minimal adapter. Generic mode keeps the LLM's analytical repair choice
        # intact so experimental feature boundaries remain meaningful.
        implicated = self._implicated_artifacts(state, blockers)
        if (
            self.guidance_mode == "gis_aware"
            and len(implicated) == 1
            and blocker_codes == {"METRIC_DISTANCE_UNSUPPORTED"}
        ):
            projection_tasks = [
                item
                for item in inserted_tasks
                if isinstance(item, dict)
                and str(item.get("operation") or "").lower()
                in {"reproject", "assign_crs"}
            ]
            if projection_tasks:
                if len(inserted_tasks) != 1:
                    changes.append(
                        f"discarded_nonminimal_metric_repair_tasks={len(inserted_tasks) - 1}"
                    )
                inserted_tasks = [projection_tasks[0]]
                normalized["insert_tasks"] = inserted_tasks
        elif (
            self.guidance_mode == "gis_aware"
            and len(implicated) == 1
            and blocker_codes == {"INVALID_GEOMETRY"}
        ):
            geometry_tasks = [
                item
                for item in inserted_tasks
                if isinstance(item, dict)
                and str(item.get("operation") or "").lower() == "geometry_repair"
            ]
            if geometry_tasks:
                if len(inserted_tasks) != 1:
                    changes.append(
                        f"discarded_nonminimal_geometry_repair_tasks={len(inserted_tasks) - 1}"
                    )
                inserted_tasks = [geometry_tasks[0]]
                normalized["insert_tasks"] = inserted_tasks

        descendants = self._descendants(state, decision.task_id)
        unsafe_dependency_ids = descendants | {decision.task_id}
        repair_ids: set[str] = set()

        for task in inserted_tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("task_id") or "<unknown>")
            if task_id != "<unknown>":
                repair_ids.add(task_id)
            removed = sorted(controller_fields & set(task))
            for field in removed:
                task.pop(field, None)
            if removed:
                changes.append(
                    f"{task_id}:removed_controller_fields={','.join(removed)}"
                )

            operation = str(task.get("operation") or "").lower()
            inputs = task.get("input_requirements")
            outputs = task.get("expected_outputs")
            source_artifact = self._select_repair_source_artifact(
                state, blocked_task, task, blockers
            )

            # If exactly one objective artifact is implicated, its role is the
            # authoritative role for a one-input repair.  Do not allow the LLM to
            # invent names such as `boundary_layer` that have no producer.
            if (
                source_artifact is not None
                and isinstance(inputs, list)
                and len(inputs) == 1
                and isinstance(inputs[0], dict)
            ):
                source_role = str(source_artifact.role)
                if str(inputs[0].get("role") or "") != source_role:
                    inputs[0]["role"] = source_role
                    changes.append(f"{task_id}:grounded_input_role={source_role}")

            # GIS-aware CRS safety. `assign_crs` is metadata definition, never a
            # substitute for coordinate transformation. Generic Replanner mode is
            # intentionally not given this GIS repair policy.
            if (
                self.guidance_mode == "gis_aware"
                and operation in {"assign_crs", "reproject"}
            ):
                parameters = task.setdefault("parameters", {})
                target_crs = str(parameters.get("target_crs") or "").strip()
                source_crs = (
                    self._artifact_crs(state, source_artifact)
                    if source_artifact is not None
                    else None
                )
                trusted = self._has_trusted_crs_assertion(
                    state, target_crs=target_crs or None
                )

                if operation == "assign_crs":
                    if source_crs and blocker_codes & {
                        "METRIC_DISTANCE_UNSUPPORTED",
                        "CRS_INCOMPATIBLE",
                    }:
                        task["operation"] = "reproject"
                        task["required_capabilities"] = ["reproject"]
                        task["instructions"] = (
                            "Transform the input coordinates to the explicit target CRS; "
                            "do not merely relabel CRS metadata."
                        )
                        operation = "reproject"
                        changes.append(f"{task_id}:assign_crs_changed_to_reproject")
                    elif not source_crs and not trusted:
                        raise ValueError(
                            "Unsafe CRS repair: assign_crs was proposed for an artifact "
                            "whose CRS is unknown without an explicit user/trusted CRS assertion. "
                            "Do not guess CRS; route human clarification instead."
                        )

                if operation == "reproject":
                    if not source_crs and not trusted:
                        raise ValueError(
                            "Unsafe reprojection: source CRS is unknown and no trusted CRS assertion exists."
                        )
                    if "METRIC_DISTANCE_UNSUPPORTED" in blocker_codes:
                        if source_artifact is not None:
                            derived = self._derive_local_metric_crs(state, source_artifact)
                        else:
                            derived = None
                        if not self._metric_target_is_safe(target_crs):
                            if derived and self._metric_target_is_safe(derived):
                                parameters["target_crs"] = derived
                                target_crs = derived
                                changes.append(
                                    f"{task_id}:target_crs_grounded_to_local_metric={derived}"
                                )
                            else:
                                raise ValueError(
                                    "Metric-distance repair requires an explicit projected CRS with metre units."
                                )
                    elif "CRS_INCOMPATIBLE" in blocker_codes and not target_crs:
                        preferred = (
                            self._preferred_crs_for_incompatibility(
                                state, source_artifact, blockers
                            )
                            if source_artifact is not None
                            else None
                        )
                        if preferred:
                            parameters["target_crs"] = preferred
                            changes.append(
                                f"{task_id}:target_crs_grounded_to_peer={preferred}"
                            )

            operation = str(task.get("operation") or "").lower()
            if (
                operation in role_preserving_operations
                and isinstance(inputs, list)
                and len(inputs) == 1
                and isinstance(inputs[0], dict)
                and isinstance(outputs, list)
                and len(outputs) == 1
                and isinstance(outputs[0], dict)
            ):
                input_role = str(inputs[0].get("role") or "")
                output_role = str(outputs[0].get("role") or "")
                if input_role and output_role != input_role:
                    outputs[0]["role"] = input_role
                    changes.append(f"{task_id}:grounded_output_role={input_role}")

            # An inserted repair must never depend on the blocked consumer or one
            # of its descendants.  For one-input adapters with an identified source
            # artifact, the direct source producer is the only required predecessor.
            if operation in adapter_operations:
                current_deps = [
                    str(item)
                    for item in (task.get("depends_on") or [])
                    if str(item) not in unsafe_dependency_ids
                ]
                if source_artifact is not None:
                    producer = getattr(source_artifact, "producer_task_id", None)
                    safe_deps = [str(producer)] if producer else []
                    if current_deps != safe_deps:
                        task["depends_on"] = safe_deps
                        changes.append(
                            f"{task_id}:repair_dependencies_grounded={','.join(safe_deps) or '<initial_artifact>'}"
                        )
                elif current_deps != list(task.get("depends_on") or []):
                    task["depends_on"] = current_deps
                    changes.append(f"{task_id}:removed_downstream_dependencies")

        # Canonicalize the common one-adapter topology.  Preserve unrelated
        # prerequisites, replace the repaired artifact's direct producer when it
        # exists, and make the blocked consumer depend on the adapter.
        if blocked_task is not None and len(inserted_tasks) == 1:
            repair = inserted_tasks[0]
            if isinstance(repair, dict):
                repair_id = str(repair.get("task_id") or "")
                operation = str(repair.get("operation") or "").lower()
                source_artifact = self._select_repair_source_artifact(
                    state, blocked_task, repair, blockers
                )
                if repair_id and operation in adapter_operations:
                    original_deps = [str(item) for item in blocked_task.depends_on]
                    producer = (
                        str(getattr(source_artifact, "producer_task_id", "") or "")
                        if source_artifact is not None
                        else ""
                    )
                    new_deps = [item for item in original_deps if item != producer]
                    if repair_id not in new_deps:
                        new_deps.append(repair_id)
                    updates = normalized.setdefault("update_tasks", {})
                    if not isinstance(updates, dict):
                        raise ValueError("GraphPatch update_tasks must be an object.")
                    update = updates.get(decision.task_id)
                    if not isinstance(update, dict):
                        update = {}
                    update["depends_on"] = new_deps
                    updates[decision.task_id] = update
                    changes.append(
                        f"{decision.task_id}:consumer_dependencies_grounded={','.join(new_deps)}"
                    )

        # Remove explicit dependency additions that contradict the safe adapter
        # direction.  Graph validation remains the final authority for all other
        # topology.
        additions = normalized.get("add_dependencies")
        if isinstance(additions, list) and repair_ids:
            filtered = []
            for edge in additions:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    source, target = str(edge[0]), str(edge[1])
                    if source == decision.task_id and target in repair_ids:
                        changes.append(
                            f"removed_forbidden_dependency={source}->{target}"
                        )
                        continue
                filtered.append(edge)
            normalized["add_dependencies"] = filtered

        return normalized, changes

    def repair(
        self,
        state: WorkflowState,
        decision: QCDecision,
        *,
        allow_structural_patching: bool,
    ) -> WorkflowState:
        reserve_replan(state)
        prompt = self._prompt()
        blocked_task = state.tasks.get(decision.task_id)
        capabilities_by_agent = {
            str(snapshot.get("agent_id") or index): snapshot
            for index, snapshot in enumerate(self.capability_snapshots)
        }
        capabilities_by_agent.update(
            {
                binding.agent_id: binding.agent_profile_snapshot
                for binding in state.bindings.values()
                if binding.agent_profile_snapshot
            }
        )
        active_blockers = [
            item
            for item in state.violations.values()
            if item.status == "active"
            and (item.task_id == decision.task_id or item.violation_id in set(decision.acknowledged_finding_ids))
        ]
        blocker_codes = {item.code for item in active_blockers}
        blocker_codes.update(
            str(item)
            for item in (getattr(decision, "recovery_parameters", {}) or {}).get("blocking_codes", [])
            if str(item)
        )
        input_payload = {
            "user_goal": state.user_goal,
            "base_plan_version": state.current_plan_version,
            "tasks": [item.model_dump(mode="json") for item in state.tasks.values()],
            "edges": state.edges,
            "worker_capabilities": [
                snapshot
                for _, snapshot in sorted(capabilities_by_agent.items())
            ],
            "accepted_artifacts": [
                item.model_dump(mode="json")
                for item in state.artifacts.values()
                if item.validation_status.value == "pass"
            ],
            "runtime_evidence": [
                state.evidence[evidence_id].model_dump(mode="json")
                for artifact in state.artifacts.values()
                if artifact.validation_status.value == "pass"
                for evidence_id in artifact.evidence_ids
                if evidence_id in state.evidence
            ],
            "derivations": [item.model_dump(mode="json") for item in state.derivations.values()],
            "active_commitments": [
                item.model_dump(mode="json")
                for item in state.commitments.values()
                if item.active
            ],
            "qc_decision": decision.model_dump(mode="json"),
            "blocked_task": blocked_task.model_dump(mode="json") if blocked_task else None,
            "required_terminal_roles": list(getattr(state, "required_terminal_roles", []) or []),
            "terminal_role_producers": self._terminal_role_producers(state),
            "repair_contract": self._repair_contract(blocker_codes) if self.guidance_mode == "gis_aware" else [],
            "blocking_artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "role": artifact.role,
                    "producer_task_id": artifact.producer_task_id,
                    "data_model": artifact.data_model,
                    "format": artifact.format,
                    "inspection": self._latest_inspection_values(state, artifact),
                }
                for artifact in self._implicated_artifacts(state, active_blockers)
            ],
            "repair_topology_constraints": {
                "blocked_task_id": decision.task_id,
                "existing_direct_prerequisite_task_ids": (
                    list(blocked_task.depends_on) if blocked_task else []
                ),
                "required_direction": "inserted_repair_task -> blocked_task",
                "forbidden_direction": "blocked_task -> inserted_repair_task",
                "instruction": (
                    "An inserted repair may depend on existing successful prerequisite "
                    "tasks. It must not depend on the blocked task. Update the blocked "
                    "task so it depends directly on the inserted repair task."
                ),
            },
            "violations": [
                item.model_dump(mode="json")
                for item in state.violations.values()
                if item.status == "active"
            ],
        }
        responses = []
        patch = None
        revised = None
        last_error: Exception | None = None
        for correction_round in range(state.budgets.replanner_retry_limit + 1):
            reserve_model_call(state)
            response = self.client.complete(
                component=self.component_name,
                system_prompt=prompt,
                input_payload=input_payload,
                output_schema=GraphPatch.model_json_schema(),
                retry_limit=0,
            )
            responses.append(response)
            state.budgets.total_tokens += response.input_tokens + response.output_tokens
            try:
                normalized_payload, normalizations = self._normalize_patch_payload(
                    response.payload,
                    state=state,
                    decision=decision,
                    blockers=active_blockers,
                )
                if normalizations:
                    state.record_event(
                        "replanner_patch_normalized",
                        actor="graph_controller",
                        task_id=decision.task_id,
                        reason=(
                            "Applied controller-owned patch normalization before "
                            "graph validation."
                        ),
                        metadata={"normalizations": normalizations},
                    )
                patch = GraphPatch.model_validate(normalized_payload)
                if not patch.changes_structure():
                    raise ValueError(
                        "Structural Replanner returned a patch with no structural changes."
                    )
                revised = apply_graph_patch(
                    state,
                    patch,
                    allow_structural_patching=allow_structural_patching,
                )
                if capabilities_by_agent:
                    for task in revised.tasks.values():
                        eligible = []
                        for capability in capabilities_by_agent.values():
                            operations = set(capability.get("operations") or [])
                            if task.operation not in operations and "*" not in operations:
                                continue
                            supported = operations | {
                                str(capability.get("agent_id") or "")
                            }
                            if (
                                set(task.required_capabilities).issubset(supported)
                                and capability_supports_expected_outputs(
                                    capability, task
                                )
                            ):
                                eligible.append(capability.get("agent_id"))
                        if not eligible:
                            raise ValueError(
                                f"Patched task {task.task_id!r} uses unsupported "
                                f"operation {task.operation!r} or capability "
                                f"requirements {task.required_capabilities!r}."
                            )
                self._validate_goal_preservation(state, revised)
                break
            except (ValueError, TypeError) as exc:
                last_error = exc
                patch = None
                revised = None
                state.record_event(
                    "replanner_patch_rejected",
                    actor="graph_controller",
                    task_id=decision.task_id,
                    reason=str(exc),
                    metadata={"correction_round": correction_round + 1},
                )
                input_payload = dict(input_payload)
                input_payload["previous_patch_validation"] = {
                    "status": "REJECTED",
                    "error": str(exc),
                    "previous_patch": response.payload,
                    "instruction": (
                        "Return a corrected minimal patch that resolves the supplied "
                        "validation error without changing unrelated completed work. "
                        "If the error is a cycle, make the inserted repair depend only "
                        "on already-successful prerequisite producers and make the "
                        "blocked consumer depend on the repair; the repair must never "
                        "depend on the blocked consumer. If the error reports an "
                        "unsupported operation, choose an operation and capability "
                        "explicitly advertised together by the same worker in "
                        "worker_capabilities. Inserted tasks must not copy requirement "
                        "IDs from the blocked task. A role-preserving repair must emit "
                        "the same role it consumes so the blocked consumer uses the "
                        "repaired artifact. Preserve required_terminal_roles and all active analytical commitments exactly; never make the patch valid by dropping a requested final output. Follow repair_contract when supplied and ground any schema-field update to exact runtime evidence."
                    ),
                }
        if revised is None or patch is None:
            if responses:
                state.component_calls.append(
                    ComponentCallRecord(
                        component="replanner",
                        prompt_hash=stable_hash(prompt),
                        tool_schema_hash=stable_hash(GraphPatch.model_json_schema()),
                        model=responses[-1].model,
                        provider=responses[-1].provider,
                        attempts=sum(item.attempts for item in responses),
                        input_tokens=sum(item.input_tokens for item in responses),
                        output_tokens=sum(item.output_tokens for item in responses),
                        duration_seconds=sum(item.duration_seconds for item in responses),
                        disposition="failed_patch_validation",
                    )
                )
            raise ValueError(
                "Replanner failed to produce a valid structural patch after "
                f"{len(responses)} attempt(s): {last_error}"
            )
        revised.replan_history.append(
            ReplanDecision(
                base_plan_version=patch.base_plan_version,
                rationale=patch.rationale,
                patch=patch.model_dump(mode="json"),
                accepted=True,
            )
        )
        revised.component_calls.append(
            ComponentCallRecord(
                component="replanner",
                prompt_hash=stable_hash(prompt),
                tool_schema_hash=stable_hash(GraphPatch.model_json_schema()),
                model=responses[-1].model,
                provider=responses[-1].provider,
                attempts=sum(item.attempts for item in responses),
                input_tokens=sum(item.input_tokens for item in responses),
                output_tokens=sum(item.output_tokens for item in responses),
                duration_seconds=sum(item.duration_seconds for item in responses),
                disposition=(
                    "successful"
                    if len(responses) == 1
                    else "successful_after_patch_validation_retry"
                ),
            )
        )
        return revised
