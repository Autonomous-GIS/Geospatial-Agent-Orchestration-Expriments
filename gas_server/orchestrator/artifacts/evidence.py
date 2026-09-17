"""Register bounded factual artifact evidence and deterministic findings.

The invariant layer is deliberately requirement-driven and benchmark-agnostic.  It
turns inspectable facts into typed blocking violations so the controller receives a
clear alarm before an invalid transition is accepted.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from pyproj import CRS, Transformer

from gas_server.orchestrator.artifacts.inspectors import INSPECTOR_VERSION, inspect_path
from gas_server.orchestrator.core.models import (
    ArtifactState,
    EvidenceRecord,
    Requirement,
    ViolationRecord,
    WorkflowState,
    utc_now,
)
from gas_server.orchestrator.security.inspection_policy import InspectionPolicy


class EvidenceEngine:
    def __init__(self, policy: InspectionPolicy | None = None):
        self.policy = policy or InspectionPolicy()

    def inspect_artifacts(
        self, state: WorkflowState, artifacts: Iterable[ArtifactState]
    ) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        for artifact in artifacts:
            try:
                values = inspect_path(artifact.location, self.policy)
            except Exception as exc:
                # Inspection failure is itself useful technical evidence.  Do not let
                # one unreadable artifact erase the rest of the workflow state.
                values = {
                    "path": str(artifact.location),
                    "format": getattr(artifact, "format", None),
                    "complete": False,
                    "limitations": ["inspection_failed"],
                    "inspection_error_type": type(exc).__name__,
                    "inspection_error": str(exc)[:1000],
                }
            record = EvidenceRecord(
                evidence_type="artifact_inspection",
                subject_ids=[artifact.artifact_id],
                values=values,
                source="deterministic_artifact_inspector",
                inspector_version=INSPECTOR_VERSION,
                complete=bool(values.get("complete", False)),
                limitations=list(values.get("limitations", [])),
            )
            state.evidence[record.evidence_id] = record
            artifact.evidence_ids.append(record.evidence_id)
            artifact.data_model = values.get("data_model") or artifact.data_model
            artifact.format = values.get("format") or artifact.format
            records.append(record)
        return records


def _inspection_record(state: WorkflowState, artifact: ArtifactState):
    for evidence_id in reversed(artifact.evidence_ids):
        if evidence_id not in state.evidence:
            continue
        record = state.evidence[evidence_id]
        if record.evidence_type == "artifact_inspection":
            return record
    return None


def _inspection(state: WorkflowState, artifact: ArtifactState) -> dict:
    record = _inspection_record(state, artifact)
    return record.values if record is not None else {}


def _crs(value: str | None) -> CRS | None:
    if not value:
        return None
    try:
        return CRS.from_user_input(value)
    except Exception:
        return None


def _crs_equal(left: str | None, right: str | None) -> bool:
    left_crs = _crs(left)
    right_crs = _crs(right)
    if left_crs is None or right_crs is None:
        return False
    try:
        return bool(left_crs.equals(right_crs))
    except Exception:
        return left_crs == right_crs


def _metric_crs(value: str | None) -> bool:
    crs = _crs(value)
    if crs is None or not crs.is_projected:
        return False
    for axis in crs.axis_info:
        unit = (axis.unit_name or "").lower()
        if unit in {"meter", "metre", "meters", "metres"}:
            return True
    return False


def _format_key(value: str | None) -> str:
    key = str(value or "").lower().lstrip(".")
    aliases = {
        "tiff": "tif",
        "geotiff": "tif",
        "cog": "tif",
        "json": "geojson",
        "geopackage": "gpkg",
        "shapefile": "shp",
    }
    return aliases.get(key, key)


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _field_parameter_keys(parameters: dict[str, Any]) -> set[str]:
    """Return parameters that must be grounded to existing input schema fields.

    Output-field/name parameters are intentionally excluded because they may create
    new columns rather than reference an existing one.
    """

    result: set[str] = set()
    explicit = {
        "field", "fields", "columns", "group_by", "groupby",
        "vector_key", "table_key", "join_key", "date_field", "time_field",
        "value_field", "category_field", "x_field", "y_field",
        "longitude_field", "latitude_field", "zone_field", "id_field",
    }
    for key in parameters:
        lowered = str(key).lower()
        if lowered.startswith("output_") or lowered.startswith("new_"):
            continue
        if (
            lowered in explicit
            or lowered.endswith("_input_field")
            or lowered.endswith("_source_field")
            or lowered.endswith("_join_key")
            or lowered in {"left_key", "right_key"}
        ):
            result.add(str(key))
    return result


def _parameter_field_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _task_text(task: Any) -> str:
    return " ".join(
        str(getattr(task, name, "") or "")
        for name in ("operation", "title", "purpose", "instructions")
    ).lower()


def _requested_year(task: Any) -> str | None:
    params = dict(getattr(task, "parameters", {}) or {})
    for key in ("year", "requested_year", "target_year"):
        value = params.get(key)
        if value is not None and re.fullmatch(r"(?:19|20)\d{2}", str(value).strip()):
            return str(value).strip()
    match = re.search(r"\b((?:19|20)\d{2})\b", _task_text(task))
    return match.group(1) if match else None


def _requested_max_resolution(task: Any) -> float | None:
    params = dict(getattr(task, "parameters", {}) or {})
    for key in (
        "max_resolution", "maximum_resolution", "resolution_max",
        "max_resolution_m", "maximum_resolution_m",
    ):
        value = params.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    text = _task_text(task)
    patterns = (
        r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)[ -]*(?:or|and)\s+finer",
        r"(?:resolution|cell size|pixel size)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _float_seq_equal(left, right, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    if left is None or right is None:
        return False
    try:
        left_values = list(left)
        right_values = list(right)
    except TypeError:
        return False
    if len(left_values) != len(right_values):
        return False
    for a, b in zip(left_values, right_values):
        try:
            if not math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        except (TypeError, ValueError):
            if a != b:
                return False
    return True


def _raster_grids_equal(left: dict, right: dict) -> bool:
    if not _crs_equal(left.get("crs"), right.get("crs")):
        return False
    if not _float_seq_equal(left.get("resolution"), right.get("resolution"), rel_tol=1e-8, abs_tol=1e-8):
        return False
    if not _float_seq_equal(left.get("transform"), right.get("transform"), rel_tol=1e-9, abs_tol=1e-8):
        return False
    if left.get("width") != right.get("width") or left.get("height") != right.get("height"):
        return False
    return True


def _parse_temporal(value: str | None, *, end: bool = False) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc) if end else datetime(year, 1, 1, tzinfo=timezone.utc)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        # Accept simple YYYY-MM-DD-like prefixes even when the source adds an
        # unsupported suffix.
        match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
        if not match:
            return None
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                23 if end else 0, 59 if end else 0, 59 if end else 0,
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None


def _coverage_matches_request(coverage: dict, requested_start: str, requested_end: str) -> bool:
    actual_start = _parse_temporal(coverage.get("start"), end=False)
    actual_end = _parse_temporal(coverage.get("end"), end=True)
    request_start = _parse_temporal(requested_start, end=False)
    request_end = _parse_temporal(requested_end or requested_start, end=True)
    if actual_start is None or actual_end is None or request_start is None or request_end is None:
        return False
    return actual_start <= request_start and actual_end >= request_end


def _bounds_cover(
    data_bounds,
    data_crs_value: str | None,
    target_bounds,
    target_crs_value: str | None,
) -> bool | None:
    if not data_bounds or not target_bounds:
        return None
    data_crs = _crs(data_crs_value)
    target_crs = _crs(target_crs_value)
    if data_crs is None or target_crs is None:
        return None
    try:
        target = [float(item) for item in target_bounds]
        if not data_crs.equals(target_crs):
            transformer = Transformer.from_crs(target_crs, data_crs, always_xy=True)
            target = list(transformer.transform_bounds(*target, densify_pts=21))
        data = [float(item) for item in data_bounds]
    except Exception:
        return None
    tolerance_x = max(1e-9, abs(data[2] - data[0]) * 1e-9)
    tolerance_y = max(1e-9, abs(data[3] - data[1]) * 1e-9)
    return bool(
        data[0] <= target[0] + tolerance_x
        and data[1] <= target[1] + tolerance_y
        and data[2] >= target[2] - tolerance_x
        and data[3] >= target[3] - tolerance_y
    )


def _commitments(state: WorkflowState, key: str | None = None):
    result = []
    for item in state.commitments.values():
        if not getattr(item, "active", False):
            continue
        if key and str(getattr(item, "key", "")) != key:
            continue
        result.append(item)
    return result


class InvariantEngine:
    """Evaluate explicit, machine-checkable requirements into typed violations."""

    STATE_REQUIREMENTS = {
        "analytical_unit_commitment",
        "commitment_present",
        "required_commitment",
        "commitment_disclosed",
        "disclosure_required",
    }

    REQUIREMENT_KEYS = {
        "metric_distance": {"crs"},
        "compatible_crs": {"crs"},
        "aligned_raster_grid": {"crs", "resolution", "transform", "width", "height"},
        "valid_geometry": {"geometry_valid"},
        "required_fields": {"schema"},
        "schema_grounded": {"schema"},
        "coordinate_fields": {"schema", "coordinate_candidates"},
        "spatializable_table": {"schema", "coordinate_candidates"},
        "temporal_range": {"temporal_candidates"},
        "spatial_coverage": {"bounds", "crs"},
        "max_resolution": {"resolution", "crs"},
        "format_compatible": {"format"},
        "crs_present": {"crs"},
    }

    @classmethod
    def is_evaluable(
        cls,
        state: WorkflowState,
        requirement: Requirement,
        artifacts_by_role: dict[str, ArtifactState],
    ) -> bool:
        """Return whether the facts needed by this requirement are observable.

        A key correction from the previous implementation is that *inspection
        completeness is no longer all-or-nothing*.  A sampled geometry check does
        not make a CRS, format, schema, or raster resolution fact unknowable.
        """

        requirement_type = str(requirement.requirement_type)
        if requirement_type in cls.STATE_REQUIREMENTS:
            return True
        if not requirement.artifact_roles:
            return False
        if any(role not in artifacts_by_role for role in requirement.artifact_roles):
            return False
        required_keys = cls.REQUIREMENT_KEYS.get(requirement_type)
        if required_keys is None:
            # Preserve the old conservative behavior for unknown requirement types.
            for role in requirement.artifact_roles:
                record = _inspection_record(state, artifacts_by_role[role])
                if record is None or not record.complete:
                    return False
            return True
        for role in requirement.artifact_roles:
            values = _inspection(state, artifacts_by_role[role])
            if not values:
                return False
            # Presence of the key is enough; None can itself be the factual signal
            # for missing CRS/coverage/etc.
            if any(key not in values for key in required_keys):
                return False
        return True

    def evaluable_requirements(
        self,
        state: WorkflowState,
        requirements: Iterable[Requirement],
        artifacts_by_role: dict[str, ArtifactState],
    ) -> list[Requirement]:
        return [
            requirement
            for requirement in requirements
            if self.is_evaluable(state, requirement, artifacts_by_role)
        ]

    def _state_requirement_code(self, state: WorkflowState, requirement: Requirement) -> str | None:
        params = dict(requirement.parameters or {})
        requirement_type = str(requirement.requirement_type)
        key = str(params.get("key") or ("analytical_unit" if requirement_type == "analytical_unit_commitment" else "")).strip() or None
        matches = _commitments(state, key)
        if requirement_type in {"analytical_unit_commitment", "commitment_present", "required_commitment"}:
            if not matches:
                return "ANALYTICAL_COMMITMENT_MISSING" if key == "analytical_unit" else "COMMITMENT_MISSING"
        elif requirement_type in {"commitment_disclosed", "disclosure_required"}:
            if not matches or any(not bool(getattr(item, "disclosed", False)) for item in matches):
                return "COMMITMENT_DISCLOSURE_MISSING"
        return None

    @staticmethod
    def _required_fields_missing(requirement: Requirement, related: list[ArtifactState], state: WorkflowState) -> bool:
        params = dict(requirement.parameters or {})
        fields_by_role = params.get("fields_by_role")
        if isinstance(fields_by_role, dict):
            role_to_artifact = dict(zip(requirement.artifact_roles, related))
            for role, fields in fields_by_role.items():
                artifact = role_to_artifact.get(role)
                if artifact is None:
                    return True
                schema = set((_inspection(state, artifact).get("schema") or {}).keys())
                if set(fields or []) - schema:
                    return True
            return False
        required = set(params.get("fields", []) or [])
        if not required:
            return False
        schemas = [set((_inspection(state, item).get("schema") or {}).keys()) for item in related]
        if len(schemas) == 1:
            return bool(required - schemas[0])
        # For multi-input operations, fields can legitimately belong to different
        # inputs.  Require coverage by the union unless role-specific mapping exists.
        union = set().union(*schemas) if schemas else set()
        return bool(required - union)

    @staticmethod
    def _schema_grounding_missing(requirement: Requirement, related: list[ArtifactState], state: WorkflowState) -> bool:
        params = dict(requirement.parameters or {})
        actual_field = str(params.get("actual_field") or params.get("field") or "").strip()
        if not actual_field:
            return True
        return all(actual_field not in (_inspection(state, item).get("schema") or {}) for item in related)

    @staticmethod
    def _coordinate_requirement_missing(requirement: Requirement, related: list[ArtifactState], state: WorkflowState) -> bool:
        params = dict(requirement.parameters or {})
        x_field = str(params.get("x_field") or params.get("longitude_field") or "").strip()
        y_field = str(params.get("y_field") or params.get("latitude_field") or "").strip()
        for item in related:
            inspection = _inspection(state, item)
            schema = set((inspection.get("schema") or {}).keys())
            if x_field and y_field and x_field in schema and y_field in schema:
                return False
            if inspection.get("coordinate_candidates"):
                return False
        return True

    @staticmethod
    def _contract_evidence_ids(artifacts: Iterable[ArtifactState]) -> list[str]:
        return sorted({
            evidence_id
            for artifact in artifacts
            for evidence_id in artifact.evidence_ids
        })

    @staticmethod
    def _add_contract_finding(
        findings: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
        code: str,
        artifacts: list[ArtifactState],
        *,
        computationally_resolvable: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        artifact_ids = tuple(sorted(item.artifact_id for item in artifacts))
        findings[(code, artifact_ids)] = {
            "code": code,
            "artifacts": list(artifacts),
            "computationally_resolvable": computationally_resolvable,
            "details": dict(details or {}),
        }

    def _task_contract_findings(
        self,
        state: WorkflowState,
        task_id: str,
        artifacts_by_role: dict[str, ArtifactState],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Derive unavoidable runtime alarms from the task/worker input contract.

        These checks do not depend on benchmark family IDs.  They cover constraints
        that are inherent in the requested operation or in the task's declared input
        contract, which makes detection robust even if the Planner forgot to create a
        redundant explicit Requirement object.
        """

        task = state.tasks.get(task_id)
        if task is None:
            return [], False
        required_roles = {
            item.role
            for item in task.input_requirements
            if bool(getattr(item, "required", True))
        }
        complete_inputs = required_roles.issubset(artifacts_by_role)
        findings: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

        # 1) Declared data-model and format contracts.
        for role_requirement in task.input_requirements:
            artifact = artifacts_by_role.get(role_requirement.role)
            if artifact is None:
                continue
            inspection = _inspection(state, artifact)
            actual_model = str(inspection.get("data_model") or artifact.data_model or "").lower()
            required_model = str(getattr(role_requirement, "data_model", None) or "").lower()
            if required_model and actual_model and required_model != actual_model:
                if (
                    actual_model == "table"
                    and required_model == "vector"
                    and bool(inspection.get("coordinate_candidates"))
                ):
                    self._add_contract_finding(
                        findings,
                        "TABLE_REQUIRES_SPATIALIZATION",
                        [artifact],
                        computationally_resolvable=True,
                        details={
                            "role": role_requirement.role,
                            "required_data_model": required_model,
                            "actual_data_model": actual_model,
                            "coordinate_candidates": inspection.get("coordinate_candidates"),
                        },
                    )
                else:
                    self._add_contract_finding(
                        findings,
                        "DATA_MODEL_INCOMPATIBLE",
                        [artifact],
                        computationally_resolvable=True,
                        details={
                            "role": role_requirement.role,
                            "required_data_model": required_model,
                            "actual_data_model": actual_model,
                        },
                    )
            allowed_formats = {
                _format_key(value) for value in list(getattr(role_requirement, "formats", []) or [])
            }
            actual_format = _format_key(inspection.get("format") or artifact.format)
            if allowed_formats and actual_format and actual_format not in allowed_formats:
                self._add_contract_finding(
                    findings,
                    "FORMAT_UNSUPPORTED",
                    [artifact],
                    computationally_resolvable=True,
                    details={
                        "role": role_requirement.role,
                        "actual_format": actual_format,
                        "allowed_formats": sorted(allowed_formats),
                    },
                )

        artifacts = list(artifacts_by_role.values())
        vector_artifacts = [
            item for item in artifacts
            if str(_inspection(state, item).get("data_model") or item.data_model or "").lower() == "vector"
        ]
        raster_artifacts = [
            item for item in artifacts
            if str(_inspection(state, item).get("data_model") or item.data_model or "").lower() == "raster"
        ]
        op = _normalized_name(getattr(task, "operation", ""))
        text = _task_text(task)

        # 2) Field parameters must be grounded in observed runtime schema.
        schema_fields = {
            field
            for artifact in artifacts
            for field in (_inspection(state, artifact).get("schema") or {}).keys()
        }
        if schema_fields:
            for key in _field_parameter_keys(dict(task.parameters or {})):
                requested_fields = _parameter_field_values(task.parameters.get(key))
                missing = [field for field in requested_fields if field not in schema_fields]
                if missing:
                    self._add_contract_finding(
                        findings,
                        "SCHEMA_PARAMETER_UNRESOLVED",
                        artifacts,
                        computationally_resolvable=True,
                        details={
                            "parameter": key,
                            "missing_fields": missing,
                            "available_fields": sorted(schema_fields)[:80],
                        },
                    )

        # 3) Metric-distance operations require a trustworthy metric CRS.
        params = dict(task.parameters or {})
        distance_keys = {
            key for key in params
            if any(token in str(key).lower() for token in ("distance", "radius", "buffer"))
        }
        metric_operation = "buffer" in op or bool(distance_keys)
        if metric_operation and vector_artifacts:
            crs_values = [_inspection(state, item).get("crs") for item in vector_artifacts]
            if any(not value for value in crs_values):
                self._add_contract_finding(
                    findings, "MISSING_CRS", vector_artifacts,
                    computationally_resolvable=False,
                    details={"operation": task.operation, "reason": "metric operation requires known CRS"},
                )
            elif any(not _metric_crs(value) for value in crs_values):
                self._add_contract_finding(
                    findings, "METRIC_DISTANCE_UNSUPPORTED", vector_artifacts,
                    computationally_resolvable=True,
                    details={"operation": task.operation, "crs_values": crs_values},
                )

        # 4) Geometry-consuming vector operations require valid geometry.
        geometry_tokens = (
            "buffer", "clip", "overlay", "spatialjoin", "intersect", "intersection",
            "within", "contains", "union", "difference", "zonal",
        )
        if vector_artifacts and any(token in op for token in geometry_tokens):
            invalid = [
                item for item in vector_artifacts
                if _inspection(state, item).get("geometry_valid") is False
            ]
            if invalid:
                self._add_contract_finding(
                    findings, "INVALID_GEOMETRY", invalid,
                    computationally_resolvable=True,
                    details={"operation": task.operation},
                )

        # 5) Binary spatial predicates require compatible defined CRSs.
        vector_binary_tokens = (
            "spatialjoin", "overlay", "intersect", "intersection", "union",
            "difference", "clip", "within", "contains",
        )
        if len(vector_artifacts) >= 2 and any(token in op for token in vector_binary_tokens):
            crs_values = [_inspection(state, item).get("crs") for item in vector_artifacts]
            if any(not value for value in crs_values):
                self._add_contract_finding(
                    findings, "MISSING_CRS", vector_artifacts,
                    computationally_resolvable=False,
                    details={"operation": task.operation},
                )
            elif any(not _crs_equal(crs_values[0], value) for value in crs_values[1:]):
                self._add_contract_finding(
                    findings, "CRS_INCOMPATIBLE", vector_artifacts,
                    computationally_resolvable=True,
                    details={"operation": task.operation, "crs_values": crs_values},
                )

        raster_vector_tokens = ("zonal", "rasterclip", "clipraster", "mask", "extractbymask")
        if raster_artifacts and vector_artifacts and any(token in op for token in raster_vector_tokens):
            involved = [raster_artifacts[0], vector_artifacts[0]]
            crs_values = [_inspection(state, item).get("crs") for item in involved]
            if any(not value for value in crs_values):
                self._add_contract_finding(
                    findings, "MISSING_CRS", involved,
                    computationally_resolvable=False,
                    details={"operation": task.operation},
                )
            elif not _crs_equal(crs_values[0], crs_values[1]):
                self._add_contract_finding(
                    findings, "CRS_INCOMPATIBLE", involved,
                    computationally_resolvable=True,
                    details={"operation": task.operation, "crs_values": crs_values},
                )

            # A raster used for clipping/zonal statistics must cover the requested
            # target geometry; CRS-aware bounds comparison prevents degree/metre mixups.
            raster_values = _inspection(state, raster_artifacts[0])
            vector_values = _inspection(state, vector_artifacts[0])
            if raster_values.get("crs") and vector_values.get("crs"):
                covers = _bounds_cover(
                    raster_values.get("bounds"), raster_values.get("crs"),
                    vector_values.get("bounds"), vector_values.get("crs"),
                )
                if covers is False:
                    self._add_contract_finding(
                        findings, "SPATIAL_COVERAGE_INSUFFICIENT", involved,
                        computationally_resolvable=True,
                        details={"operation": task.operation},
                    )

        # 6) Cell-wise multi-raster operations require an aligned grid.
        raster_math_tokens = (
            "rasterdifference", "subtract", "cellwise", "mapalgebra",
            "rastermath", "rasterarithmetic", "difference",
        )
        if len(raster_artifacts) >= 2 and any(token in op for token in raster_math_tokens):
            values = [_inspection(state, item) for item in raster_artifacts]
            if any(not value.get("crs") for value in values):
                self._add_contract_finding(
                    findings, "MISSING_CRS", raster_artifacts,
                    computationally_resolvable=False,
                    details={"operation": task.operation},
                )
            elif any(not _raster_grids_equal(values[0], value) for value in values[1:]):
                self._add_contract_finding(
                    findings, "RASTER_GRID_INCOMPATIBLE", raster_artifacts,
                    computationally_resolvable=True,
                    details={"operation": task.operation},
                )

        # 7) Explicit temporal and resolution criteria are checked against returned
        # candidate evidence even when the Planner omitted a duplicate requirement.
        year = _requested_year(task)
        if year:
            temporal_artifacts = [
                item for item in artifacts
                if _inspection(state, item).get("temporal_candidates")
            ]
            for artifact in temporal_artifacts:
                candidates = list(_inspection(state, artifact).get("temporal_candidates") or [])
                if candidates and not any(_coverage_matches_request(c, year, year) for c in candidates):
                    self._add_contract_finding(
                        findings, "TEMPORAL_REQUIREMENT_UNMET", [artifact],
                        computationally_resolvable=True,
                        details={"requested_year": year, "observed": candidates},
                    )

        maximum_resolution = _requested_max_resolution(task)
        if maximum_resolution is not None:
            for artifact in raster_artifacts:
                inspection = _inspection(state, artifact)
                resolution = inspection.get("resolution") or []
                if not resolution:
                    continue
                try:
                    values = [abs(float(value)) for value in resolution]
                except (TypeError, ValueError):
                    continue
                factor = inspection.get("linear_unit_to_meter")
                if inspection.get("is_projected") and factor:
                    values = [value * float(factor) for value in values]
                elif inspection.get("is_geographic"):
                    self._add_contract_finding(
                        findings, "RESOLUTION_REQUIREMENT_UNMET", [artifact],
                        computationally_resolvable=True,
                        details={
                            "maximum_resolution_m": maximum_resolution,
                            "reason": "angular raster resolution cannot satisfy a metric threshold directly",
                        },
                    )
                    continue
                if max(values) > maximum_resolution:
                    self._add_contract_finding(
                        findings, "RESOLUTION_REQUIREMENT_UNMET", [artifact],
                        computationally_resolvable=True,
                        details={
                            "maximum_resolution_m": maximum_resolution,
                            "observed_resolution_m": values,
                        },
                    )

        return list(findings.values()), complete_inputs

    def _apply_task_contract_findings(
        self,
        state: WorkflowState,
        task_id: str,
        artifacts_by_role: dict[str, ArtifactState],
    ) -> list[ViolationRecord]:
        findings, complete_inputs = self._task_contract_findings(
            state, task_id, artifacts_by_role
        )
        active_keys: set[tuple[str, tuple[str, ...]]] = set()
        result: list[ViolationRecord] = []
        for finding in findings:
            artifacts = list(finding["artifacts"])
            artifact_ids = [item.artifact_id for item in artifacts]
            key = (finding["code"], tuple(sorted(artifact_ids)))
            active_keys.add(key)
            evidence_ids = self._contract_evidence_ids(artifacts)
            violation = next(
                (
                    item
                    for item in state.violations.values()
                    if item.status == "active"
                    and item.task_id == task_id
                    and item.layer == "deterministic_task_contract"
                    and item.code == finding["code"]
                    and tuple(sorted(item.artifact_ids)) == key[1]
                ),
                None,
            )
            if violation is None:
                violation = ViolationRecord(
                    code=finding["code"],
                    layer="deterministic_task_contract",
                    task_id=task_id,
                    artifact_ids=artifact_ids,
                    requirement_ids=[],
                    evidence_ids=evidence_ids,
                    blocking=True,
                    deterministic=True,
                    computationally_resolvable=bool(finding["computationally_resolvable"]),
                    created_in_plan_version=max(1, state.current_plan_version),
                )
                state.violations[violation.violation_id] = violation
                state.record_event(
                    "deterministic_contract_violation_detected",
                    actor="invariant_engine",
                    task_id=task_id,
                    reason=f"Detected task-contract violation: {finding['code']}.",
                    metadata={
                        "violation_id": violation.violation_id,
                        "code": finding["code"],
                        "artifact_ids": artifact_ids,
                        "computationally_resolvable": finding["computationally_resolvable"],
                        "details": finding["details"],
                    },
                )
            else:
                violation.evidence_ids = sorted(set(violation.evidence_ids) | set(evidence_ids))
            result.append(violation)

        # Resolve contract alarms only when all required inputs are present.  This
        # avoids treating an incomplete evidence projection as proof that a prior
        # incompatibility disappeared.
        if complete_inputs:
            for existing in state.violations.values():
                if (
                    existing.status == "active"
                    and existing.task_id == task_id
                    and existing.layer == "deterministic_task_contract"
                ):
                    key = (existing.code, tuple(sorted(existing.artifact_ids)))
                    if key in active_keys:
                        continue
                    existing.status = "resolved"
                    existing.resolved_by_evidence_ids = self._contract_evidence_ids(
                        artifacts_by_role.values()
                    )
                    existing.resolved_in_plan_version = max(1, state.current_plan_version)
                    existing.resolved_at = utc_now()
                    state.record_event(
                        "deterministic_contract_violation_resolved",
                        actor="invariant_engine",
                        task_id=task_id,
                        reason=f"Task-contract violation {existing.code} is no longer present.",
                        metadata={"violation_id": existing.violation_id, "code": existing.code},
                    )
        return result

    def evaluate(
        self,
        state: WorkflowState,
        task_id: str,
        requirements: Iterable[Requirement],
        artifacts_by_role: dict[str, ArtifactState],
    ) -> list[ViolationRecord]:
        violations: list[ViolationRecord] = self._apply_task_contract_findings(
            state, task_id, artifacts_by_role
        )
        for requirement in requirements:
            if not self.is_evaluable(state, requirement, artifacts_by_role):
                continue

            related = [
                artifacts_by_role[role]
                for role in requirement.artifact_roles
                if role in artifacts_by_role
            ]
            evidence_ids = [eid for artifact in related for eid in artifact.evidence_ids]
            code: str | None = None
            requirement_type = str(requirement.requirement_type)

            if requirement_type in self.STATE_REQUIREMENTS:
                code = self._state_requirement_code(state, requirement)
            elif requirement_type == "metric_distance":
                crs_values = [_inspection(state, item).get("crs") for item in related]
                if any(not value for value in crs_values):
                    code = "MISSING_CRS"
                elif any(not _metric_crs(value) for value in crs_values):
                    code = "METRIC_DISTANCE_UNSUPPORTED"
            elif requirement_type == "compatible_crs" and len(related) >= 2:
                values = [_inspection(state, item).get("crs") for item in related]
                if any(not value for value in values):
                    code = "MISSING_CRS"
                elif any(not _crs_equal(values[0], value) for value in values[1:]):
                    code = "CRS_INCOMPATIBLE"
            elif requirement_type == "aligned_raster_grid" and len(related) >= 2:
                values = [_inspection(state, item) for item in related]
                if any(not value.get("crs") for value in values):
                    code = "MISSING_CRS"
                elif any(not _raster_grids_equal(values[0], value) for value in values[1:]):
                    code = "RASTER_GRID_INCOMPATIBLE"
            elif requirement_type == "valid_geometry":
                if any(_inspection(state, item).get("geometry_valid") is False for item in related):
                    code = "INVALID_GEOMETRY"
            elif requirement_type == "required_fields":
                if self._required_fields_missing(requirement, related, state):
                    code = "SCHEMA_REQUIREMENT_UNRESOLVED"
            elif requirement_type == "schema_grounded":
                if self._schema_grounding_missing(requirement, related, state):
                    code = "SCHEMA_REQUIREMENT_UNRESOLVED"
            elif requirement_type in {"coordinate_fields", "spatializable_table"}:
                if self._coordinate_requirement_missing(requirement, related, state):
                    code = "COORDINATE_FIELDS_UNRESOLVED"
            elif requirement_type == "temporal_range":
                params = dict(requirement.parameters or {})
                requested_start = str(params.get("start") or params.get("year") or "")
                requested_end = str(params.get("end") or requested_start)
                requested_field = str(params.get("field") or "").strip()
                for item in related:
                    inspection = _inspection(state, item)
                    candidates = list(inspection.get("temporal_candidates") or [])
                    if requested_field:
                        candidates = [c for c in candidates if str(c.get("field")) == requested_field]
                    if not candidates or not any(
                        _coverage_matches_request(candidate, requested_start, requested_end)
                        for candidate in candidates
                    ):
                        code = "TEMPORAL_REQUIREMENT_UNMET"
                        break
            elif requirement_type == "spatial_coverage" and len(related) >= 2:
                data = _inspection(state, related[0])
                target = _inspection(state, related[1])
                if not data.get("crs") or not target.get("crs"):
                    code = "MISSING_CRS"
                else:
                    covers = _bounds_cover(
                        data.get("bounds"), data.get("crs"),
                        target.get("bounds"), target.get("crs"),
                    )
                    if covers is not True:
                        code = "SPATIAL_COVERAGE_INSUFFICIENT"
            elif requirement_type == "max_resolution":
                params = dict(requirement.parameters or {})
                try:
                    maximum = float(params.get("value", 0) or 0)
                except (TypeError, ValueError):
                    maximum = 0.0
                if maximum:
                    for item in related:
                        inspection = _inspection(state, item)
                        resolution = inspection.get("resolution") or []
                        if not resolution:
                            code = "RESOLUTION_REQUIREMENT_UNMET"
                            break
                        values = [abs(float(v)) for v in resolution]
                        unit = str(params.get("unit") or "meter").lower()
                        if unit in {"meter", "meters", "metre", "metres", "m"}:
                            factor = inspection.get("linear_unit_to_meter")
                            if inspection.get("is_projected") and factor:
                                values = [v * float(factor) for v in values]
                            elif inspection.get("is_geographic"):
                                # Angular cell size cannot be compared directly with a
                                # metric threshold.  Treat it as a compatibility defect.
                                code = "RESOLUTION_REQUIREMENT_UNMET"
                                break
                        if max(values) > maximum:
                            code = "RESOLUTION_REQUIREMENT_UNMET"
                            break
            elif requirement_type == "format_compatible":
                allowed = {_format_key(item) for item in dict(requirement.parameters or {}).get("formats", [])}
                if allowed and any(
                    _format_key(_inspection(state, item).get("format")) not in allowed
                    for item in related
                ):
                    code = "FORMAT_UNSUPPORTED"
            elif requirement_type == "crs_present":
                if any(not _inspection(state, item).get("crs") for item in related):
                    code = "MISSING_CRS"

            if code:
                requirement.status = "violated"
                requirement.satisfied_by_evidence_ids = []
                computationally_resolvable = code in {
                    "METRIC_DISTANCE_UNSUPPORTED",
                    "CRS_INCOMPATIBLE",
                    "RASTER_GRID_INCOMPATIBLE",
                    "INVALID_GEOMETRY",
                    "SCHEMA_REQUIREMENT_UNRESOLVED",
                    "COORDINATE_FIELDS_UNRESOLVED",
                    "FORMAT_UNSUPPORTED",
                    "RESOLUTION_REQUIREMENT_UNMET",
                    "TEMPORAL_REQUIREMENT_UNMET",
                    "SPATIAL_COVERAGE_INSUFFICIENT",
                    "ANALYTICAL_COMMITMENT_MISSING",
                    "COMMITMENT_DISCLOSURE_MISSING",
                }
                artifact_ids = [item.artifact_id for item in related]
                violation = next(
                    (
                        item
                        for item in state.violations.values()
                        if item.status == "active"
                        and item.task_id == task_id
                        and item.code == code
                        and requirement.requirement_id in item.requirement_ids
                        and item.artifact_ids == artifact_ids
                    ),
                    None,
                )
                if violation is None:
                    violation = ViolationRecord(
                        code=code,
                        layer="deterministic_requirement",
                        task_id=task_id,
                        artifact_ids=artifact_ids,
                        requirement_ids=[requirement.requirement_id],
                        evidence_ids=evidence_ids,
                        blocking=requirement.blocking,
                        deterministic=True,
                        computationally_resolvable=computationally_resolvable,
                        created_in_plan_version=max(1, state.current_plan_version),
                    )
                    state.violations[violation.violation_id] = violation
                    state.record_event(
                        "deterministic_violation_detected",
                        actor="invariant_engine",
                        task_id=task_id,
                        reason=f"Detected blocking requirement violation: {code}.",
                        metadata={
                            "violation_id": violation.violation_id,
                            "requirement_id": requirement.requirement_id,
                            "code": code,
                            "artifact_ids": artifact_ids,
                            "computationally_resolvable": computationally_resolvable,
                        },
                    )
                else:
                    violation.evidence_ids = sorted(set(violation.evidence_ids) | set(evidence_ids))
                violations.append(violation)
            else:
                requirement.status = "satisfied"
                requirement.satisfied_by_evidence_ids = evidence_ids
                for existing in state.violations.values():
                    if (
                        existing.status == "active"
                        and requirement.requirement_id in existing.requirement_ids
                    ):
                        existing.status = "resolved"
                        existing.resolved_by_evidence_ids = list(evidence_ids)
                        existing.resolved_in_plan_version = max(1, state.current_plan_version)
                        existing.resolved_at = utc_now()
                        state.record_event(
                            "deterministic_violation_resolved",
                            actor="invariant_engine",
                            task_id=task_id,
                            reason=(
                                f"Requirement {requirement.requirement_id} is now "
                                "satisfied by current evidence."
                            ),
                            metadata={
                                "violation_id": existing.violation_id,
                                "requirement_id": requirement.requirement_id,
                                "resolved_by_evidence_ids": list(evidence_ids),
                                "resolved_in_plan_version": existing.resolved_in_plan_version,
                            },
                        )
        return violations
