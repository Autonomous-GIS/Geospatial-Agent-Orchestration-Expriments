"""Independent Planner component and strict plan commit boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from gas_server.orchestrator.binding.binder import capability_supports_expected_outputs
from gas_server.orchestrator.components.common import ComponentClient, stable_hash
from gas_server.orchestrator.core.budgets import reserve_model_call
from gas_server.orchestrator.core.graph import (
    derive_edges,
    topological_order,
    validate_task_graph,
)
from gas_server.orchestrator.core.models import (
    AnalyticalCommitment,
    ComponentCallRecord,
    PlanVersion,
    Requirement,
    StrictModel,
    TaskState,
    WorkflowState,
)
from gas_server.orchestrator.requirements.derivation import (
    SUPPORTED_REQUIREMENT_TYPES,
    derive_operation_requirements,
)


class PlannerProposal(StrictModel):
    summary: str
    tasks: list[TaskState] = Field(min_length=1)
    requirements: list[Requirement] = Field(default_factory=list)
    commitments: list[AnalyticalCommitment] = Field(default_factory=list)
    open_questions: list[dict[str, Any]] = Field(default_factory=list)
    required_terminal_roles: list[str] = Field(default_factory=list)

    @field_validator("tasks", mode="before")
    @classmethod
    def _discard_controller_owned_task_state(cls, value):
        if not isinstance(value, list):
            return value
        controller_fields = {
            "status",
            "attempts",
            "active_binding_id",
            "output_artifact_ids",
            "last_error",
            "created_in_plan_version",
            "superseded_in_plan_version",
        }
        normalized = []
        for item in value:
            payload = (
                item.model_dump(mode="python")
                if isinstance(item, TaskState)
                else dict(item)
                if isinstance(item, dict)
                else item
            )
            if isinstance(payload, dict):
                payload = {
                    key: field_value
                    for key, field_value in payload.items()
                    if key not in controller_fields
                }
                requirements = payload.get("input_requirements")
                if isinstance(requirements, list):
                    seen_roles: set[str] = set()
                    normalized_requirements = []
                    for index, requirement in enumerate(requirements):
                        requirement_payload = (
                            dict(requirement)
                            if isinstance(requirement, dict)
                            else requirement
                        )
                        if isinstance(requirement_payload, dict):
                            role = str(requirement_payload.get("role") or "")
                            if role in seen_roles:
                                requirement_payload["role"] = (
                                    f"__proposed_role_alias_{index}"
                                )
                            seen_roles.add(role)
                        normalized_requirements.append(requirement_payload)
                    payload["input_requirements"] = normalized_requirements
            normalized.append(payload)
        return normalized


class PlannerComponent:
    component_name = "planner"

    def __init__(
        self,
        client: ComponentClient,
        *,
        guidance_mode: str,
        capability_snapshots: list[dict[str, Any]],
    ):
        if guidance_mode not in {"generic", "gis_aware"}:
            raise ValueError("Planner guidance mode must be generic or gis_aware.")
        self.client = client
        self.guidance_mode = guidance_mode
        self.capability_snapshots = capability_snapshots

    def _prompt(self) -> str:
        suffix = "gis" if self.guidance_mode == "gis_aware" else "generic"
        path = Path(__file__).parent / "prompts" / f"planner_{suffix}.txt"
        prompt = path.read_text(encoding="utf-8")
        if self.guidance_mode != "gis_aware":
            return prompt
        return prompt + r"""

RUNTIME-COMPLETENESS ADDENDUM (GIS-AWARE MODE)
- Plan the complete end-to-end transformation chain required by the user goal. A plan is not complete merely because each proposed task can execute.
- Populate required_terminal_roles with every artifact role that must exist before the original goal can truthfully be declared complete. Every such role must be produced by the plan.
- Do not omit prerequisite transformations that are semantically necessary for a downstream operation. Examples of general rules: coordinate tables must become spatial features before vector predicates; metric distance requires a suitable metric CRS; binary spatial operations require compatible CRSs; geometry-consuming operations require valid geometry; workers must receive supported formats; runtime field names must be grounded from observed schema; explicit year/coverage/resolution criteria must be satisfied by the selected candidate; multi-raster arithmetic requires compatible grids; raster-vector operations require compatible CRSs.
- Treat retrieved files as candidates, not automatically valid analytical inputs. Do not invent runtime properties that are not yet observed. When a candidate's fitness cannot be known until execution, retain the user criterion as an explicit requirement so runtime QC can evaluate it.
- Unknown consequential CRS information is not permission to guess. Plan the supported analytical operation and let runtime evidence route clarification if the CRS remains genuinely unknown.
- Keep CRS definition and coordinate transformation distinct: assign_crs only defines metadata for coordinates whose CRS is genuinely unknown and only when the user or trusted persisted evidence explicitly supplies that CRS. If an input already has a known CRS and a different CRS is needed, use reproject. Never use assign_crs as a shortcut for reprojection, and never infer a missing CRS from another nearby layer.
- When the user requests an areal aggregation or choropleth but does not state the analytical spatial unit and multiple defensible units may exist, make one explicit AnalyticalCommitment, propagate it consistently, and ensure it is disclosed. Do not create such a commitment when the user already specifies the unit.
- Prefer the smallest valid workflow. If evidence shows inputs already satisfy a precondition, do not add a defensive repair merely 'just in case'.
- Use only advertised operations/capabilities and only supported requirement types supplied in the input payload.
"""

    @staticmethod
    def _goal_text(value: str) -> str:
        return " ".join(str(value or "").lower().replace("-", " ").split())

    @classmethod
    def _needs_analytical_unit_commitment(cls, user_goal: str) -> bool:
        """Detect a genuinely underspecified areal-unit choice from the user wording.

        This is intentionally semantic and benchmark-neutral: it only flags requests
        whose result depends on choosing an aggregation geography while the user did
        not name that geography.
        """
        text = cls._goal_text(user_goal)
        aggregation_signal = any(
            token in text
            for token in (
                "choropleth",
                "distributed across",
                "distribution across",
                "aggregate across",
                "counts across",
                "count across",
                "by area",
                "per area",
            )
        )
        if not aggregation_signal:
            return False
        explicit_units = (
            "census tract",
            "tract level",
            "tract-level",
            "block group",
            "county subdivision",
            "municipality",
            "borough",
            "zip code",
            "postal code",
            "grid cell",
            "hexagon",
            "h3",
            "by county",
            "per county",
            "by state",
            "per state",
        )
        return not any(unit in text for unit in explicit_units)

    @staticmethod
    def _commitment_looks_like_spatial_unit(commitment: AnalyticalCommitment) -> bool:
        payload = commitment.model_dump(mode="json")
        text = " ".join(
            str(payload.get(key) or "").lower().replace("_", " ").replace("-", " ")
            for key in ("key", "value", "rationale", "evidence_source")
        )
        return any(token in text for token in ("spatial unit", "analytical unit", "aggregation unit", "geographic unit"))

    @staticmethod
    def _produced_roles(candidate: WorkflowState) -> set[str]:
        return {
            output.role
            for task in candidate.tasks.values()
            for output in task.expected_outputs
            if output.role
        }

    def _validate_goal_contract(
        self,
        candidate: WorkflowState,
        proposal: PlannerProposal,
    ) -> None:
        if self.guidance_mode != "gis_aware":
            return

        terminal_roles = [str(role).strip() for role in proposal.required_terminal_roles if str(role).strip()]
        if not terminal_roles:
            raise ValueError(
                "GIS-aware Planner must declare required_terminal_roles so completion can be checked against the original goal."
            )
        if len(terminal_roles) != len(set(terminal_roles)):
            raise ValueError("required_terminal_roles contains duplicate roles.")
        produced = self._produced_roles(candidate)
        missing = sorted(set(terminal_roles) - produced)
        if missing:
            raise ValueError(
                "Planner declared terminal roles that no task produces: " + ", ".join(missing)
            )

        # A required terminal role must remain reachable from the committed graph.
        # validate_task_graph checks dependency validity; here we additionally make
        # the end-to-end completion contract explicit and auditable.
        producers = {
            role: [
                task.task_id
                for task in candidate.tasks.values()
                if any(output.role == role for output in task.expected_outputs)
            ]
            for role in terminal_roles
        }
        if any(not ids for ids in producers.values()):
            raise ValueError("Every required terminal role needs an active producer task.")

        if self._needs_analytical_unit_commitment(candidate.user_goal):
            commitments = [
                item
                for item in proposal.commitments
                if self._commitment_looks_like_spatial_unit(item)
            ]
            if not commitments:
                raise ValueError(
                    "The user goal requires an areal aggregation/choropleth but does not specify the analytical spatial unit. "
                    "Record one defensible spatial-unit AnalyticalCommitment and propagate/disclose it."
                )

    @staticmethod
    def _canonical_format(value: str | None) -> str:
        normalized = str(value or "").lower().lstrip(".")
        return {
            "tif": "geotiff",
            "tiff": "geotiff",
            "geopackage": "gpkg",
        }.get(normalized, normalized)

    def _ground_unambiguous_roles(self, candidate: WorkflowState) -> None:
        initial_sources = [
            {
                "role": artifact.role,
                "data_model": artifact.data_model,
                "formats": [artifact.format] if artifact.format else [],
                "source": "initial_artifact",
            }
            for artifact in candidate.artifacts.values()
            if artifact.producer_task_id is None
        ]
        for task_id in topological_order(candidate.tasks):
            task = candidate.tasks[task_id]
            sources = list(initial_sources)
            for dependency_id in task.depends_on:
                for output in candidate.tasks[dependency_id].expected_outputs:
                    sources.append(
                        {
                            "role": output.role,
                            "data_model": output.data_model,
                            "formats": list(output.formats),
                            "source": dependency_id,
                        }
                    )

            def compatible(requirement, source) -> bool:
                if (
                    requirement.data_model
                    and source["data_model"]
                    and requirement.data_model != source["data_model"]
                ):
                    return False
                required_formats = {
                    self._canonical_format(value) for value in requirement.formats
                }
                source_formats = {
                    self._canonical_format(value) for value in source["formats"]
                }
                return not (
                    required_formats
                    and source_formats
                    and not (required_formats & source_formats)
                )

            for requirement in task.input_requirements:
                exact = [
                    source
                    for source in sources
                    if source["role"] == requirement.role
                    and compatible(requirement, source)
                ]
                if exact:
                    continue
                matches = [
                    source for source in sources if compatible(requirement, source)
                ]
                unique_roles = {source["role"] for source in matches}
                if len(unique_roles) != 1:
                    continue
                old_role = requirement.role
                source = matches[0]
                requirement.role = source["role"]
                candidate.record_event(
                    "planner_artifact_role_grounded",
                    actor="planner_validator",
                    task_id=task.task_id,
                    reason=(
                        f"Grounded proposed role {old_role!r} to the only compatible "
                        f"available role {requirement.role!r}."
                    ),
                    metadata={
                        "proposed_role": old_role,
                        "grounded_role": requirement.role,
                        "source": source["source"],
                    },
                )
            grounded_roles = [item.role for item in task.input_requirements]
            if len(grounded_roles) != len(set(grounded_roles)):
                raise ValueError(
                    f"Task {task.task_id!r} has ambiguous input roles after "
                    "deterministic role grounding."
                )

    @staticmethod
    def _ground_unambiguous_requirement_task(
        candidate: WorkflowState,
        requirement: Requirement,
    ) -> None:
        """Attach a Planner requirement when its semantic target is unique.

        The LLM occasionally emits a stale or invented task id even though the
        requirement itself clearly belongs to exactly one task.  Correcting that
        reference is deterministic and safer than spending all Planner retries on
        an identifier typo.  Ambiguous references are never guessed.
        """
        if requirement.task_id in candidate.tasks:
            return

        original_task_id = requirement.task_id
        referenced_roles = set(requirement.artifact_roles)
        candidates = []
        for task in candidate.tasks.values():
            task_roles = {item.role for item in task.input_requirements}
            operation_types = {
                item.requirement_type for item in derive_operation_requirements(task)
            }
            roles_match = bool(referenced_roles) and referenced_roles.issubset(
                task_roles
            )
            type_matches = requirement.requirement_type in operation_types
            if roles_match or (not referenced_roles and type_matches):
                candidates.append(task)

        if len(candidates) != 1:
            return

        task = candidates[0]
        requirement.task_id = task.task_id
        candidate.record_event(
            "planner_requirement_task_grounded",
            actor="planner_validator",
            task_id=task.task_id,
            reason=(
                f"Grounded requirement {requirement.requirement_id!r} to the "
                "only semantically compatible task."
            ),
            metadata={
                "requirement_id": requirement.requirement_id,
                "requirement_type": requirement.requirement_type,
                "artifact_roles": list(requirement.artifact_roles),
                "proposed_task_id": original_task_id,
                "grounded_task_id": task.task_id,
            },
        )

    @staticmethod
    def _latest_artifact_inspection(
        state: WorkflowState, artifact
    ) -> dict[str, Any]:
        for evidence_id in reversed(getattr(artifact, "evidence_ids", []) or []):
            if evidence_id not in state.evidence:
                continue
            record = state.evidence[evidence_id]
            if getattr(record, "evidence_type", None) == "artifact_inspection":
                return dict(getattr(record, "values", {}) or {})
        return {}

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

    @staticmethod
    def _has_user_asserted_crs(
        state: WorkflowState, *, target_crs: str | None = None
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
        return False

    def _validate_initial_crs_assignment_safety(
        self, state: WorkflowState, candidate: WorkflowState
    ) -> None:
        """Prevent the Planner from guessing or relabelling CRS metadata.

        This rule is benchmark-neutral.  `assign_crs` is metadata definition, not
        coordinate transformation.  It is safe only when the CRS is genuinely
        missing and an explicit trusted assertion supplies the value.
        """
        if self.guidance_mode != "gis_aware":
            return

        initial_by_role: dict[str, list[Any]] = {}
        for artifact in state.artifacts.values():
            if artifact.producer_task_id is not None:
                continue
            initial_by_role.setdefault(str(artifact.role), []).append(artifact)

        for task in candidate.tasks.values():
            if str(task.operation or "").lower() != "assign_crs":
                continue
            target_crs = str((task.parameters or {}).get("target_crs") or "").strip()
            if not target_crs:
                raise ValueError(
                    f"Task {task.task_id!r} uses assign_crs without an explicit target_crs."
                )

            matched = []
            for requirement in task.input_requirements:
                matched.extend(initial_by_role.get(str(requirement.role), []))
            if not matched:
                # The task may consume a future runtime artifact; its safety cannot
                # yet be established from initial evidence, so leave it to runtime.
                continue

            trusted = self._has_user_asserted_crs(
                state, target_crs=target_crs
            ) or self._goal_explicitly_assigns_crs(state.user_goal, target_crs)
            for artifact in matched:
                values = self._latest_artifact_inspection(state, artifact)
                observed_crs = str(
                    values.get("crs_authority") or values.get("crs") or ""
                ).strip()
                if observed_crs:
                    if trusted:
                        # Explicit metadata correction is a legitimate user request.
                        # Do not reinterpret it as analytical reprojection.
                        continue
                    raise ValueError(
                        f"Task {task.task_id!r} uses assign_crs on role {artifact.role!r}, "
                        f"but runtime evidence already reports {observed_crs}. Use reproject "
                        "when coordinates must be transformed; do not relabel known coordinates "
                        "unless the user explicitly requested a metadata correction."
                    )
                if not trusted:
                    raise ValueError(
                        f"Task {task.task_id!r} attempts to assign {target_crs} to role "
                        f"{artifact.role!r} whose CRS is unknown, but no explicit user/trusted "
                        "CRS assertion exists. Do not guess; keep the analytical task and let "
                        "runtime clarification resolve the missing CRS."
                    )

    def plan(self, state: WorkflowState) -> WorkflowState:
        if state.tasks:
            raise ValueError("Planner cannot overwrite an existing task graph.")
        prompt = self._prompt()
        known_input_artifacts = []
        for item in state.artifacts.values():
            if item.producer_task_id is not None:
                continue
            inspection = next(
                (
                    state.evidence[evidence_id]
                    for evidence_id in reversed(item.evidence_ids)
                    if evidence_id in state.evidence
                    and state.evidence[evidence_id].evidence_type
                    == "artifact_inspection"
                ),
                None,
            )
            known_input_artifacts.append(
                {
                    "artifact_id": item.artifact_id,
                    "role": item.role,
                    "format": item.format,
                    "data_model": item.data_model,
                    "evidence_ids": item.evidence_ids,
                    "bounded_inspection": (
                        inspection.model_dump(mode="json") if inspection else None
                    ),
                }
            )
        input_payload = {
            "user_goal": state.user_goal,
            "known_input_artifacts": known_input_artifacts,
            "worker_capabilities": self.capability_snapshots,
            "guidance_mode": self.guidance_mode,
            "supported_requirement_types": sorted(SUPPORTED_REQUIREMENT_TYPES),
            "completion_contract": {
                "required_terminal_roles_required": self.guidance_mode == "gis_aware",
                "analytical_unit_commitment_needed": (
                    self._needs_analytical_unit_commitment(state.user_goal)
                    if self.guidance_mode == "gis_aware"
                    else False
                ),
                "instruction": (
                    "Represent the complete original goal, not only a locally executable prefix. "
                    "Declare the final artifact roles that must exist before completion."
                ),
            },
        }
        responses = []
        proposal = None
        candidate = None
        last_error: Exception | None = None
        for correction_round in range(state.budgets.planner_retry_limit + 1):
            reserve_model_call(state)
            response = self.client.complete(
                component=self.component_name,
                system_prompt=prompt,
                input_payload=input_payload,
                output_schema=PlannerProposal.model_json_schema(),
                retry_limit=0,
            )
            responses.append(response)
            state.budgets.total_tokens += response.input_tokens + response.output_tokens
            try:
                proposal = PlannerProposal.model_validate(response.payload)
                task_map = {task.task_id: task for task in proposal.tasks}
                if len(task_map) != len(proposal.tasks):
                    raise ValueError("Planner returned duplicate task IDs.")
                for task in proposal.tasks:
                    eligible = []
                    for capability in self.capability_snapshots:
                        operations = set(capability.get("operations") or [])
                        if task.operation not in operations and "*" not in operations:
                            continue
                        supported = operations | {str(capability.get("agent_id") or "")}
                        if (
                            set(task.required_capabilities).issubset(supported)
                            and capability_supports_expected_outputs(capability, task)
                        ):
                            eligible.append(capability.get("agent_id"))
                    if not eligible:
                        raise ValueError(
                            f"Task {task.task_id!r} uses unsupported operation "
                            f"{task.operation!r} or capability requirements "
                            f"{task.required_capabilities!r}."
                        )
                candidate = state.model_copy(deep=True)
                candidate.tasks = task_map
                candidate.current_plan_version = 1
                candidate.edges = derive_edges(task_map)
                self._ground_unambiguous_roles(candidate)
                self._validate_initial_crs_assignment_safety(state, candidate)

                for requirement in proposal.requirements:
                    if (
                        self.guidance_mode == "generic"
                        and requirement.provenance.value == "PLANNER_INFERRED"
                    ):
                        raise ValueError(
                            "Generic Planner cannot create GIS Planner-inferred requirements."
                        )
                    if requirement.requirement_type not in SUPPORTED_REQUIREMENT_TYPES:
                        raise ValueError(
                            f"Planner requirement {requirement.requirement_id!r} uses "
                            f"unsupported type {requirement.requirement_type!r}."
                        )
                    self._ground_unambiguous_requirement_task(candidate, requirement)
                    if requirement.blocking and requirement.task_id not in task_map:
                        raise ValueError(
                            f"Blocking Planner requirement {requirement.requirement_id!r} "
                            "must reference an existing task_id."
                        )
                    candidate.requirements[requirement.requirement_id] = requirement
                    if (
                        requirement.task_id in candidate.tasks
                        and requirement.requirement_id
                        not in candidate.tasks[requirement.task_id].requirement_ids
                    ):
                        candidate.tasks[requirement.task_id].requirement_ids.append(
                            requirement.requirement_id
                        )
                for task in candidate.tasks.values():
                    for requirement in derive_operation_requirements(task):
                        candidate.requirements[requirement.requirement_id] = requirement
                        task.requirement_ids.append(requirement.requirement_id)
                for commitment in proposal.commitments:
                    if self.guidance_mode == "generic":
                        raise ValueError(
                            "Generic Planner cannot create GIS analytical commitments."
                        )
                    candidate.commitments[commitment.commitment_id] = commitment
                candidate.required_terminal_roles = list(dict.fromkeys(
                    str(role).strip()
                    for role in proposal.required_terminal_roles
                    if str(role).strip()
                ))
                validate_task_graph(candidate)
                self._validate_goal_contract(candidate, proposal)
                break
            except (ValueError, TypeError) as exc:
                last_error = exc
                candidate = None
                input_payload = dict(input_payload)
                input_payload["previous_plan_validation"] = {
                    "status": "REJECTED",
                    "error": str(exc),
                    "previous_proposal": response.payload,
                    "instruction": (
                        "Return a corrected minimal plan using only advertised worker "
                        "operations and capabilities. Do not create a checking task for "
                        "metadata already reported as unknown; let the supported "
                        "analytical task's runtime preconditions route clarification. "
                        "Every depends_on value must be another task_id in the proposal; "
                        "initial artifact roles belong only in input_requirements and "
                        "must never appear in depends_on. Requirement types are not "
                        "worker operations. If metadata is unknown, remove any checker "
                        "task and any dependency on it; keep only the supported analysis "
                        "whose runtime precondition will route clarification. Ensure "
                        "each task's expected output data model and format match the "
                        "same worker capability. Schema grounding is performed from "
                        "runtime evidence by QC, not by a grounding worker task."
                        " Preserve every known initial artifact role verbatim, and make "
                        "each downstream input role exactly equal to either that initial "
                        "role or an expected output role of a direct dependency; never "
                        "rename roles implicitly. "
                        "Also preserve the entire original objective: declare every required final artifact role in required_terminal_roles, ensure each declared role is produced by a task, and do not stop the plan before every requested transformation/output is represented. "
                        "If the validation error says an analytical spatial-unit commitment is required, create one defensible AnalyticalCommitment and keep it consistent through aggregation and final disclosure. "
                        "CRS safety is strict: known source CRS + different target means reproject; missing source CRS without an explicit user/trusted assertion means do not add assign_crs and let runtime clarification handle it. If a requirement references a nonexistent task id, attach it only to an existing semantically compatible task id from the proposal."
                    ),
                }
        if candidate is None or proposal is None:
            raise ValueError(
                "Planner failed to produce a valid executable plan after "
                f"{len(responses)} attempt(s): {last_error}"
            )
        candidate.plan_versions.append(
            PlanVersion(
                version=1,
                reason="Initial plan committed by independent Planner.",
                task_ids=sorted(candidate.tasks),
                edges=list(candidate.edges),
            )
        )
        candidate.component_calls.append(
            ComponentCallRecord(
                component="planner",
                prompt_hash=stable_hash(prompt),
                tool_schema_hash=stable_hash(PlannerProposal.model_json_schema()),
                model=responses[-1].model,
                provider=responses[-1].provider,
                attempts=sum(item.attempts for item in responses),
                input_tokens=sum(item.input_tokens for item in responses),
                output_tokens=sum(item.output_tokens for item in responses),
                duration_seconds=sum(item.duration_seconds for item in responses),
                disposition=(
                    "successful"
                    if len(responses) == 1
                    else "successful_after_plan_validation_retry"
                ),
            )
        )
        candidate.record_event(
            "initial_plan_committed",
            actor="planner_component",
            reason=proposal.summary,
            metadata={"task_ids": sorted(candidate.tasks)},
        )
        return candidate
