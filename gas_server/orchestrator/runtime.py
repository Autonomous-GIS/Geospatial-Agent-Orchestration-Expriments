"""Construct the production runtime behind public execute and resume modes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gas_server.core.config import DATA_DIR, PROJECT_ROOT
from gas_server.orchestrator.artifacts.evidence import EvidenceEngine
from gas_server.orchestrator.binding.binder import Binder
from gas_server.orchestrator.binding.capability_loader import load_local_capabilities
from gas_server.orchestrator.components.agent_client import AgentComponentClient
from gas_server.orchestrator.components.planner import PlannerComponent
from gas_server.orchestrator.components.qc import QCComponent
from gas_server.orchestrator.components.replanner import ReplannerComponent
from gas_server.orchestrator.components.technical import TechnicalTransitionReviewer
from gas_server.orchestrator.config import STORE_PATH, features_for_condition
from gas_server.orchestrator.core.controller import LoopController
from gas_server.orchestrator.core.models import ArtifactState, ValidationStatus, WorkflowState
from gas_server.orchestrator.execution.executor import WorkerExecutor
from gas_server.orchestrator.execution.local_transport import LocalRegistryTransport
from gas_server.orchestrator.execution.gas_http_transport import GasHttpTransport
from gas_server.orchestrator.persistence.resume import ResumeService
from gas_server.orchestrator.persistence.sqlite_store import SQLiteWorkflowStore
from gas_server.orchestrator.projection.canvas import (
    workflow_plan_projection,
    write_canvas_projection,
)
from gas_server.orchestrator.recovery.local import LocalRecoveryManager


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AutonomousRuntime:
    def __init__(self, public_agent: Any):
        self.public_agent = public_agent

    def _client(self) -> AgentComponentClient:
        return AgentComponentClient(
            chat_completion=self.public_agent._chat_completion,
            extract_json=self.public_agent._extract_json,
            model_name=lambda: str(self.public_agent.model),
        )

    @staticmethod
    def _store(parameters: dict[str, Any]) -> SQLiteWorkflowStore:
        return SQLiteWorkflowStore(parameters.get("workflow_store_path") or STORE_PATH)

    @staticmethod
    def _capabilities():
        return load_local_capabilities(PROJECT_ROOT / "gas_server" / "capabilities")

    def _controller(
        self,
        state: WorkflowState,
        store: SQLiteWorkflowStore,
        parameters: dict[str, Any] | None = None,
    ):
        parameters = parameters or {}
        features = features_for_condition(state.feature_profile)
        capabilities, snapshots = self._capabilities()
        client = self._client()
        planner = PlannerComponent(
            client,
            guidance_mode=features.planner_mode,
            capability_snapshots=snapshots,
        )
        reviewer = (
            TechnicalTransitionReviewer()
            if features.qc_mode == "none"
            else QCComponent(client, guidance_mode=features.qc_mode)
        )
        replanner = (
            None
            if features.replanner_mode == "none"
            else ReplannerComponent(
                client,
                guidance_mode=features.replanner_mode,
                capability_snapshots=snapshots,
            )
        )
        worker_transport = (
            GasHttpTransport(
                timeout_seconds=float(parameters.get("worker_timeout_seconds", 120.0)),
                headers=dict(parameters.get("worker_headers") or {}),
            )
            if parameters.get("worker_transport") == "http"
            else LocalRegistryTransport()
        )
        return LoopController(
            store=store,
            binder=Binder(capabilities),
            executor=WorkerExecutor(worker_transport),
            reviewer=reviewer,
            planner=planner,
            local_recovery=LocalRecoveryManager(),
            replanner=replanner,
        )

    def execute(
        self,
        *,
        query: str,
        input_paths: list[str],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        condition = str(parameters.get("condition") or parameters.get("feature_profile") or "C2").upper()
        features = features_for_condition(condition)
        state = WorkflowState(
            user_goal=query,
            feature_profile=condition,
            feature_config=asdict(features),
        )
        roles = list(parameters.get("input_roles") or [])
        evidence_engine = EvidenceEngine()
        initial_artifacts = []
        for index, value in enumerate(input_paths):
            path = Path(value).resolve(strict=True)
            artifact = ArtifactState(
                location=str(path),
                content_hash=_hash_file(path),
                format=path.suffix.lower().lstrip(".") or None,
                role=roles[index] if index < len(roles) else f"input_{index + 1}",
                validation_status=ValidationStatus.PASS,
                metadata={"source": "user_input"},
            )
            state.artifacts[artifact.artifact_id] = artifact
            initial_artifacts.append(artifact)
        if initial_artifacts:
            evidence_engine.inspect_artifacts(state, initial_artifacts)

        store = self._store(parameters)
        state = store.create(state)
        result = self._controller(state, store, parameters).run(state.workflow_id)
        return self._project(result)

    def resume(self, *, parameters: dict[str, Any]) -> dict[str, Any]:
        store = self._store(parameters)
        payload = parameters.get("clarification") or {}
        state = ResumeService(store).answer(
            workflow_id=str(parameters["workflow_id"]),
            expected_state_version=int(parameters["expected_state_version"]),
            clarification_id=str(payload["clarification_id"]),
            answer=payload.get("answer"),
        )
        result = self._controller(state, store, parameters).run(state.workflow_id)
        return self._project(result)

    def _project(self, state: WorkflowState) -> dict[str, Any]:
        canvas_path = write_canvas_projection(state, DATA_DIR / "orchestrator" / "canvases")
        final_paths = [
            state.artifacts[item].location
            for item in state.final_artifact_ids
            if item in state.artifacts
        ]
        clarification = (
            state.human_clarification.model_dump(mode="json")
            if state.human_clarification and not state.human_clarification.answered
            else None
        )
        return {
            "agent_name": self.public_agent.agent_name,
            "agent_version": self.public_agent.agent_version,
            "model": self.public_agent.model,
            "outputs": {
                "text": state.final_disclosure or state.terminal_reason or "Workflow state updated.",
                "workflow_id": state.workflow_id,
                "terminal_state": state.terminal_state.value if state.terminal_state else None,
                "final_artifacts": final_paths,
                "artifacts": final_paths,
                "workflow_plan": workflow_plan_projection(state),
                "workflow_json_file": str(canvas_path),
                "clarification": clarification,
                "active_commitments": [
                    item.model_dump(mode="json")
                    for item in state.commitments.values()
                    if item.active
                ],
            },
            "workflow_state": state.model_dump(mode="json"),
            "metrics": {
                "worker_calls": state.budgets.worker_calls,
                "model_calls": state.budgets.total_model_calls,
                "total_tokens": state.budgets.total_tokens,
                "local_recoveries": state.budgets.local_recoveries,
                "replans": state.budgets.replans,
                "artifact_count": len(state.artifacts),
            },
            "orchestration": {
                "state_version": state.state_version,
                "plan_version": state.current_plan_version,
                "feature_profile": state.feature_profile,
                "component_call_counts": {
                    component: sum(1 for item in state.component_calls if item.component == component)
                    for component in ("planner", "qc", "replanner")
                },
                "budget_disposition": (
                    "exhausted"
                    if state.terminal_state and state.terminal_state.value == "budget_exhausted"
                    else "within_budget"
                ),
            },
        }
