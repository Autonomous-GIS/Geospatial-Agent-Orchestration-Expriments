"""Constraint-Driven BAFO Oracle for GAS Benchmark.

Integrates Challenge Activation Validation, Execution Ledger Events, Physical Artifact
Registry, and Executable Constraint Dispatching with 3-valued BAFO logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.evaluation.artifact_registry import ARTIFACT_REGISTRY, InspectedArtifact
from benchmark.evaluation.challenge_validator import ChallengeActivationValidator
from benchmark.evaluation.constraint_dispatcher import ConstraintDispatcher, ConstraintResult
from benchmark.evaluation.ledger import ExecutionEvent

logger = logging.getLogger(__name__)


PRODUCTION_TO_EVALUATOR_TERMINAL_STATE = {
    "completed": "successful",
    "successful": "successful",
    "success": "successful",
    "task_state_completed": "successful",
    "human_intervention_required": "human_intervention_required",
    "clarification_needed": "human_intervention_required",
    "failed_unrecoverable": "failed",
    "failed": "failed",
    "budget_exhausted": "budget_exhausted",
    "cancelled": "cancelled",
    "invalid_state": "failed",
}


def normalize_terminal_state(value: Any) -> str:
    """Map production terminal vocabulary to the evaluator's stable vocabulary."""
    raw = str(value or "failed").lower()
    return PRODUCTION_TO_EVALUATOR_TERMINAL_STATE.get(raw, raw)


class BenchmarkOracle:
    """Evaluates task execution trajectories for BAFO (Benchmark-Aligned Final Outcome) success."""

    def __init__(self, manifests_dir: Path | None = None, fixtures_dir: Path | None = None):
        self.manifests_dir = manifests_dir or Path(__file__).resolve().parents[1] / "manifests"
        self.fixtures_dir = fixtures_dir or Path(__file__).resolve().parents[1] / "fixtures"
        self._manifests: Dict[str, Dict[str, Any]] = {}
        self.activation_validator = ChallengeActivationValidator(fixtures_dir=self.fixtures_dir)
        self.dispatcher = ConstraintDispatcher()
        self._load_manifests()

    def _load_manifests(self):
        candidate_dirs = [self.manifests_dir]
        for d in candidate_dirs:
            if d.exists():
                for m_file in d.glob("*.json"):
                    try:
                        with open(m_file, "r", encoding="utf-8") as f:
                            entries = json.load(f)
                            for entry in entries:
                                t_id = entry.get("task_id")
                                if t_id:
                                    self._manifests[t_id] = entry
                    except Exception as e:
                        logger.warning("Could not load manifest file %s: %s", m_file, e)

    def evaluate_trajectory(
        self,
        task_id: str,
        trajectory: Dict[str, Any],
        ledger_events: List[ExecutionEvent] | None = None,
        output_artifact_paths: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate a single completed trajectory against the task manifest.

        Parameters
        ----------
        task_id : str
            Unique task identifier (e.g. 'ISS_Q01', 'CCS_Cmp01').
        trajectory : dict
            Execution record containing status, outputs, timing, and parameters.
        ledger_events : list of ExecutionEvent, optional
            Recorded worker service events from ExecutionLedger.
        output_artifact_paths : list of str, optional
            Paths to physical output files created during execution.

        Returns
        -------
        dict
            Evaluation result:
            - bafo_success: Optional[bool] (True, False, or None)
            - evaluability_status: str ('EVALUABLE', 'UNEVALUABLE')
            - challenge_activated: bool
            - trajectory_classification: str ('valid_bafo_pass', 'valid_bafo_fail', 'benchmark_invalid', 'evaluation_invalid')
            - constraint_results: dict of constraint -> ConstraintResult dict
            - diagnostic_summary: str
        """
        manifest = self._manifests.get(task_id)
        if not manifest:
            return {
                "bafo_success": None,
                "evaluability_status": "UNEVALUABLE",
                "challenge_activated": False,
                "trajectory_classification": "evaluation_invalid",
                "error": f"Task manifest not found for task_id: {task_id}",
            }

        # Extract initial input paths
        input_paths = list(trajectory.get("input_artifact_paths") or trajectory.get("input_datasets") or [])
        if not input_paths:
            for art_name in manifest.get("input_artifacts", []):
                p = self.fixtures_dir / art_name
                if p.exists():
                    input_paths.append(str(p))

        # 1. Challenge Activation Validation
        is_activated, unactivated_reasons = self.activation_validator.validate_activation(
            task_manifest=manifest,
            initial_input_paths=input_paths,
        )

        if not is_activated:
            return {
                "task_id": task_id,
                "bafo_success": None,
                "evaluability_status": "UNEVALUABLE",
                "challenge_activated": False,
                "trajectory_classification": "benchmark_invalid",
                "unactivated_reasons": unactivated_reasons,
                "diagnostic_summary": f"Challenge was not active in initial environment: {unactivated_reasons}",
            }

        # 2. Collect Execution Events and Physical Artifacts
        events = list(ledger_events or [])
        if not events and "ledger_events" in trajectory:
            for ev_dict in trajectory["ledger_events"]:
                try:
                    events.append(ExecutionEvent(**ev_dict))
                except Exception:
                    pass

        # Collect and inspect output artifacts
        out_paths = list(output_artifact_paths or [])
        if not out_paths:
            # Check outputs.artifacts or outputs.workflow_steps
            out_paths.extend(trajectory.get("outputs", {}).get("artifacts", []))
            out_paths.extend(trajectory.get("outputs", {}).get("produced_artifacts", []))
            out_paths.extend(trajectory.get("produced_artifacts", []))
            out_paths.extend(trajectory.get("artifact_paths", []))
            state_artifacts = trajectory.get("workflow_state", {}).get("artifacts", {})
            out_paths.extend(
                artifact.get("location")
                for artifact in state_artifacts.values()
                if artifact.get("producer_task_id") and artifact.get("location")
            )
            for ev in events:
                out_paths.extend(ev.output_artifact_paths)

        unique_out_paths = sorted(list(set(p for p in out_paths if Path(p).is_file())))
        inspected_artifacts = []
        for p in unique_out_paths:
            insp = ARTIFACT_REGISTRY.inspect(p)
            if insp:
                inspected_artifacts.append(insp)

        # 3. Evaluate Executable Constraints via Dispatcher
        constraint_results = self.dispatcher.evaluate_constraints(
            task_manifest=manifest,
            events=events,
            artifacts=inspected_artifacts,
            response=trajectory.get("outputs", {}) or trajectory,
        )

        # 4. Check Terminal State
        acceptable_states = manifest.get("acceptable_terminal_states", ["successful"])
        production_terminal_state = str(
            trajectory.get("workflow_state", {}).get("terminal_state")
            or trajectory.get("status")
            or "failed"
        ).lower()
        traj_status = normalize_terminal_state(production_terminal_state)

        state_valid = traj_status in acceptable_states

        # 5. Three-Valued BAFO Decision
        has_unevaluable = any(r.status == "UNEVALUABLE" for r in constraint_results.values())
        has_fail = any(r.status == "FAIL" for r in constraint_results.values()) or not state_valid
        all_pass = all(r.status == "PASS" for r in constraint_results.values()) and state_valid

        if has_unevaluable and not has_fail:
            bafo_success = None
            evaluability_status = "UNEVALUABLE"
            classification = "evaluation_invalid"
        elif all_pass:
            bafo_success = True
            evaluability_status = "EVALUABLE"
            classification = "valid_bafo_pass"
        else:
            bafo_success = False
            evaluability_status = "EVALUABLE"
            classification = "valid_bafo_fail"

        # Build diagnostic summary
        passed_keys = [k for k, r in constraint_results.items() if r.status == "PASS"]
        failed_keys = [k for k, r in constraint_results.items() if r.status == "FAIL"]
        uneval_keys = [k for k, r in constraint_results.items() if r.status == "UNEVALUABLE"]

        diagnostic_summary = (
            f"BAFO: {bafo_success} | State: {traj_status} (Valid: {state_valid}) | "
            f"Constraints: {len(passed_keys)} PASS, {len(failed_keys)} FAIL, {len(uneval_keys)} UNEVALUABLE | "
            f"Events Recorded: {len(events)} | Artifacts Inspected: {len(inspected_artifacts)}"
        )

        return {
            "task_id": task_id,
            "bafo_success": bafo_success,
            "all_challenges_resolved": bool(bafo_success is True),
            "evaluability_status": evaluability_status,
            "challenge_activated": True,
            "trajectory_classification": classification,
            "production_terminal_state": production_terminal_state,
            "terminal_state": traj_status,
            "constraint_results": {k: r.to_dict() for k, r in constraint_results.items()},
            "passed_constraints": passed_keys,
            "failed_constraints": failed_keys,
            "unevaluable_constraints": uneval_keys,
            "inspected_artifacts_count": len(inspected_artifacts),
            "events_count": len(events),
            "diagnostic_summary": diagnostic_summary,
        }
