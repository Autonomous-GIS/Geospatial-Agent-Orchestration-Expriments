"""Randomized 600-Trajectory Schedule Generator and Headless Execution Kernel.

Features:
- Full LLM parity: C0-C5 all use gpt-4o-mini
- Infrastructure Failure Distinction: Catches provider/network errors, marks status 'infrastructure_invalid', and retries the slot
- Fine-grained Latency Logging: total_wall_time, rate_limit_wait_time, provider_retry_wait_time, orchestration_active_time, worker_execution_time
- 30 ISS x 6 conditions x 3 runs = 540 trajectories
- 10 CCS x 2 conditions x 3 runs = 60 trajectories
Total = 600 trajectories.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gas_server.core.config import DATA_DIR
from gas_server.core.llm_client import (
    build_llm_client,
    get_llm_timing_stats,
    reset_llm_timing_stats,
    InfrastructureError,
)
from gas_server.core.service_registry import get_service_registration
from benchmark.baselines.magentic_one_adapter import MagenticOneC0Adapter
from benchmark.evaluation.ledger import ExecutionEvent, ExecutionLedger
from benchmark.evaluation.oracle import BenchmarkOracle
from benchmark.progressive_executor import ProgressiveExecutionKernel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MANIFESTS_DIR = BENCHMARK_DIR / "manifests"
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"


def _oracle_artifact_paths(
    final_artifacts: list[str] | None,
    produced_artifacts: list[str] | None,
) -> list[str]:
    """Expose the full produced derivation set to the independent oracle."""
    return list(dict.fromkeys([*(final_artifacts or []), *(produced_artifacts or [])]))
RESULTS_DIR = BENCHMARK_DIR / "results" / "runs"


def load_suite_tasks(suite_name: str) -> List[Dict[str, Any]]:
    """Load tasks for the specified suite."""
    if suite_name == "dev":
        m_file = BENCHMARK_DIR / "dev_suite" / "dev_manifest.json"
    elif suite_name == "compound":
        m_file = MANIFESTS_DIR / "compound_suite.json"
    else:
        m_file = MANIFESTS_DIR / "isolated_suite.json"

    if not m_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {m_file}")

    with open(m_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_schedule(
    suites: List[str],
    conditions: List[str],
    num_runs: int = 3,
    seed: int = 42,
    task_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Generate a randomized blocked execution schedule."""
    random.seed(seed)
    schedule: List[Dict[str, Any]] = []

    for suite in suites:
        tasks = load_suite_tasks(suite)
        if task_id:
            tasks = [task for task in tasks if task.get("task_id") == task_id]
        else:
            tasks = [
                task
                for task in tasks
                if task.get("development_scope", "matrix") == "matrix"
            ]
        allowed_conds = conditions
        if suite == "compound":
            allowed_conds = [c for c in conditions if c in {"C1", "C2"}]

        for task in tasks:
            for cond in allowed_conds:
                for run_idx in range(1, num_runs + 1):
                    traj_id = f"traj_{task['task_id']}_{cond}_run{run_idx}"
                    schedule_item = {
                        "trajectory_id": traj_id,
                        "task_id": task["task_id"],
                        "suite": suite,
                        "condition": cond,
                        "run_index": run_idx,
                        "user_question": task["user_question"],
                        "input_artifacts": task.get("input_artifacts", []),
                    }
                    if task.get("catalog_manifest"):
                        schedule_item["catalog_manifest"] = task["catalog_manifest"]
                    schedule.append(schedule_item)

    random.shuffle(schedule)
    return schedule


def execute_trajectory(
    item: Dict[str, Any],
    oracle: BenchmarkOracle,
    results_dir: Path,
    max_infra_retries: int = 3,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Execute a single trajectory under the configured condition and score with Oracle.

    Distinguishes genuine orchestration results from transient infrastructure errors.
    If an InfrastructureError occurs, logs status 'infrastructure_invalid' and retries.
    """
    traj_id = item["trajectory_id"]
    task_id = item["task_id"]
    condition = item["condition"]
    question = item["user_question"]

    # Prepare input artifact paths
    input_paths = []
    for art_name in item.get("input_artifacts", []):
        path = FIXTURES_DIR / art_name
        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation input artifact not found in fixture root: {art_name}"
            )
        input_paths.append(str(path))

    for infra_attempt in range(1, max_infra_retries + 1):
        logger.info("Executing %s | Task: %s | Condition: %s (Attempt %d/%d) [Model: %s]", traj_id, task_id, condition, infra_attempt, max_infra_retries, model)

        # Pre-run state reset
        scratch_dir = DATA_DIR / "scratch" / traj_id
        scratch_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = FIXTURES_DIR / "FIXTURE_MANIFEST.json"
        os.environ["GAS_BENCHMARK_CATALOG"] = str(catalog_path.resolve(strict=True))
        reset_llm_timing_stats()

        start_time = time.time()

        try:
            with ExecutionLedger(traj_id) as ledger:
                if condition == "C0":
                    # Magentic-One baseline adapter with strict model parity
                    adapter = MagenticOneC0Adapter(model=model)
                    raw_result = adapter.run(
                        query=question,
                        input_dataset_paths=input_paths,
                    )
                    duration = time.time() - start_time
                    timing = get_llm_timing_stats()
                    events = ledger.get_events()
                    output_artifacts = _oracle_artifact_paths(
                        raw_result.get("outputs", {}).get("artifacts", []),
                        raw_result.get("outputs", {}).get("produced_artifacts", []),
                    )

                    trajectory_record = {
                        "trajectory_id": traj_id,
                        "task_id": task_id,
                        "condition": condition,
                        "model": model,
                        "run_index": item["run_index"],
                        "suite": item["suite"],
                        "user_question": question,
                        "input_artifact_paths": input_paths,
                        "status": raw_result.get("status", "successful"),
                        "total_wall_time": round(duration, 3),
                        "rate_limit_wait_time": timing["rate_limit_wait_time"],
                        "provider_retry_wait_time": timing["provider_retry_wait_time"],
                        "orchestration_active_time": round(max(0.0, duration - timing["rate_limit_wait_time"] - timing["provider_retry_wait_time"] - raw_result.get("worker_execution_time", 0.0)), 3),
                        "worker_execution_time": raw_result.get("worker_execution_time", 0.0),
                        "total_tokens": raw_result.get("total_tokens", 0),
                        "input_tokens": raw_result.get("input_tokens", 0),
                        "output_tokens": raw_result.get("output_tokens", 0),
                        "model_calls": raw_result.get("model_calls", 0),
                        "outputs": raw_result.get("outputs", {}),
                        "produced_artifacts": raw_result.get("outputs", {}).get(
                            "produced_artifacts", []
                        ),
                        "execution_steps": raw_result.get("steps_executed", []),
                        "ledger_events": [e.to_dict() for e in events],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

                else:
                    # GAS Orchestrator conditions (C1-C5)
                    params: Dict[str, Any] = {
                        "condition": condition,
                        "debug": True,
                        "max_steps": 12,
                    }
                    if condition == "C1":
                        params.update({"gis_planner": False, "gis_qc": False, "structural_replan": True})
                    elif condition == "C2":
                        params.update({"gis_planner": False, "gis_qc": True, "structural_replan": True})
                    elif condition == "C3":
                        params.update({"gis_planner": True, "gis_qc": True, "structural_replan": False})
                    elif condition == "C4":
                        params.update({"gis_planner": True, "gis_qc": False, "structural_replan": True})
                    elif condition == "C5":
                        params.update({"gis_planner": True, "gis_qc": True, "structural_replan": True})

                    reg = get_service_registration("autonmous_gis_orchestrator")
                    orchestrator = reg.build_agent()
                    orchestrator.model = model
                    orchestrator.client = build_llm_client(
                        service_name=orchestrator.service_name,
                        model=model,
                    )
                    orchestrator.set_request_parameters(params)

                    kernel = ProgressiveExecutionKernel(
                        orchestrator=orchestrator,
                        condition=condition,
                    )
                    exec_res = kernel.run_pipeline(
                        query=question,
                        initial_input_paths=input_paths,
                        trajectory_id=traj_id,
                    )
                    duration = time.time() - start_time
                    timing = get_llm_timing_stats()
                    events = ledger.get_events()
                    if not events:
                        events = [
                            ExecutionEvent(**item) for item in exec_res.execution_events
                        ]
                    output_artifacts = _oracle_artifact_paths(
                        exec_res.output_artifacts,
                        exec_res.produced_artifacts,
                    )
                    component_calls = exec_res.workflow_state.get("component_calls", [])
                    component_input_tokens = sum(
                        int(call.get("input_tokens", 0) or 0) for call in component_calls
                    )
                    component_output_tokens = sum(
                        int(call.get("output_tokens", 0) or 0) for call in component_calls
                    )

                    trajectory_record = {
                        "trajectory_id": traj_id,
                        "task_id": task_id,
                        "condition": condition,
                        "model": model,
                        "run_index": item["run_index"],
                        "suite": item["suite"],
                        "user_question": question,
                        "input_artifact_paths": input_paths,
                        "status": exec_res.status,
                        "total_wall_time": round(duration, 3),
                        "rate_limit_wait_time": timing["rate_limit_wait_time"],
                        "provider_retry_wait_time": timing["provider_retry_wait_time"],
                        "orchestration_active_time": round(
                            max(
                                0.0,
                                duration
                                - timing["rate_limit_wait_time"]
                                - timing["provider_retry_wait_time"]
                                - exec_res.worker_execution_time,
                            ),
                            3,
                        ),
                        "worker_execution_time": round(exec_res.worker_execution_time, 3),
                        "total_tokens": component_input_tokens + component_output_tokens,
                        "input_tokens": component_input_tokens,
                        "output_tokens": component_output_tokens,
                        "outputs": {
                            "summary": exec_res.summary,
                            "artifacts": exec_res.output_artifacts,
                            "produced_artifacts": exec_res.produced_artifacts,
                            "workflow_plan": exec_res.workflow_plan,
                        },
                        "workflow_state": exec_res.workflow_state,
                        "execution_steps": exec_res.executed_steps,
                        "qc_evaluations": exec_res.qc_evaluations,
                        "qc_blocks": exec_res.qc_blocks,
                        "qc_interventions": exec_res.qc_interventions,
                        "effective_blocks": exec_res.effective_blocks,
                        "effective_interventions": exec_res.effective_interventions,
                        "qc_policy_overrides": exec_res.qc_policy_overrides,
                        "replanning_iterations": exec_res.replanning_iterations,
                        "error": exec_res.error,
                        "ledger_events": [e.to_dict() for e in events],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

                # 3. Constraint-Driven Oracle Scoring
                eval_result = oracle.evaluate_trajectory(
                    task_id=task_id,
                    trajectory=trajectory_record,
                    ledger_events=events,
                    output_artifact_paths=output_artifacts,
                )
                trajectory_record["evaluation"] = eval_result
                trajectory_record["bafo_success"] = eval_result.get("bafo_success")
                trajectory_record["all_challenges_resolved"] = eval_result.get("all_challenges_resolved", False)
                trajectory_record["trajectory_classification"] = eval_result.get("trajectory_classification")

                # Persist valid trajectory
                out_file = results_dir / f"{traj_id}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(trajectory_record, f, indent=2)

                logger.info(
                    "Finished %s | BAFO: %s (%s) | Wall: %.2fs (Pacing: %.2fs, Active: %.2fs)",
                    traj_id,
                    trajectory_record["bafo_success"],
                    trajectory_record["trajectory_classification"],
                    duration,
                    timing["rate_limit_wait_time"],
                    trajectory_record["orchestration_active_time"],
                )
                return trajectory_record

        except InfrastructureError as ie:
            duration = time.time() - start_time
            logger.warning("Infrastructure error in %s (attempt %d): %s", traj_id, infra_attempt, ie)
            
            # Save invalid infrastructure record
            invalid_record = {
                "trajectory_id": traj_id,
                "task_id": task_id,
                "condition": condition,
                "model": model,
                "status": "infrastructure_invalid",
                "bafo_success": False,
                "all_challenges_resolved": False,
                "trajectory_classification": "infrastructure_invalid",
                "error": str(ie),
                "duration_seconds": round(duration, 3),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            inv_file = results_dir / f"{traj_id}_attempt{infra_attempt}_infra_invalid.json"
            with open(inv_file, "w", encoding="utf-8") as f:
                json.dump(invalid_record, f, indent=2)

            if infra_attempt < max_infra_retries:
                backoff_time = (2 ** infra_attempt) * 5.0
                logger.info("Retrying slot %s in %.2fs (attempt %d/%d)...", traj_id, backoff_time, infra_attempt + 1, max_infra_retries)
                time.sleep(backoff_time)
            else:
                logger.error("Slot %s permanently invalid due to repeated infrastructure failures.", traj_id)
                return invalid_record

        except Exception as e:
            # Genuine software/orchestration error
            duration = time.time() - start_time
            timing = get_llm_timing_stats()
            logger.error("Orchestration error in %s: %s", traj_id, e)
            trajectory_record = {
                "trajectory_id": traj_id,
                "task_id": task_id,
                "condition": condition,
                "model": model,
                "run_index": item["run_index"],
                "suite": item["suite"],
                "user_question": question,
                "status": "failed",
                "error": str(e),
                "total_wall_time": round(duration, 3),
                "rate_limit_wait_time": timing["rate_limit_wait_time"],
                "provider_retry_wait_time": timing["provider_retry_wait_time"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            eval_result = oracle.evaluate_trajectory(task_id, trajectory_record)
            trajectory_record["evaluation"] = eval_result
            trajectory_record["bafo_success"] = False
            trajectory_record["all_challenges_resolved"] = False

            out_file = results_dir / f"{traj_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(trajectory_record, f, indent=2)
            return trajectory_record

    return {"status": "infrastructure_invalid", "bafo_success": False}


def main():
    parser = argparse.ArgumentParser(description="GAS Benchmark Execution Runner")
    parser.add_argument("--suite", choices=["isolated", "compound", "dev", "all"], default="dev", help="Suite to run")
    parser.add_argument("--conditions", default="C1,C2,C3,C4,C5", help="Comma-separated conditions")
    parser.add_argument("--runs", type=int, default=3, help="Number of repetitions per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for schedule randomization")
    parser.add_argument("--dry-run", action="store_true", help="Print schedule without running")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Output directory for trajectory logs")
    parser.add_argument(
        "--model",
        choices=["gpt-4o-mini"],
        default="gpt-4o-mini",
        help="Fixed LLM model for all benchmark conditions.",
    )
    parser.add_argument("--max-trajectories", type=int, default=None, help="Maximum number of new trajectories to execute")
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Optional exact task ID filter for a targeted development smoke test.",
    )

    args = parser.parse_args()
    suites = ["isolated", "compound"] if args.suite == "all" else [args.suite]
    conditions = [c.strip() for c in args.conditions.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(
        suites=suites,
        conditions=conditions,
        num_runs=args.runs,
        seed=args.seed,
        task_id=args.task_id,
    )
    if args.task_id:
        schedule = [item for item in schedule if item["task_id"] == args.task_id]
        if not schedule:
            parser.error(f"No scheduled trajectory matched task ID {args.task_id!r}.")

    print("=" * 60)
    print(f"GAS Orchestration Evaluation Runner")
    print(f"Suites: {suites} | Conditions: {conditions} | Runs: {args.runs} | Model: {args.model}")
    print(f"Total Scheduled Trajectories: {len(schedule)}")
    print(f"Results Output Directory: {out_dir}")
    if args.max_trajectories:
        print(f"Max New Trajectories Cap: {args.max_trajectories}")
    print("=" * 60)

    if args.dry_run:
        print("\nDry Run Schedule Preview (First 10 items):")
        for idx, item in enumerate(schedule[:10], 1):
            print(f"  {idx}. {item['trajectory_id']} ({item['suite']} | {item['condition']})")
        if len(schedule) > 10:
            print(f"  ... and {len(schedule) - 10} more trajectories.")
        return

    oracle = BenchmarkOracle()
    completed = []
    successes = 0
    newly_executed = 0

    for idx, item in enumerate(schedule, 1):
        out_file = out_dir / f"{item['trajectory_id']}.json"
        if out_file.exists():
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    res = json.load(f)
                if res.get("status") != "infrastructure_invalid":
                    logger.info("[%d/%d] Skipping already completed %s (BAFO=%s)", idx, len(schedule), item["trajectory_id"], res.get("bafo_success"))
                    completed.append(res)
                    if res.get("bafo_success"):
                        successes += 1
                    continue
            except Exception:
                pass

        if args.max_trajectories is not None and newly_executed >= args.max_trajectories:
            print(f"\n[INFO] Reached requested limit of {args.max_trajectories} new trajectories. Stopping run.")
            break

        print(f"\n[{idx}/{len(schedule)}] Executing {item['trajectory_id']} with {args.model}...")
        res = execute_trajectory(item, oracle, out_dir, model=args.model)
        completed.append(res)
        newly_executed += 1
        if res.get("bafo_success"):
            successes += 1

    valid_trajs = [c for c in completed if c.get("status") != "infrastructure_invalid"]
    print("\n" + "=" * 60)
    print("Benchmark Execution Batch Complete!")
    print(f"Total Trajectories Processed: {len(completed)}")
    print(f"Newly Executed in this batch: {newly_executed}")
    print(f"Total BAFO Successes: {successes}/{len(valid_trajs)} ({successes/max(1, len(valid_trajs))*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
