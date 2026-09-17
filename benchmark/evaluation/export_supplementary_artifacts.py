"""Export manuscript-supporting artifacts from frozen benchmark trajectories.

This exporter is intentionally read-only with respect to trajectory logs.  It
creates: (1) Supplementary Table S2, the runtime and token-use summary; and (2) a C5 failure
record whose provenance points to the exact C5 trajectory files used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results" / "experiment_luna_20260821_131749"
DEFAULT_MANIFEST = PROJECT_ROOT / "benchmark" / "manifests" / "isolated_suite.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    """Return the linearly interpolated percentile used by NumPy."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_trajectories(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for path in sorted(results_dir.glob("traj_*.json")):
        with path.open(encoding="utf-8") as handle:
            records.append((path, json.load(handle)))
    if not records:
        raise ValueError(f"No trajectory logs found in {results_dir}")
    return records


def write_table_s2(records: list[tuple[Path, dict[str, Any]]], output: Path) -> None:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, record in records:
        by_condition[record["condition"]].append(record)

    lines = [
        "# Supplementary Table S2 Runtime and Token Use by Experimental Condition",
        "",
        "Summary of all 540 isolated-suite trajectories. Wall time is recorded per completed trajectory; token counts are the model-provider counts recorded in each trajectory log.",
        "",
        "| Condition | Trajectories | Total wall time (h) | Mean wall time (s) | Median wall time (s) | 95th percentile wall time (s) | Total tokens | Mean tokens per trajectory | Input tokens | Output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in sorted(by_condition):
        group = by_condition[condition]
        wall_times = [float(record.get("total_wall_time") or 0.0) for record in group]
        tokens = [int(record.get("total_tokens") or 0) for record in group]
        input_tokens = [int(record.get("input_tokens") or 0) for record in group]
        output_tokens = [int(record.get("output_tokens") or 0) for record in group]
        lines.append(
            "| {condition} | {count} | {hours:.2f} | {mean_wall:.1f} | {median_wall:.1f} | "
            "{p95:.1f} | {total_tokens:,} | {mean_tokens:,.0f} | {input_tokens:,} | {output_tokens:,} |".format(
                condition=condition,
                count=len(group),
                hours=sum(wall_times) / 3600,
                mean_wall=mean(wall_times),
                median_wall=median(wall_times),
                p95=percentile(wall_times, 0.95),
                total_tokens=sum(tokens),
                mean_tokens=mean(tokens),
                input_tokens=sum(input_tokens),
                output_tokens=sum(output_tokens),
            )
        )

    all_records = [record for _, record in records]
    lines.extend(
        [
            "",
            "**All conditions:** 540 trajectories; {:.2f} total wall-clock hours; {:,} total tokens ({:,} input; {:,} output).".format(
                sum(float(record.get("total_wall_time") or 0.0) for record in all_records) / 3600,
                sum(int(record.get("total_tokens") or 0) for record in all_records),
                sum(int(record.get("input_tokens") or 0) for record in all_records),
                sum(int(record.get("output_tokens") or 0) for record in all_records),
            ),
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def manifest_families(manifest_path: Path) -> dict[str, dict[str, str]]:
    with manifest_path.open(encoding="utf-8") as handle:
        tasks = json.load(handle)
    return {
        task["task_id"]: {
            "challenge_family_id": task.get("family_id", ""),
            "challenge_family": task.get("family_name", ""),
        }
        for task in tasks
        if "task_id" in task
    }


def normalise_ledger_events(events: list[dict[str, Any]], trajectory_id: str) -> tuple[list[dict[str, Any]], int]:
    """Correct only an internally inconsistent display identifier.

    The original identifier is retained in ``source_trajectory_id`` so the
    generated record remains auditable without changing the raw trajectory.
    """
    normalized = []
    changed = 0
    for event in events:
        event_copy = copy.deepcopy(event)
        original_id = event_copy.get("trajectory_id")
        if original_id and original_id != trajectory_id:
            event_copy["source_trajectory_id"] = original_id
            event_copy["trajectory_id"] = trajectory_id
            changed += 1
        normalized.append(event_copy)
    return normalized, changed


def compact_failure_record(path: Path, record: dict[str, Any], family: dict[str, str]) -> tuple[dict[str, Any], int]:
    state = record.get("workflow_state", {})
    evaluation = record.get("evaluation", {})
    ledger_events, corrected_event_ids = normalise_ledger_events(record.get("ledger_events", []), record["trajectory_id"])
    task_states = list(state.get("tasks", {}).values())

    failure = {
        "task_id": record["task_id"],
        **family,
        "condition": record["condition"],
        "run_number": record["run_index"],
        "trajectory_id": record["trajectory_id"],
        "user_question": record.get("user_question"),
        "oracle": {
            "bafo_success": record.get("bafo_success"),
            "trajectory_classification": record.get("trajectory_classification"),
            "failed_benchmark_criteria": evaluation.get("failed_constraints", []),
            "passed_benchmark_criteria": evaluation.get("passed_constraints", []),
            "unevaluable_benchmark_criteria": evaluation.get("unevaluable_constraints", []),
            "diagnostic_summary": evaluation.get("diagnostic_summary"),
            "terminal_state": evaluation.get("terminal_state"),
            "production_terminal_state": evaluation.get("production_terminal_state"),
        },
        "observed_runtime": {
            "execution_steps": record.get("execution_steps", []),
            "ledger_events": ledger_events,
            "workflow_violations": list(state.get("violations", {}).values()),
            "worker_error": record.get("error"),
        },
        "recovery_and_control": {
            "qc_evaluations": record.get("qc_evaluations", []),
            "qc_blocks": record.get("qc_blocks", []),
            "qc_interventions": record.get("qc_interventions", []),
            "replanning_iterations": record.get("replanning_iterations"),
        },
        "after_action_and_final_disposition": {
            "task_states": task_states,
            "terminal_reason": state.get("terminal_reason"),
            "status": record.get("status"),
        },
        "source": {
            "file": path.name,
            "sha256": sha256(path),
            "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "source_trajectory_id": record["trajectory_id"],
            "source_condition": record["condition"],
        },
    }
    return failure, corrected_event_ids


def write_c5_failure_record(
    records: list[tuple[Path, dict[str, Any]]], manifest_path: Path, output: Path
) -> None:
    families = manifest_families(manifest_path)
    failures = []
    corrected_event_ids = 0
    for path, record in records:
        if record.get("suite") != "isolated" or record.get("condition") != "C5" or record.get("bafo_success") is not False:
            continue
        failure, corrected = compact_failure_record(path, record, families.get(record["task_id"], {}))
        failures.append(failure)
        corrected_event_ids += corrected

    family_counts = Counter(item.get("challenge_family_id") for item in failures)
    output_data = {
        "metadata": {
            "description": "Failed C5 isolated trajectories (Full GIS-Aware Orchestrator) exported from raw C5 logs.",
            "source_directory": str(output.parent),
            "selection": "suite=isolated, condition=C5, bafo_success=false",
            "trajectory_count": len(failures),
            "model": "gpt-5.6-luna",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "family_failure_counts": dict(sorted(family_counts.items())),
            "provenance_note": (
                "Each source record names and hashes its actual C5 trajectory file. "
                "Embedded ledger-event trajectory identifiers that conflicted with the enclosing raw C5 log "
                "were normalized for display; their original values are retained as source_trajectory_id."
            ),
            "normalised_embedded_ledger_event_trajectory_ids": corrected_event_ids,
        },
        "trajectories": failures,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export supplementary benchmark artifacts from frozen trajectory logs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--table-s2-output", type=Path)
    parser.add_argument("--c5-failures-output", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    table_s2_output = args.table_s2_output or results_dir / "table_s2_runtime_and_token_use.md"
    c5_failures_output = args.c5_failures_output or results_dir / "c5_isolated_failed_trajectories.json"
    records = load_trajectories(results_dir)

    write_table_s2(records, table_s2_output)
    write_c5_failure_record(records, args.manifest.resolve(), c5_failures_output)
    print(f"Wrote {table_s2_output}")
    print(f"Wrote {c5_failures_output}")


if __name__ == "__main__":
    main()
