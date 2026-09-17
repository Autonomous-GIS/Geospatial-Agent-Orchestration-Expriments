
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results" / "runs"


def load_all_trajectory_results(results_dir: Path) -> pd.DataFrame:
    """Load all trajectory JSON logs from the results directory into a DataFrame."""
    records = []
    for f in results_dir.glob("*.json"):
        if not f.name.startswith("traj_"):
            continue
        if "infra_invalid" in f.name:
            continue
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                status = data.get("status", "")
                if status == "infrastructure_invalid":
                    continue

                bafo_raw = data.get("bafo_success")
                # 3-valued: True (1.0), False (0.0), None (np.nan)
                if bafo_raw is True:
                    bafo_val = 1.0
                elif bafo_raw is False:
                    bafo_val = 0.0
                else:
                    bafo_val = np.nan

                eval_data = data.get("evaluation", {})
                classification = data.get("trajectory_classification") or eval_data.get("trajectory_classification", "unknown")

                records.append({
                    "trajectory_id": data.get("trajectory_id"),
                    "task_id": data.get("task_id"),
                    "suite": data.get("suite"),
                    "condition": data.get("condition"),
                    "run_index": data.get("run_index"),
                    "status": status,
                    "classification": classification,
                    "bafo_success": bafo_val,
                    "is_evaluable": int(not np.isnan(bafo_val)),
                    "total_wall_time": data.get("total_wall_time", data.get("duration_seconds", 0.0)),
                    "rate_limit_wait_time": data.get("rate_limit_wait_time", 0.0),
                    "provider_retry_wait_time": data.get("provider_retry_wait_time", 0.0),
                    "orchestration_active_time": data.get("orchestration_active_time", 0.0),
                    "worker_execution_time": data.get("worker_execution_time", 0.0),
                    "total_tokens": data.get("total_tokens", 0),
                    "input_tokens": data.get("input_tokens", 0),
                    "output_tokens": data.get("output_tokens", 0),
                })
        except Exception as e:
            logger.warning("Could not read result file %s: %s", f.name, e)

    df = pd.DataFrame(records)
    return df


def paired_bootstrap_ci(
    task_scores_a: Dict[str, float],
    task_scores_b: Dict[str, float],
    rng: np.random.Generator,
    n_bootstraps: int = 50000,
) -> Tuple[float, float, float]:
    """Compute paired mean difference and 95% CI via task-clustered bootstrap.

    Returns: (mean_diff, ci_lower, ci_upper)
    """
    task_ids = sorted(list(set(task_scores_a.keys()) & set(task_scores_b.keys())))
    n_tasks = len(task_ids)
    if n_tasks == 0:
        return 0.0, 0.0, 0.0

    diffs = np.array([task_scores_a[t] - task_scores_b[t] for t in task_ids])
    observed_mean_diff = float(np.mean(diffs))

    sample_indices = rng.choice(n_tasks, size=(n_bootstraps, n_tasks), replace=True)
    boot_diffs = np.mean(diffs[sample_indices], axis=1)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return observed_mean_diff, ci_lower, ci_upper


def paired_sign_flip_test(
    task_scores_a: Dict[str, float],
    task_scores_b: Dict[str, float],
    rng: np.random.Generator,
    n_permutations: int = 100000,
) -> float:
    """Compute exact or Monte Carlo two-sided paired sign-flip randomization p-value for H0: E[d] = 0."""
    task_ids = sorted(list(set(task_scores_a.keys()) & set(task_scores_b.keys())))
    n_tasks = len(task_ids)
    if n_tasks == 0:
        return 1.0

    diffs = np.array([task_scores_a[t] - task_scores_b[t] for t in task_ids])
    observed_t = float(np.abs(np.mean(diffs)))

    if observed_t == 0.0:
        return 1.0

    # For small n (e.g. CCS with n=10 tasks), compute exact 2^10 = 1024 enumeration
    if n_tasks <= 12:
        num_configs = 1 << n_tasks
        perm_means = []
        for mask in range(num_configs):
            signs = np.array([1.0 if (mask & (1 << i)) else -1.0 for i in range(n_tasks)])
            perm_means.append(np.abs(np.mean(diffs * signs)))
        p_val = float(np.mean(np.array(perm_means) >= observed_t))
        return p_val

    # Monte Carlo sign-flip for larger n
    random_signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_tasks))
    perm_means = np.abs(np.mean(diffs * random_signs, axis=1))
    p_val = float(np.mean(perm_means >= observed_t))
    return p_val


def compute_resource_table(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute detailed summary statistics (Mean, SD, Median, IQR) for resource metrics."""
    res = {}
    for cond, group in df.groupby("condition"):
        res[cond] = {
            "trajectory_count": int(len(group)),
            "total_wall_time": {
                "mean": round(float(group["total_wall_time"].mean()), 2),
                "sd": round(float(group["total_wall_time"].std()), 2),
                "median": round(float(group["total_wall_time"].median()), 2),
                "iqr_25": round(float(group["total_wall_time"].quantile(0.25)), 2),
                "iqr_75": round(float(group["total_wall_time"].quantile(0.75)), 2),
            },
            "rate_limit_wait_time": {
                "mean": round(float(group["rate_limit_wait_time"].mean()), 2),
                "median": round(float(group["rate_limit_wait_time"].median()), 2),
            },
            "provider_retry_wait_time": {
                "mean": round(float(group["provider_retry_wait_time"].mean()), 2),
                "median": round(float(group["provider_retry_wait_time"].median()), 2),
            },
            "orchestration_active_time": {
                "mean": round(float(group["orchestration_active_time"].mean()), 2),
                "median": round(float(group["orchestration_active_time"].median()), 2),
            },
            "worker_execution_time": {
                "mean": round(float(group["worker_execution_time"].mean()), 2),
                "median": round(float(group["worker_execution_time"].median()), 2),
            },
            "total_tokens": {
                "mean": round(float(group["total_tokens"].mean()), 1),
                "median": round(float(group["total_tokens"].median()), 1),
            },
        }
    return res


def analyze_results(df: pd.DataFrame, seed: int = 20260815) -> Dict[str, Any]:
    """Perform complete confirmatory statistical analysis."""
    rng = np.random.default_rng(seed)
    report: Dict[str, Any] = {"metadata": {"seed": seed, "total_trajectories": len(df)}}

    # Separate ISS and CCS
    iss_df = df[df["suite"] == "isolated"]
    ccs_df = df[df["suite"] == "compound"]

    # 1. Compute ISS task-level mean success rates
    iss_eval = iss_df.dropna(subset=["bafo_success"])
    task_cond_means = iss_eval.groupby(["task_id", "condition"])["bafo_success"].mean().to_dict()

    cond_task_scores = defaultdict(dict)
    for (t_id, cond), score in task_cond_means.items():
        cond_task_scores[cond][t_id] = score

    cond_means = {cond: float(np.mean(list(scores.values()))) for cond, scores in cond_task_scores.items()}
    report["condition_means_iss"] = cond_means

    # 2. RQ2 Primary Matched & Ecological Contrasts
    if "C5" in cond_task_scores and "C1" in cond_task_scores:
        m_diff, ci_l, ci_u = paired_bootstrap_ci(cond_task_scores["C5"], cond_task_scores["C1"], rng=rng)
        p_val = paired_sign_flip_test(cond_task_scores["C5"], cond_task_scores["C1"], rng=rng)
        report["rq2_c5_vs_c1"] = {"mean_diff": m_diff, "ci_95": [ci_l, ci_u], "p_value": p_val}

    if "C5" in cond_task_scores and "C0" in cond_task_scores:
        m_diff, ci_l, ci_u = paired_bootstrap_ci(cond_task_scores["C5"], cond_task_scores["C0"], rng=rng)
        p_val = paired_sign_flip_test(cond_task_scores["C5"], cond_task_scores["C0"], rng=rng)
        report["rq2_c5_vs_c0"] = {"mean_diff": m_diff, "ci_95": [ci_l, ci_u], "p_value": p_val}

    # 3. RQ3 Component Ablation Contrasts
    for ablated in ["C2", "C3", "C4"]:
        if "C5" in cond_task_scores and ablated in cond_task_scores:
            m_diff, ci_l, ci_u = paired_bootstrap_ci(cond_task_scores["C5"], cond_task_scores[ablated], rng=rng)
            p_val = paired_sign_flip_test(cond_task_scores["C5"], cond_task_scores[ablated], rng=rng)
            report[f"rq3_c5_vs_{ablated.lower()}"] = {"mean_diff": m_diff, "ci_95": [ci_l, ci_u], "p_value": p_val}

    # 4. RQ4 Compound Evaluation
    if not ccs_df.empty:
        ccs_eval = ccs_df.dropna(subset=["bafo_success"])
        ccs_task_cond = ccs_eval.groupby(["task_id", "condition"])["bafo_success"].mean().to_dict()
        ccs_scores = defaultdict(dict)
        for (t_id, cond), score in ccs_task_cond.items():
            ccs_scores[cond][t_id] = score

        report["condition_means_ccs"] = {cond: float(np.mean(list(scores.values()))) for cond, scores in ccs_scores.items()}
        if "C2" in ccs_scores and "C1" in ccs_scores:
            m_diff, ci_l, ci_u = paired_bootstrap_ci(ccs_scores["C2"], ccs_scores["C1"], rng=rng)
            p_val = paired_sign_flip_test(ccs_scores["C2"], ccs_scores["C1"], rng=rng)
            report["rq4_c2_vs_c1"] = {"mean_diff": m_diff, "ci_95": [ci_l, ci_u], "p_value": p_val}

    # 5. Suite-Separated Resource Tables
    report["resource_accounting_iss"] = compute_resource_table(iss_df)
    if not ccs_df.empty:
        report["resource_accounting_ccs"] = compute_resource_table(ccs_df)

    return report


def main():
    parser = argparse.ArgumentParser(description="GAS Benchmark Statistical Analysis Engine")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Directory containing run JSONs")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(BENCHMARK_DIR / "results" / "trajectory_level_analysis_report.json"),
    )
    parser.add_argument("--seed", type=int, default=20260815)

    args = parser.parse_args()
    r_dir = Path(args.results_dir)

    if not r_dir.exists():
        print(f"Results directory {r_dir} does not exist.")
        return

    df = load_all_trajectory_results(r_dir)
    print(f"Loaded {len(df)} trajectory records ({df['is_evaluable'].sum()} evaluable).")

    if df.empty:
        print("No records found.")
        return

    report = analyze_results(df, seed=args.seed)
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    task_level_file = out_file.parent / "task_level_results.json"
    records = []
    for _, row in df.sort_values(by=["task_id", "condition", "run_index"]).iterrows():
        records.append({
            "trajectory_id": row["trajectory_id"],
            "task_id": row["task_id"],
            "suite": row["suite"],
            "condition": row["condition"],
            "run_index": int(row["run_index"]),
            "status": row["status"],
            "bafo_success": bool(row["bafo_success"]) if not pd.isna(row["bafo_success"]) else None,
            "classification": row["classification"],
            "total_wall_time": float(row["total_wall_time"]),
            "total_tokens": int(row["total_tokens"]),
        })
    with open(task_level_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print(f"\nSaved analysis report to: {out_file}")
    print(f"Saved task-level results to: {task_level_file}")


if __name__ == "__main__":
    main()
