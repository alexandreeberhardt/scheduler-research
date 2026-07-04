#!/usr/bin/env python3
"""
Compare CFS / heuristic / model on real scheduling metrics.

Usage:
    python compare_policies.py \
        --taskfiles emulator/tasks/demo.tasks emulator/tasks/generated/workload_000.tasks \
        --model-run-dir model/artifacts/training_runs/candidate_large_heuristic \
        --seeds 0 1 2 3 4 \
        --duration 260
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskfiles", nargs="+", required=True)
    parser.add_argument("--model-run-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=int, default=260)
    parser.add_argument("--output", default=None, help="JSON output path")
    return parser.parse_args()


METRICS = [
    "avg_turnaround_ms",
    "avg_wait_ms",
    "avg_response_ms",
    "ctx_switches",
    "fairness_index",
    "starvation_count",
]

# Lower is better for all except fairness_index (higher = fairer)
LOWER_IS_BETTER = {
    "avg_turnaround_ms": True,
    "avg_wait_ms": True,
    "avg_response_ms": True,
    "ctx_switches": True,
    "fairness_index": False,
    "starvation_count": True,
}


def delta_pct(policy_val: float, cfs_val: float, lower_is_better: bool) -> float:
    """Positive = improvement vs CFS, negative = worse."""
    if cfs_val == 0:
        return 0.0
    raw = (policy_val - cfs_val) / cfs_val * 100
    return -raw if lower_is_better else raw


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    emulator_dir = repo_root / "emulator"
    if str(emulator_dir) not in sys.path:
        sys.path.insert(0, str(emulator_dir))

    from constants import NS_PER_MS
    from heuristic_policy import MetricHeuristicPolicy
    from model_policy import SchedulerModelPolicy
    from sched_em import run_one
    from task_parser import POLICY_MAP, parse_taskfile

    print("Loading model...", flush=True)
    model_policy = SchedulerModelPolicy(args.model_run_dir)
    heuristic_policy = MetricHeuristicPolicy()

    cfs_policy = POLICY_MAP["CFS"]
    duration_ns = args.duration * NS_PER_MS

    # Results: policy_name -> list of per-run metric dicts
    results: dict[str, list[dict]] = {"cfs": [], "heuristic": [], "model": []}

    taskfiles = [Path(tf) for tf in args.taskfiles]
    total_runs = len(taskfiles) * len(args.seeds)
    run_number = 0

    for taskfile in taskfiles:
        tasks = parse_taskfile(str(taskfile))
        for seed in args.seeds:
            run_number += 1
            print(f"[{run_number}/{total_runs}] {taskfile.name}  seed={seed}", flush=True)

            stats_cfs, _ = run_one(
                tasks, cfs_policy, duration_ns, False, seed=seed, decision_mode="baseline"
            )
            stats_heuristic, _ = run_one(
                tasks,
                cfs_policy,
                duration_ns,
                False,
                seed=seed,
                model_policy=heuristic_policy,
                decision_mode="closed_loop",
                decision_source_name="heuristic",
            )
            stats_model, _ = run_one(
                tasks,
                cfs_policy,
                duration_ns,
                False,
                seed=seed,
                model_policy=model_policy,
                decision_mode="closed_loop",
                decision_source_name="model",
            )

            results["cfs"].append(asdict(stats_cfs))
            results["heuristic"].append(asdict(stats_heuristic))
            results["model"].append(asdict(stats_model))

    # Aggregate
    print("\n" + "=" * 64)
    print(f"{'Metric':<26} {'CFS':>10} {'Heuristic':>12} {'Model':>10}")
    print(f"{'':26} {'':>10} {'vs CFS':>12} {'vs CFS':>10}")
    print("-" * 64)

    summary = {}
    for metric in METRICS:
        cfs_values = [r[metric] for r in results["cfs"]]
        heuristic_values = [r[metric] for r in results["heuristic"]]
        model_values = [r[metric] for r in results["model"]]

        cfs_mean = mean(cfs_values)
        heuristic_mean = mean(heuristic_values)
        model_mean = mean(model_values)

        is_lower_better = LOWER_IS_BETTER[metric]
        heuristic_delta = delta_pct(heuristic_mean, cfs_mean, is_lower_better)
        model_delta = delta_pct(model_mean, cfs_mean, is_lower_better)

        heuristic_str = f"{heuristic_delta:+.1f}%"
        model_str = f"{model_delta:+.1f}%"

        print(f"{metric:<26} {cfs_mean:>10.3f} {heuristic_str:>12} {model_str:>10}")
        summary[metric] = {
            "cfs": cfs_mean,
            "heuristic": heuristic_mean,
            "heuristic_delta_pct": heuristic_delta,
            "model": model_mean,
            "model_delta_pct": model_delta,
        }

    print("=" * 64)
    print(f"  Runs : {len(results['cfs'])}  ({len(taskfiles)} taskfiles x {len(args.seeds)} seeds)", flush=True)
    print()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump({"summary": summary, "runs": results}, fh, indent=2)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
