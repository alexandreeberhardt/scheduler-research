#!/usr/bin/env python3

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construit un dataset de décisions de scheduling à partir de l'émulateur."
    )
    parser.add_argument("--taskfiles", nargs="+", required=True, help="Liste des fichiers .tasks à rejouer")
    parser.add_argument("--output-dir", default="artifacts/candidate_dataset", help="Dossier de sortie du dataset")
    parser.add_argument("--policies", nargs="+", default=["CFS"], help="Politiques utilisées pour générer les états")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Seeds rejouées pour chaque workload")
    parser.add_argument("--duration", type=int, default=400, help="Durée de simulation en ms")
    parser.add_argument(
        "--teacher",
        choices=["cfs", "oracle", "heuristic"],
        default="oracle",
        help="Cible supervisée à apprendre",
    )
    parser.add_argument("--max-history", type=int, default=256, help="Longueur maximale d'historique à exposer au modèle")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    emulator_dir = repo_root / "emulator"
    if str(emulator_dir) not in sys.path:
        sys.path.insert(0, str(emulator_dir))

    from constants import NS_PER_MS
    from decision_features import global_feature_names, task_feature_names
    from sched_em import policy_name, run_one
    from task_parser import POLICY_MAP, parse_taskfile

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_ns = args.duration * NS_PER_MS
    episodes = []

    for taskfile_arg in args.taskfiles:
        taskfile = Path(taskfile_arg)
        tasks = parse_taskfile(str(taskfile))
        for policy_text in args.policies:
            policy = POLICY_MAP[policy_text.upper()]
            policy_label = policy_name(policy)
            for seed in args.seeds:
                print(
                    f"Replay {taskfile}  policy={policy_label}  seed={seed}  teacher={args.teacher}",
                    flush=True,
                )
                stats, sched = run_one(
                    tasks,
                    policy,
                    duration_ns,
                    verbose=False,
                    seed=seed,
                    record_decisions=True,
                    decision_teacher=args.teacher,
                    max_history=args.max_history,
                )
                episodes.append(
                    {
                        "taskfile": str(taskfile),
                        "policy": policy_label,
                        "seed": seed,
                        "duration_ms": args.duration,
                        "stats": asdict(stats),
                        "samples": sched.decision_samples,
                    }
                )

    indices = list(range(len(episodes)))
    random.Random(args.split_seed).shuffle(indices)
    splits = split_episode_indices(indices, args.train_ratio, args.val_ratio)

    payload = {
        "teacher": args.teacher,
        "max_history": args.max_history,
        "task_feature_names": task_feature_names(),
        "global_feature_names": global_feature_names(),
        "episodes": episodes,
        "splits": splits,
    }
    with (output_dir / "decision_dataset.json").open("w") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)

    summary = {
        "teacher": args.teacher,
        "episodes": len(episodes),
        "samples": sum(len(episode["samples"]) for episode in episodes),
        "split_episode_counts": {name: len(values) for name, values in splits.items()},
    }
    with (output_dir / "metadata.json").open("w") as file_handle:
        json.dump(summary, file_handle, indent=2, ensure_ascii=False)

    print(
        f"Dataset construit dans {output_dir} : "
        f"{summary['episodes']} épisodes, {summary['samples']} décisions.",
        flush=True,
    )


def split_episode_indices(
    indices: list[int],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[int]]:
    n = len(indices)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": list(indices), "val": [], "test": []}
    if n == 2:
        return {"train": [indices[0]], "val": [], "test": [indices[1]]}

    train_cut = max(1, int(n * train_ratio))
    val_cut = int(n * (train_ratio + val_ratio))
    val_cut = max(train_cut, val_cut)
    val_cut = min(val_cut, n - 1)
    if train_cut >= n:
        train_cut = n - 1

    return {
        "train": indices[:train_cut],
        "val": indices[train_cut:val_cut],
        "test": indices[val_cut:],
    }


if __name__ == "__main__":
    main()
