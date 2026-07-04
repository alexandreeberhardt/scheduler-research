#!/usr/bin/env python3

import argparse
from pathlib import Path
from types import SimpleNamespace

import wandb

from common import train


sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [128, 256, 512]},
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 3e-3,
        },
        "task_hidden_dim": {"values": [96, 128, 192]},
        "history_hidden_dim": {"values": [96, 128, 192]},
        "num_layers": {"values": [1, 2]},
        "dropout": {"values": [0.0, 0.1, 0.2]},
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
    },
}


def sweep_train(args: argparse.Namespace) -> None:
    run = wandb.init(project=args.wandb_project)
    config = run.config
    config.update(build_sweep_runtime_config(args, run.id), allow_val_change=True)

    train_args = SimpleNamespace(**dict(config))
    output_dir = Path(train_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train(train_args, Path(train_args.dataset_dir), output_dir, wandb_run=run)


def build_sweep_runtime_config(args: argparse.Namespace, run_id: str) -> dict:
    return {
        "dataset_dir": args.dataset_dir,
        "output_dir": f"artifacts/training_runs/sweep_{run_id}",
        "seed": args.seed,
        "init_checkpoint": args.init_checkpoint,
        "max_history": args.max_history,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "wandb_mode": "online",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance un sweep WandB sur le modèle candidat."
    )
    parser.add_argument("--dataset-dir", default="artifacts/candidate_dataset")
    parser.add_argument("--wandb-project", default="GLO-7030-projet")
    parser.add_argument("--sweep-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--max-history", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def run_sweep_agent(args: argparse.Namespace, sweep_id: str) -> None:
    def train_once() -> None:
        sweep_train(args)

    wandb.agent(sweep_id, function=train_once, count=args.sweep_count)


if __name__ == "__main__":
    args = parse_args()
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    run_sweep_agent(args, sweep_id)
