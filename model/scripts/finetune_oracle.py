#!/usr/bin/env python3

import argparse
from pathlib import Path

from common import train
from train_baseline import init_wandb_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune le modèle CFS sur le dataset oracle."
    )
    parser.add_argument("--dataset-dir", default="artifacts/candidate_dataset_oracle")
    parser.add_argument(
        "--output-dir",
        default="artifacts/training_runs/candidate_oracle_finetune",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="artifacts/training_runs/candidate_baseline/best_model.pt",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-history", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--task-hidden-dim", type=int, default=128)
    parser.add_argument("--history-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--wandb-project", default="GLO-7030-projet")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--wandb-dir", default="artifacts/wandb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb_run(args)
    try:
        train(args, dataset_dir, output_dir, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
