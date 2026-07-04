#!/usr/bin/env python3
"""
Évalue le modèle de base (baseline) sur le jeu de test.
"""

import argparse
import torch
from common import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évalue le modèle candidat baseline sur le split de test."
    )
    parser.add_argument("--dataset-dir", default="artifacts/candidate_dataset")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/training_runs/candidate_baseline/best_model.pt",
    )
    parser.add_argument(
        "--artifacts-json",
        default="artifacts/training_runs/candidate_baseline/dataset_artifacts.json",
    )
    parser.add_argument(
        "--run-config",
        default="artifacts/training_runs/candidate_baseline/run_config.json",
    )
    parser.add_argument("--output-dir", default="artifacts/evaluation/candidate_baseline")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=256)
    
    # Paramètres requis par compute_loss dans common.py
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.2)
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    run_evaluation(args, device)


if __name__ == "__main__":
    main()
