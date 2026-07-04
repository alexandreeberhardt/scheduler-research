#!/usr/bin/env python3
"""
Evaluation d'un modele student entraine par distillation.

Ce script est compatible avec le pipeline actuel:
    - charge les sequences depuis artifacts/parsed_sched_switch
    - reconstruit le split train/val/test comme evaluate_baseline.py
    - recharge le meilleur checkpoint student produit par train_distillation.py
    - calcule les metriques globales et par classe
    - genere le rapport de classification et la matrice de confusion

Usage:
    python scripts/evaluate_distilled.py \
        --parsed-dir artifacts/parsed_sched_switch \
        --checkpoint artifacts/training_runs/lstm_distilled/best_student_model.pt \
        --artifacts-json artifacts/training_runs/lstm_distilled/dataset_artifacts.json \
        --run-config artifacts/training_runs/lstm_distilled/run_config.json \
        --output-dir artifacts/evaluation/lstm_distilled \
        --batch-size 1024 \
        --num-workers 4 \
        --top-n-confusion 15
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).parent))
from dataset import SchedSwitchDataset, split_sequences, load_sequences
from model import LSTMScheduler
from metrics import collect_predictions, classification_report, plot_confusion_matrix


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main(args: argparse.Namespace) -> None:
    parsed_dir = Path(args.parsed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ------------------------------------------------------------
    # Chargement des artefacts
    # ------------------------------------------------------------
    artifacts: dict[str, Any] = load_json(Path(args.artifacts_json))
    run_config: dict[str, Any] = load_json(Path(args.run_config))

    id_to_token: dict[int, str] = {
        int(k): v for k, v in artifacts["id_to_token"].items()
    }
    pad_id: int = int(artifacts["pad_id"])
    window_size: int = int(artifacts["window_size"])

    # ------------------------------------------------------------
    # Chargement des sequences et split temporel
    # ------------------------------------------------------------
    with open(parsed_dir / "metadata.json") as f:
        metadata: dict[str, Any] = json.load(f)

    sequences = load_sequences(parsed_dir, metadata["cpu_lengths"])

    _, _, test_seqs = split_sequences(
        sequences,
        train_ratio=artifacts["train_ratio"],
        val_ratio=artifacts["val_ratio"],
    )

    test_ds = SchedSwitchDataset(test_seqs, window_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Jeu de test : {len(test_ds):,} fenetres")

    # ------------------------------------------------------------
    # Chargement du checkpoint student
    # ------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "model_config" not in ckpt:
        raise KeyError(
            "Le checkpoint ne contient pas 'model_config'. "
            "Il doit etre sauvegarde par train_distillation.py."
        )

    model = LSTMScheduler(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"Checkpoint charge : epoque {ckpt['epoch']}  "
        f"val_loss={ckpt['val_loss']:.4f}  "
        f"val_top1={ckpt.get('val_top1_acc', 0.0):.4f}"
    )

    # ------------------------------------------------------------
    # Collecte des predictions
    # ------------------------------------------------------------
    print("Collecte des predictions sur le jeu de test ...")
    TOP_K = (1, 3, 5)
    targets, top1_preds, topk_correct = collect_predictions(
        model, test_loader, device, top_k=TOP_K
    )

    n = len(targets)
    if n == 0:
        raise ValueError("Le jeu de test est vide.")

    # ------------------------------------------------------------
    # Metriques globales
    # ------------------------------------------------------------
    eval_metrics: dict[str, Any] = {
        "n_samples": n,
        "top1_accuracy": sum(t == p for t, p in zip(targets, top1_preds)) / n,
    }
    for k, hits in topk_correct.items():
        eval_metrics[f"top{k}_accuracy"] = sum(hits) / n

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            total_count += len(y)

    eval_metrics["loss"] = total_loss / total_count

    print("\nMetriques globales")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} : {v:.4f}")
        else:
            print(f"  {k:<20} : {v}")

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------
    # Precision par classe
    # ------------------------------------------------------------
    class_correct: dict[int, int] = defaultdict(int)
    class_total: dict[int, int] = defaultdict(int)

    for t, p in zip(targets, top1_preds):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    per_class: dict[str, Any] = {}
    for cls, total in sorted(class_total.items(), key=lambda x: -x[1]):
        name = id_to_token.get(cls, f"<id:{cls}>")
        per_class[name] = {
            "accuracy": class_correct[cls] / total,
            "support": total,
            "id": cls,
        }

    with open(out_dir / "per_class_accuracy.json", "w") as f:
        json.dump(per_class, f, indent=2, ensure_ascii=False)

    print("\nTop-20 classes par support")
    print(f"  {'Classe':<35} {'Support':>8}  {'Precision':>10}")
    print("  " + "-" * 58)
    for name, info in list(per_class.items())[:20]:
        print(f"  {name:<35} {info['support']:>8}  {info['accuracy']:>10.4f}")

    # ------------------------------------------------------------
    # Rapport de classification
    # ------------------------------------------------------------
    report = classification_report(targets, top1_preds, id_to_token, top_n=50)
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nRapport de classification sauvegarde : {report_path}")

    # ------------------------------------------------------------
    # Matrice de confusion
    # ------------------------------------------------------------
    plot_confusion_matrix(
        targets,
        top1_preds,
        id_to_token,
        top_n=args.top_n_confusion,
        out_path=out_dir / f"confusion_matrix_top{args.top_n_confusion}.png",
    )

    # ------------------------------------------------------------
    # Petit resume du run de distillation
    # ------------------------------------------------------------
    distilled_summary = {
        "checkpoint": str(args.checkpoint),
        "best_epoch": ckpt["epoch"],
        "best_val_loss": ckpt["val_loss"],
        "best_val_top1_acc": ckpt.get("val_top1_acc", 0.0),
        "student_config": ckpt.get("model_config", {}),
        "teacher_checkpoint": run_config.get("teacher_checkpoint"),
        "temperature": run_config.get("temperature"),
        "alpha": run_config.get("alpha"),
    }
    with open(out_dir / "distillation_summary.json", "w") as f:
        json.dump(distilled_summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation terminee. Resultats dans : {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evalue le modele student distille sur le jeu de test."
    )
    ap.add_argument("--parsed-dir", default="artifacts/parsed_sched_switch")
    ap.add_argument(
        "--checkpoint",
        default="artifacts/training_runs/lstm_distilled/best_student_model.pt",
    )
    ap.add_argument(
        "--artifacts-json",
        default="artifacts/training_runs/lstm_distilled/dataset_artifacts.json",
    )
    ap.add_argument(
        "--run-config",
        default="artifacts/training_runs/lstm_distilled/run_config.json",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/evaluation/lstm_distilled",
    )
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--top-n-confusion", type=int, default=15)
    main(ap.parse_args())
