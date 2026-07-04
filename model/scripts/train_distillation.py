#!/usr/bin/env python3
"""
Entraînement avec distillation pour le LSTM de scheduling.

Pipeline:
    1) charge les séquences tokenisées depuis artifacts/parsed_sched_switch
    2) reconstruit le teacher depuis un checkpoint baseline
    3) entraîne un student plus petit avec une loss combinée:
           L = alpha * CE + (1 - alpha) * T^2 * KL(student || teacher)
    4) sauvegarde les artefacts au même format que train_baseline.py

Usage:
    uv run python scripts/train_distillation.py \
        --parsed-dir artifacts/parsed_sched_switch \
        --teacher-checkpoint artifacts/training_runs/lstm_baseline/best_lstm_model.pt \
        --output-dir artifacts/training_runs/lstm_distilled \
        --batch-size 1024 --epochs 10 --lr 1e-3 --window-size 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from common import evaluate_topk
from dataset import SchedSwitchDataset, split_sequences, load_sequences
from model import LSTMScheduler


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def infer_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_teacher_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    fallback_model_config: dict[str, Any],
) -> tuple[LSTMScheduler, dict[str, Any]]:
    """
    Recharge le teacher en reconstruisant l'architecture à partir du checkpoint.
    Compatible avec le format sauvegardé par BestModelCheckpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = ckpt.get("model_config", fallback_model_config)
    teacher = LSTMScheduler(**model_config).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return teacher, ckpt


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retourne:
        total_loss, ce_loss, kd_loss
    """
    ce_loss = F.cross_entropy(student_logits, targets, ignore_index=pad_id)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature**2)

    total_loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
    return total_loss, ce_loss, kd_loss


@torch.no_grad()
def evaluate_student(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float, float, float]:
    return evaluate_topk(model, loader, device, pad_id)


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    alpha: float,
    pad_id: int,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_items = 0
    total_correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            teacher_logits = teacher(x)

        student_logits = student(x)

        loss, ce_loss, kd_loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=y,
            temperature=temperature,
            alpha=alpha,
            pad_id=pad_id,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)

        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_ce += ce_loss.item() * bs
        total_kd += kd_loss.item() * bs
        total_items += bs
        total_correct += (student_logits.argmax(dim=1) == y).sum().item()

    return {
        "loss": total_loss / total_items,
        "ce_loss": total_ce / total_items,
        "kd_loss": total_kd / total_items,
        "accuracy": total_correct / total_items,
    }


@torch.no_grad()
def validate_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    alpha: float,
    pad_id: int,
) -> dict[str, float]:
    student.eval()
    teacher.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_items = 0
    total_correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        teacher_logits = teacher(x)
        student_logits = student(x)

        loss, ce_loss, kd_loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=y,
            temperature=temperature,
            alpha=alpha,
            pad_id=pad_id,
        )

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_ce += ce_loss.item() * bs
        total_kd += kd_loss.item() * bs
        total_items += bs
        total_correct += (student_logits.argmax(dim=1) == y).sum().item()

    return {
        "loss": total_loss / total_items,
        "ce_loss": total_ce / total_items,
        "kd_loss": total_kd / total_items,
        "accuracy": total_correct / total_items,
    }

def main(args: argparse.Namespace) -> None:
    parsed_dir = Path(args.parsed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = infer_device()
    print(f"Device : {device}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        dir=str(args.wandb_dir) if args.wandb_dir else None,
    )

    try:
        # ------------------------------------------------------------
        # Chargement des artefacts de parsing
        # ------------------------------------------------------------
        with open(parsed_dir / "vocab.json") as f:
            vocab: dict[str, int] = json.load(f)
        with open(parsed_dir / "metadata.json") as f:
            metadata: dict[str, Any] = json.load(f)

        vocab_size = len(vocab)
        pad_id = vocab["<PAD>"]

        print(f"Vocabulaire : {vocab_size} tokens")
        sequences = load_sequences(parsed_dir, metadata["cpu_lengths"])
        for cpu, seq in zip(sorted(int(k) for k in metadata["cpu_lengths"]), sequences):
            print(f"  CPU {cpu} : {len(seq):,} tokens")

        # ------------------------------------------------------------
        # Split / datasets / loaders
        # ------------------------------------------------------------
        train_seqs, val_seqs, test_seqs = split_sequences(sequences)
        train_ds = SchedSwitchDataset(train_seqs, args.window_size)
        val_ds = SchedSwitchDataset(val_seqs, args.window_size)
        test_ds = SchedSwitchDataset(test_seqs, args.window_size)

        print(
            f"\nDataset : train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}"
        )

        loader_kw = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

        # ------------------------------------------------------------
        # Teacher
        # ------------------------------------------------------------
        teacher_fallback_config = dict(
            vocab_size=vocab_size,
            embed_dim=args.teacher_embed_dim,
            hidden_size=args.teacher_hidden_size,
            num_layers=args.teacher_num_layers,
            dropout=args.teacher_dropout,
            pad_idx=pad_id,
        )

        teacher, teacher_ckpt = load_teacher_checkpoint(
            checkpoint_path=Path(args.teacher_checkpoint),
            device=device,
            fallback_model_config=teacher_fallback_config,
        )

        # ------------------------------------------------------------
        # Student
        # ------------------------------------------------------------
        student_config = dict(
            vocab_size=vocab_size,
            embed_dim=args.student_embed_dim,
            hidden_size=args.student_hidden_size,
            num_layers=args.student_num_layers,
            dropout=args.student_dropout,
            pad_idx=pad_id,
        )
        student = LSTMScheduler(**student_config).to(device)

        n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"Student : {n_params:,} parametres entrainables")

        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        # ------------------------------------------------------------
        # Sauvegarde configuration / artefacts
        # ------------------------------------------------------------
        run_config = {
            **vars(args),
            "device": str(device),
            "vocab_size": vocab_size,
            "pad_id": pad_id,
            "student_n_params": n_params,
            "teacher_best_epoch": teacher_ckpt.get("epoch"),
            "teacher_best_val_loss": teacher_ckpt.get("val_loss"),
        }
        save_json(out_dir / "run_config.json", run_config)

        id_to_token = {v: k for k, v in vocab.items()}
        dataset_artifacts = {
            "vocab": vocab,
            "id_to_token": {str(k): v for k, v in id_to_token.items()},
            "pad_id": pad_id,
            "vocab_size": vocab_size,
            "window_size": args.window_size,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
        }
        save_json(out_dir / "dataset_artifacts.json", dataset_artifacts)

        # ------------------------------------------------------------
        # Entraînement
        # ------------------------------------------------------------
        best_ckpt_path = out_dir / "best_student_model.pt"
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                    student=student,
                    teacher=teacher,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    temperature=args.temperature,
                    alpha=args.alpha,
                    pad_id=pad_id,
                    max_grad_norm=args.max_grad_norm,
                )

            val_metrics = validate_one_epoch(
                    student=student,
                    teacher=teacher,
                    loader=val_loader,
                    device=device,
                    temperature=args.temperature,
                    alpha=args.alpha,
                    pad_id=pad_id,
                )

            scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ce_loss": train_metrics["ce_loss"],
                "train_kd_loss": train_metrics["kd_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_ce_loss": val_metrics["ce_loss"],
                "val_kd_loss": val_metrics["kd_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "lr": current_lr,
            }
            history.append(record)
            save_json(out_dir / "training_history.json", history)

            improved = val_metrics["loss"] < (best_val_loss - args.early_stopping_min_delta)
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_top1_acc": val_metrics["accuracy"],
                        "model_config": student_config,
                        "run_config": run_config,
                    },
                    best_ckpt_path,
                )
                print(f"  Meilleur modele sauvegarde (val_loss={val_metrics['loss']:.4f})")
            else:
                patience_counter += 1

            if args.save_every_epoch:
                epoch_ckpt_path = out_dir / f"student_epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_top1_acc": val_metrics["accuracy"],
                        "model_config": student_config,
                        "run_config": run_config,
                    },
                    epoch_ckpt_path,
                )

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"lr={current_lr:.2e}"
            )

            wandb.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/ce_loss": train_metrics["ce_loss"],
                    "train/kd_loss": train_metrics["kd_loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/ce_loss": val_metrics["ce_loss"],
                    "val/kd_loss": val_metrics["kd_loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "lr": current_lr,
                    "epoch": epoch,
                }
            )

            if patience_counter >= args.early_stopping_patience:
                print(
                    f"Early stopping: aucune amelioration depuis {args.early_stopping_patience} epoques."
                )
                break

        # ------------------------------------------------------------
        # Evaluation finale sur test avec le meilleur checkpoint
        # ------------------------------------------------------------
        print(f"\n{'='*60}")
        print("Evaluation finale sur le jeu de TEST (meilleur checkpoint)")
        print(f"{'='*60}")

        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])
        student.eval()

        test_loss, test_top1, test_top3, test_top5 = evaluate_student(
            student, test_loader, device, pad_id
        )

        print(
            f"  test  loss={test_loss:.4f}  "
            f"top1={test_top1:.4f}  top3={test_top3:.4f}  top5={test_top5:.4f}"
        )

        final_results = {
            "best_epoch": ckpt["epoch"],
            "best_val_loss": ckpt["val_loss"],
            "test_loss": test_loss,
            "test_top1_acc": test_top1,
            "test_top3_acc": test_top3,
            "test_top5_acc": test_top5,
            "student_config": student_config,
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "teacher_best_epoch": teacher_ckpt.get("epoch"),
            "teacher_best_val_loss": teacher_ckpt.get("val_loss"),
        }
        save_json(out_dir / "final_results.json", final_results)

        wandb.log(
            {
                "test/loss": test_loss,
                "test/top1_acc": test_top1,
                "test/top3_acc": test_top3,
                "test/top5_acc": test_top5,
                "best/epoch": ckpt["epoch"],
                "best/val_loss": ckpt["val_loss"],
            }
        )

        print(f"\nEntrainement termine. Artefacts dans : {out_dir}/")

    finally:
        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Entraine un student LSTM avec distillation a partir d'un teacher baseline."
    )
    ap.add_argument("--parsed-dir", default="artifacts/parsed_sched_switch")
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--output-dir", default="artifacts/training_runs/lstm_distilled")

    ap.add_argument("--window-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--student-embed-dim", type=int, default=64)
    ap.add_argument("--student-hidden-size", type=int, default=128)
    ap.add_argument("--student-num-layers", type=int, default=1)
    ap.add_argument(
        "--student-dropout",
        type=float,
        default=0.0,
        help="Dropout du student (actif seulement si num_layers > 1)",
    )

    ap.add_argument(
        "--teacher-embed-dim",
        type=int,
        default=128,
        help="Utilise seulement si le checkpoint teacher ne contient pas model_config.",
    )
    ap.add_argument(
        "--teacher-hidden-size",
        type=int,
        default=256,
        help="Utilise seulement si le checkpoint teacher ne contient pas model_config.",
    )
    ap.add_argument(
        "--teacher-num-layers",
        type=int,
        default=1,
        help="Utilise seulement si le checkpoint teacher ne contient pas model_config.",
    )
    ap.add_argument(
        "--teacher-dropout",
        type=float,
        default=0.0,
        help="Utilise seulement si le checkpoint teacher ne contient pas model_config.",
    )

    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    ap.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Clipping du gradient; mettre <= 0 pour desactiver.",
    )
    ap.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Sauvegarde un checkpoint a chaque epoque.",
    )

    ap.add_argument("--wandb-project", default="GLO-7030-projet")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-dir", default="artifacts/wandb")

    main(ap.parse_args())
