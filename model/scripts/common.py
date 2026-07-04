import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CandidatePolicyDataset, collate_candidate_batch, load_dataset_payload
from model import CandidateSchedulerModel


def evaluate_topk(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float, float, float]:
    """Évalue loss, top-1, top-3 et top-5 sans dépendre des métriques Poutyne."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    model.eval()

    total_loss = 0.0
    total_count = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)

            total_loss += criterion(logits, y).item() * len(y)
            total_count += len(y)

            top1 = logits.argmax(dim=1)
            top1_hits += (top1 == y).sum().item()

            top3 = logits.topk(min(3, logits.size(1)), dim=1).indices
            top3_hits += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()

            top5 = logits.topk(min(5, logits.size(1)), dim=1).indices
            top5_hits += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

    if total_count == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / total_count,
        top1_hits / total_count,
        top3_hits / total_count,
        top5_hits / total_count,
    )


_CHECKPOINT_KEY_RENAMES = {
    "history_encoder.": "history_adapter.",
    "recency_encoder.": "recency_adapter.",
}


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_model(model: CandidateSchedulerModel, batch: dict) -> torch.Tensor:
    return model(
        batch["history_features"],
        batch["history_mask"],
        batch["candidate_features"],
        batch["candidate_mask"],
        batch["all_task_features"],
        batch["all_task_mask"],
        batch["global_features"],
    )


def normalize_checkpoint_state_dict(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        mapped_key = key
        for old_prefix, new_prefix in _CHECKPOINT_KEY_RENAMES.items():
            if mapped_key.startswith(old_prefix):
                mapped_key = new_prefix + mapped_key[len(old_prefix) :]
                break
        normalized[mapped_key] = value
    return normalized


def mask_candidate_logits(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~candidate_mask, -1e9)


def compute_loss(logits: torch.Tensor, batch: dict, args) -> tuple[torch.Tensor, dict[str, float]]:
    masked_logits = mask_candidate_logits(logits, batch["candidate_mask"])
    ce_loss = F.cross_entropy(masked_logits, batch["target_index"])

    temperature = max(float(args.teacher_temperature), 1e-6)
    teacher_logits = mask_candidate_logits(batch["teacher_scores"] / temperature, batch["candidate_mask"])
    teacher_probs = torch.softmax(teacher_logits, dim=1)
    listwise_loss = F.kl_div(
        torch.log_softmax(masked_logits, dim=1),
        teacher_probs,
        reduction="batchmean",
    )

    cfs_scores = batch["teacher_scores"].gather(1, batch["cfs_index"].unsqueeze(1))
    pair_targets = (batch["teacher_scores"] > cfs_scores + 1e-6).float()
    pair_mask = batch["candidate_mask"].clone()
    pair_mask.scatter_(1, batch["cfs_index"].unsqueeze(1), False)
    pairwise_logits = masked_logits - masked_logits.gather(1, batch["cfs_index"].unsqueeze(1))
    pairwise_loss = F.binary_cross_entropy_with_logits(
        pairwise_logits,
        pair_targets,
        reduction="none",
    )
    pairwise_weight = pair_mask.float()
    pairwise_loss = (pairwise_loss * pairwise_weight).sum() / pairwise_weight.sum().clamp_min(1.0)

    loss = (
        float(args.ce_weight) * ce_loss
        + float(args.listwise_weight) * listwise_loss
        + float(args.pairwise_weight) * pairwise_loss
    )
    return loss, {
        "ce_loss": float(ce_loss.item()),
        "listwise_loss": float(listwise_loss.item()),
        "pairwise_loss": float(pairwise_loss.item()),
    }


def maybe_load_initial_checkpoint(
    model: CandidateSchedulerModel,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> None:
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_state = normalize_checkpoint_state_dict(checkpoint["model_state_dict"])
    model_state = model.state_dict()
    compatible_state = {
        name: value
        for name, value in checkpoint_state.items()
        if name in model_state and model_state[name].shape == value.shape
    }
    model_state.update(compatible_state)
    model.load_state_dict(model_state)


def build_dataloaders(
    args,
    dataset_dir: Path,
    device: torch.device,
) -> tuple[dict, dict[str, DataLoader]]:
    payload = load_dataset_payload(dataset_dir)
    max_history = min(int(args.max_history), int(payload["max_history"]))
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_candidate_batch,
    }

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = CandidatePolicyDataset(payload, split=split, max_history=max_history)
        loaders[split] = DataLoader(
            dataset,
            shuffle=split == "train",
            **loader_args,
        )
    return payload, loaders


def train_one_epoch(
    model: CandidateSchedulerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_top1 = 0
    total_ce = 0.0
    total_listwise = 0.0
    total_pairwise = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = run_model(model, batch)
        loss, components = compute_loss(logits, batch, args)
        loss.backward()
        optimizer.step()

        batch_size = batch["target_index"].size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_ce += components["ce_loss"] * batch_size
        total_listwise += components["listwise_loss"] * batch_size
        total_pairwise += components["pairwise_loss"] * batch_size
        masked_logits = mask_candidate_logits(logits, batch["candidate_mask"])
        total_top1 += (masked_logits.argmax(dim=1) == batch["target_index"]).sum().item()

    return {
        "loss": total_loss / max(total_examples, 1),
        "ce_loss": total_ce / max(total_examples, 1),
        "listwise_loss": total_listwise / max(total_examples, 1),
        "pairwise_loss": total_pairwise / max(total_examples, 1),
        "top1_accuracy": total_top1 / max(total_examples, 1),
    }


def evaluate_model(
    model: CandidateSchedulerModel,
    loader: DataLoader,
    device: torch.device,
    args,
    *,
    collect_predictions: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_candidate_count = 0
    topk_hits = {1: 0, 3: 0, 5: 0}
    total_ce = 0.0
    total_listwise = 0.0
    total_pairwise = 0.0
    prediction_rows = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = run_model(model, batch)
            loss, components = compute_loss(logits, batch, args)
            masked_logits = mask_candidate_logits(logits, batch["candidate_mask"])

            batch_size = batch["target_index"].size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            total_ce += components["ce_loss"] * batch_size
            total_listwise += components["listwise_loss"] * batch_size
            total_pairwise += components["pairwise_loss"] * batch_size
            total_candidate_count += int(batch["candidate_mask"].sum().item())

            for k in topk_hits:
                topk_indices = masked_logits.topk(min(k, masked_logits.size(1)), dim=1).indices
                hits = (topk_indices == batch["target_index"].unsqueeze(1)).any(dim=1)
                topk_hits[k] += int(hits.sum().item())

            if collect_predictions:
                probabilities = torch.softmax(masked_logits, dim=1)
                predicted_indices = masked_logits.argmax(dim=1)
                for row_index, metadata in enumerate(batch["metadata"]):
                    target_index = int(batch["target_index"][row_index].item())
                    predicted_index = int(predicted_indices[row_index].item())
                    prediction_rows.append(
                        {
                            **metadata,
                            "correct": predicted_index == target_index,
                            "predicted_index": predicted_index,
                            "target_index": target_index,
                            "confidence": float(
                                probabilities[row_index, predicted_index].item()
                            ),
                        }
                    )

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "ce_loss": total_ce / max(total_examples, 1),
        "listwise_loss": total_listwise / max(total_examples, 1),
        "pairwise_loss": total_pairwise / max(total_examples, 1),
        "top1_accuracy": topk_hits[1] / max(total_examples, 1),
        "top3_accuracy": topk_hits[3] / max(total_examples, 1),
        "top5_accuracy": topk_hits[5] / max(total_examples, 1),
        "mean_candidate_count": total_candidate_count / max(total_examples, 1),
        "n_samples": total_examples,
    }
    return metrics, prediction_rows


def candidate_bucket(candidate_count: int) -> str:
    if candidate_count <= 1:
        return "1"
    if candidate_count <= 3:
        return "2-3"
    if candidate_count <= 7:
        return "4-7"
    return "8+"


def finalize_breakdown(source: dict) -> dict:
    return {
        key: {
            "support": values["n"],
            "accuracy": values["correct"] / values["n"] if values["n"] else 0.0,
        }
        for key, values in sorted(source.items())
    }


def run_evaluation(args, device: torch.device) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.artifacts_json) as f:
        artifacts = json.load(f)
    with open(args.run_config) as f:
        run_config = json.load(f)

    payload = load_dataset_payload(dataset_dir)
    effective_max_history = min(
        args.max_history, artifacts.get("max_history", args.max_history)
    )

    dataset = CandidatePolicyDataset(
        payload,
        split="test",
        max_history=effective_max_history,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_candidate_batch,
    )

    print(f"Chargement du checkpoint : {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = CandidateSchedulerModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(
        normalize_checkpoint_state_dict(checkpoint["model_state_dict"])
    )

    print("Évaluation en cours ...")
    metrics, prediction_rows = evaluate_model(
        model,
        loader,
        device,
        args,
        collect_predictions=True,
    )

    save_json(output_dir / "eval_metrics.json", metrics)
    save_json(
        output_dir / "evaluation_config.json",
        {
            "dataset_dir": str(dataset_dir),
            "checkpoint": args.checkpoint,
            "artifacts_json": args.artifacts_json,
            "run_config": args.run_config,
            "requested_max_history": args.max_history,
            "effective_max_history": effective_max_history,
        },
    )

    # Breakdown des performances
    from collections import defaultdict

    def empty_stats():
        return {"n": 0, "correct": 0}

    by_policy = defaultdict(empty_stats)
    by_bucket = defaultdict(empty_stats)
    by_target = defaultdict(empty_stats)
    for row in prediction_rows:
        by_policy[row["policy"]]["n"] += 1
        by_policy[row["policy"]]["correct"] += int(row["correct"])
        bucket = candidate_bucket(row["candidate_count"])
        by_bucket[bucket]["n"] += 1
        by_bucket[bucket]["correct"] += int(row["correct"])
        by_target[row["target_name"]]["n"] += 1
        by_target[row["target_name"]]["correct"] += int(row["correct"])

    save_json(
        output_dir / "selection_breakdown.json",
        {
            "by_policy": finalize_breakdown(by_policy),
            "by_candidate_count": finalize_breakdown(by_bucket),
            "by_target_name": finalize_breakdown(by_target),
        },
    )

    # Résumé additionnel si présent dans le checkpoint (distillation)
    if "val_loss" in checkpoint:
        distilled_summary = {
            "checkpoint": str(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "val_top1_acc": checkpoint.get("val_top1_acc")
            or checkpoint.get("val_top1_accuracy"),
            "model_config": checkpoint.get("model_config"),
            "teacher": run_config.get("teacher"),
            "temperature": run_config.get("teacher_temperature"),
        }
        save_json(output_dir / "distillation_summary.json", distilled_summary)

    print("\nMetriques de test")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} : {v:.4f}")
        else:
            print(f"  {k:<20} : {v}")

    print(f"\nÉvaluation terminée. Résultats dans : {output_dir}/")


def train(args, dataset_dir: Path, out_dir: Path, wandb_run=None) -> None:
    set_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    payload, loaders = build_dataloaders(args, dataset_dir, device)
    train_size = len(loaders["train"].dataset)
    val_size = len(loaders["val"].dataset)
    test_size = len(loaders["test"].dataset)
    print(f"Dataset : train={train_size:,}  val={val_size:,}  test={test_size:,}")

    model_config = {
        "task_feature_dim": len(payload["task_feature_names"]),
        "global_feature_dim": len(payload["global_feature_names"]),
        "task_hidden_dim": args.task_hidden_dim,
        "history_hidden_dim": args.history_hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = CandidateSchedulerModel(**model_config).to(device)
    maybe_load_initial_checkpoint(
        model,
        getattr(args, "init_checkpoint", None),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_json(
        out_dir / "dataset_artifacts.json",
        {
            "dataset_dir": str(dataset_dir),
            "teacher": payload["teacher"],
            "max_history": min(int(args.max_history), int(payload["max_history"])),
            "task_feature_names": payload["task_feature_names"],
            "global_feature_names": payload["global_feature_names"],
            "split_episode_counts": {
                split: len(payload["splits"][split]) for split in ("train", "val", "test")
            },
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "task_feature_dim": len(payload["task_feature_names"]),
            "global_feature_dim": len(payload["global_feature_names"]),
        },
    )
    save_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "device": str(device),
            "teacher": payload["teacher"],
            "task_feature_dim": len(payload["task_feature_names"]),
            "global_feature_dim": len(payload["global_feature_names"]),
        },
    )

    best_checkpoint_path = out_dir / "best_model.pt"
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device, args)
        val_metrics, _ = evaluate_model(model, loaders["val"], device, args)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_listwise_loss": train_metrics["listwise_loss"],
            "train_pairwise_loss": train_metrics["pairwise_loss"],
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_ce_loss": val_metrics["ce_loss"],
            "val_listwise_loss": val_metrics["listwise_loss"],
            "val_pairwise_loss": val_metrics["pairwise_loss"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_top3_accuracy": val_metrics["top3_accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
        }
        training_history.append(row)

        print(
            f"Epoch {epoch:02d}  "
            f"train_loss={row['train_loss']:.4f}  "
            f"train_top1={row['train_top1_accuracy']:.4f}  "
            f"val_loss={row['val_loss']:.4f}  "
            f"val_top1={row['val_top1_accuracy']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(row)

        if row["val_loss"] < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": row["val_loss"],
                    "val_top1_accuracy": row["val_top1_accuracy"],
                    "model_config": model_config,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

    save_json(out_dir / "training_history.json", {"epochs": training_history})

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, _ = evaluate_model(model, loaders["test"], device, args)
    final_results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **test_metrics,
    }
    save_json(out_dir / "final_results.json", final_results)
    print(
        "Test : "
        f"loss={test_metrics['loss']:.4f}  "
        f"top1={test_metrics['top1_accuracy']:.4f}  "
        f"top3={test_metrics['top3_accuracy']:.4f}  "
        f"top5={test_metrics['top5_accuracy']:.4f}"
    )
