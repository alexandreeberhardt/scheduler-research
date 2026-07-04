import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SchedSwitchDataset(Dataset):
    """
    Fenêtres glissantes sur des séquences de tokens d'ordonnancement.

    Chaque item retourne :
        x : LongTensor de forme (window_size,) - tokens en entrée
        y : int - token cible (le suivant)
    """

    def __init__(self, sequences: list[np.ndarray], window_size: int) -> None:
        self.window_size = window_size
        self.sequences = [torch.from_numpy(seq.astype(np.int64)) for seq in sequences]
        self.lengths = [max(0, len(s) - window_size) for s in self.sequences]
        self.cumlen = np.cumsum([0] + self.lengths)

    def __len__(self) -> int:
        return int(self.cumlen[-1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq_i = int(np.searchsorted(self.cumlen[1:], idx, side="right"))
        pos = idx - int(self.cumlen[seq_i])
        seq = self.sequences[seq_i]
        return seq[pos : pos + self.window_size], int(
            seq[pos + self.window_size].item()
        )


def load_sequences(parsed_dir: Path, cpu_lengths: dict) -> list[np.ndarray]:
    """Charge les séquences tokenisées depuis le disque pour chaque CPU."""
    sequences = []
    for cpu in sorted(int(k) for k in cpu_lengths):
        seq = np.load(parsed_dir / f"cpu_{cpu}_tokens.npy")
        sequences.append(seq)
    return sequences


def split_sequences(
    sequences: list[np.ndarray],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Divise chaque séquence CPU en train / val / test en respectant l'ordre temporel."""
    train_seqs, val_seqs, test_seqs = [], [], []
    for seq in sequences:
        n = len(seq)
        t1, t2 = int(n * train_ratio), int(n * (train_ratio + val_ratio))
        train_seqs.append(seq[:t1])
        val_seqs.append(seq[t1:t2])
        test_seqs.append(seq[t2:])
    return train_seqs, val_seqs, test_seqs


def load_dataset_payload(dataset_dir: Path) -> dict:
    with (dataset_dir / "decision_dataset.json").open() as file_handle:
        return json.load(file_handle)


class CandidatePolicyDataset(Dataset):
    def __init__(self, payload: dict, split: str, max_history: int) -> None:
        self.episodes = payload["episodes"]
        self.max_history = max_history

        self.sample_index: list[tuple[int, int]] = []
        for episode_index in payload["splits"][split]:
            episode = self.episodes[episode_index]
            for sample_index in range(len(episode["samples"])):
                self.sample_index.append((episode_index, sample_index))

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict:
        episode_index, sample_index = self.sample_index[index]
        episode = self.episodes[episode_index]
        sample = episode["samples"][sample_index]

        history_start = max(0, sample_index - self.max_history)
        history = [
            episode["samples"][step]["executed_task_features"]
            for step in range(history_start, sample_index)
        ]

        target_task = sample["candidate_tasks"][sample["target_index"]]
        cfs_task = sample["candidate_tasks"][sample["cfs_index"]]
        metadata = {
            "policy": episode["policy"],
            "seed": episode["seed"],
            "taskfile": episode["taskfile"],
            "candidate_count": len(sample["candidate_features"]),
            "target_pid": target_task["pid"],
            "target_name": target_task["name"],
            "cfs_pid": cfs_task["pid"],
            "cfs_name": cfs_task["name"],
        }
        teacher_scores = sample.get("teacher_scores")
        if teacher_scores is None:
            teacher_scores = [0.0] * len(sample["candidate_features"])
            teacher_scores[sample["target_index"]] = 1.0

        return {
            "history_features": torch.tensor(history, dtype=torch.float32),
            "candidate_features": torch.tensor(
                sample["candidate_features"],
                dtype=torch.float32,
            ),
            "all_task_features": torch.tensor(
                sample["all_task_features"],
                dtype=torch.float32,
            ),
            "global_features": torch.tensor(
                sample["global_features"],
                dtype=torch.float32,
            ),
            "target_index": int(sample["target_index"]),
            "teacher_scores": torch.tensor(teacher_scores, dtype=torch.float32),
            "cfs_index": int(sample["cfs_index"]),
            "metadata": metadata,
        }


def collate_candidate_batch(batch: list[dict]) -> dict:
    batch_size = len(batch)
    task_feature_dim = batch[0]["candidate_features"].size(-1)
    global_feature_dim = batch[0]["global_features"].size(-1)
    max_history = max(item["history_features"].size(0) for item in batch)
    max_candidates = max(item["candidate_features"].size(0) for item in batch)
    max_all_tasks = max(item["all_task_features"].size(0) for item in batch)

    history_features = torch.zeros(batch_size, max_history, task_feature_dim, dtype=torch.float32)
    history_mask = torch.zeros(batch_size, max_history, dtype=torch.bool)
    candidate_features = torch.zeros(
        batch_size,
        max_candidates,
        task_feature_dim,
        dtype=torch.float32,
    )
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    all_task_features = torch.zeros(
        batch_size,
        max_all_tasks,
        task_feature_dim,
        dtype=torch.float32,
    )
    all_task_mask = torch.zeros(batch_size, max_all_tasks, dtype=torch.bool)
    global_features = torch.zeros(batch_size, global_feature_dim, dtype=torch.float32)
    target_index = torch.zeros(batch_size, dtype=torch.long)
    teacher_scores = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    cfs_index = torch.zeros(batch_size, dtype=torch.long)
    metadata = []

    for batch_index, item in enumerate(batch):
        history_len = item["history_features"].size(0)
        candidate_len = item["candidate_features"].size(0)
        all_task_len = item["all_task_features"].size(0)

        if history_len > 0:
            history_features[batch_index, :history_len] = item["history_features"]
            history_mask[batch_index, :history_len] = True
        candidate_features[batch_index, :candidate_len] = item["candidate_features"]
        candidate_mask[batch_index, :candidate_len] = True
        all_task_features[batch_index, :all_task_len] = item["all_task_features"]
        all_task_mask[batch_index, :all_task_len] = True
        global_features[batch_index] = item["global_features"]
        target_index[batch_index] = item["target_index"]
        teacher_scores[batch_index, :candidate_len] = item["teacher_scores"]
        cfs_index[batch_index] = item["cfs_index"]
        metadata.append(item["metadata"])

    return {
        "history_features": history_features,
        "history_mask": history_mask,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "all_task_features": all_task_features,
        "all_task_mask": all_task_mask,
        "global_features": global_features,
        "target_index": target_index,
        "teacher_scores": teacher_scores,
        "cfs_index": cfs_index,
        "metadata": metadata,
    }
