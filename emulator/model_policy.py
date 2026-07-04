import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelSelection:
    selected_task_name: Optional[str]
    selected_task_pid: Optional[int]
    confidence: Optional[float]
    reason: str


class SchedulerModelPolicy:
    def __init__(self, run_dir: str, device: Optional[str] = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Torch is required for model modes. Run the emulator from the LSTM environment."
            ) from exc

        repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = repo_root / "LSTM" / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from common import normalize_checkpoint_state_dict
            from model import CandidateSchedulerModel
        except ImportError as exc:
            raise RuntimeError("Unable to import LSTM/scripts/model.py") from exc

        run_path = Path(run_dir).resolve()
        checkpoint_path = run_path / "best_model.pt"
        artifacts_path = run_path / "dataset_artifacts.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Missing dataset artifacts: {artifacts_path}")

        with artifacts_path.open() as fh:
            artifacts = json.load(fh)

        torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        network = CandidateSchedulerModel(**checkpoint["model_config"]).to(torch_device)
        network.load_state_dict(normalize_checkpoint_state_dict(checkpoint["model_state_dict"]))
        network.eval()

        self._torch = torch
        self.device = torch_device
        self.network = network
        self.run_dir = str(run_path)
        self.max_history = int(artifacts["max_history"])
        self.task_feature_dim = int(artifacts["task_feature_dim"])
        self.global_feature_dim = int(artifacts["global_feature_dim"])

    def __deepcopy__(self, memo):
        return self

    def predict(self, decision_state: dict) -> ModelSelection:
        candidate_features = decision_state["candidate_features"]
        candidate_tasks = decision_state["candidate_tasks"]
        if not candidate_features:
            return ModelSelection(None, None, None, "no_candidates")

        history = decision_state["history_features"][-self.max_history :]
        all_task_features = decision_state["all_task_features"]
        global_features = decision_state["global_features"]

        history_tensor = self._tensor([history], fill_dim=self.task_feature_dim)
        history_mask = self._mask_tensor([len(history)], history_tensor.size(1))
        candidate_tensor = self._tensor([candidate_features], fill_dim=self.task_feature_dim)
        candidate_mask = self._mask_tensor([len(candidate_features)], candidate_tensor.size(1))
        all_task_tensor = self._tensor([all_task_features], fill_dim=self.task_feature_dim)
        all_task_mask = self._mask_tensor([len(all_task_features)], all_task_tensor.size(1))
        global_tensor = self._torch.tensor(
            [global_features],
            dtype=self._torch.float32,
            device=self.device,
        )

        with self._torch.inference_mode():
            logits = self.network(
                history_tensor,
                history_mask,
                candidate_tensor,
                candidate_mask,
                all_task_tensor,
                all_task_mask,
                global_tensor,
            )[0]
            probs = self._torch.softmax(logits, dim=0)
            top_index = int(logits.argmax().item())
            confidence = float(probs[top_index].item())

        selected_task = candidate_tasks[top_index]
        return ModelSelection(
            selected_task_name=selected_task["name"],
            selected_task_pid=selected_task["pid"],
            confidence=confidence,
            reason="selected_candidate",
        )

    def _tensor(self, batches: list[list[list[float]]], *, fill_dim: int):
        max_len = max((len(items) for items in batches), default=0)
        max_len = max(max_len, 1)
        tensor = self._torch.zeros(
            len(batches),
            max_len,
            fill_dim,
            dtype=self._torch.float32,
            device=self.device,
        )
        for batch_index, items in enumerate(batches):
            if items:
                tensor[batch_index, : len(items)] = self._torch.tensor(
                    items,
                    dtype=self._torch.float32,
                    device=self.device,
                )
        return tensor

    def _mask_tensor(self, lengths: list[int], width: int):
        mask = self._torch.zeros(
            len(lengths),
            max(width, 1),
            dtype=self._torch.bool,
            device=self.device,
        )
        for row_index, length in enumerate(lengths):
            if length > 0:
                mask[row_index, :length] = True
        return mask
