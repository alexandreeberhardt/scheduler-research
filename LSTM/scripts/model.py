import math

import torch
import torch.nn as nn


class LSTMScheduler(nn.Module):
    """
    LSTM de référence pour la prédiction du prochain processus ordonnancé.

    Architecture :
        Embedding(vocab_size, embed_dim)
        -> LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        -> Linear(hidden_size, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        out, _ = self.lstm(emb)  # (batch, seq_len, hidden_size)
        return self.classifier(out[:, -1, :])  # (batch, vocab_size)


class CandidateSchedulerModel(nn.Module):
    def __init__(
        self,
        task_feature_dim: int,
        global_feature_dim: int,
        task_hidden_dim: int = 128,
        history_hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.set_layers = max(num_layers, 1)

        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, task_hidden_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, task_hidden_dim),
            nn.ReLU(),
        )
        self.history_adapter = nn.Sequential(
            nn.Linear(task_hidden_dim, history_hidden_dim),
            nn.ReLU(),
            nn.Linear(history_hidden_dim, task_hidden_dim),
        )
        self.recency_adapter = nn.Linear(1, task_hidden_dim)
        self.gate = nn.Linear(task_hidden_dim * 4, 3)
        self.scorer = nn.Sequential(
            nn.Linear(task_hidden_dim * 2, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, 1),
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        all_task_features: torch.Tensor,
        all_task_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = self.task_encoder(candidate_features)
        history_embeddings = self._encode_history(
            self.task_encoder(history_features),
            history_mask,
        )
        all_task_embeddings = self._refine_set(
            self.task_encoder(all_task_features),
            all_task_mask,
        )
        history_context = self._attention(
            candidate_embeddings,
            history_embeddings,
            history_mask,
        )
        all_task_context = self._attention(
            candidate_embeddings,
            all_task_embeddings,
            all_task_mask,
        )
        global_context = self.global_encoder(global_features).unsqueeze(1).expand_as(
            candidate_embeddings
        )

        gates = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        candidate_embeddings,
                        history_context,
                        all_task_context,
                        global_context,
                    ],
                    dim=-1,
                )
            )
        )
        fused_context = candidate_embeddings
        fused_context = fused_context + gates[..., 0:1] * history_context
        fused_context = fused_context + gates[..., 1:2] * all_task_context
        fused_context = fused_context + gates[..., 2:3] * global_context
        scorer_input = torch.cat([candidate_embeddings, fused_context], dim=-1)
        logits = self.scorer(scorer_input).squeeze(-1)

        mask_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(~candidate_mask, mask_value)

    def _add_recency(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        if history_embeddings.size(1) == 0:
            return history_embeddings
        steps = history_mask.cumsum(dim=1).to(history_embeddings.dtype)
        lengths = history_mask.sum(dim=1, keepdim=True).clamp(min=1)
        recency = ((steps - 1.0) / (lengths - 1).clamp(min=1)).unsqueeze(-1)
        recency = recency * history_mask.unsqueeze(-1)
        return history_embeddings + self.recency_adapter(recency)

    def _encode_history(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.history_adapter(self._add_recency(history_embeddings, history_mask))

    def _refine_set(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for _ in range(self.set_layers):
            embeddings = embeddings + self._attention(embeddings, embeddings, mask)
        return embeddings

    def _attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if key_value.size(1) == 0:
            return query.new_zeros(query.size(0), query.size(1), query.size(2))
        scores = torch.matmul(query, key_value.transpose(1, 2))
        scores = scores / math.sqrt(query.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.unsqueeze(1).to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.matmul(weights, key_value)
