"""PyTorch models for irregularly sampled clinical event sequences."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from temporal_data import SequenceExample


class SequenceDataset(Dataset[SequenceExample]):
    def __init__(self, examples: Sequence[SequenceExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SequenceExample:
        return self.examples[index]


def collate_sequences(batch: Sequence[SequenceExample]) -> dict[str, Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(example.length for example in batch)
    dim = batch[0].x.shape[1]
    x = torch.zeros((len(batch), max_len, dim), dtype=torch.float32)
    observed_mask = torch.zeros_like(x)
    delta_hours = torch.zeros((len(batch), max_len), dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.float32)
    for index, example in enumerate(batch):
        length = example.length
        x[index, :length] = torch.from_numpy(example.x)
        observed_mask[index, :length] = torch.from_numpy(example.observed_mask)
        delta_hours[index, :length] = torch.from_numpy(example.delta_hours)
        lengths[index] = length
        labels[index] = float(example.label)
    return {
        "x": x,
        "observed_mask": observed_mask,
        "delta_hours": delta_hours,
        "lengths": lengths,
        "labels": labels,
    }


def _last_valid(hidden: Tensor, lengths: Tensor) -> Tensor:
    indices = (lengths - 1).clamp_min(0).to(hidden.device)
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, indices]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.15, pooling: str = "legacy_last_output"):
        super().__init__()
        if pooling not in {"legacy_last_output", "final_hidden"}:
            raise ValueError(f"Unknown recurrent pooling: {pooling}")
        self.pooling = pooling
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x: Tensor, observed_mask: Tensor, delta_hours: Tensor, lengths: Tensor) -> Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp_min(1), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, _) = self.encoder(packed)
        if self.pooling == "final_hidden":
            return self.head(torch.cat([hidden[-2], hidden[-1]], dim=-1)).squeeze(-1)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.head(_last_valid(output, lengths)).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.15, pooling: str = "legacy_last_output"):
        super().__init__()
        if pooling not in {"legacy_last_output", "final_hidden"}:
            raise ValueError(f"Unknown recurrent pooling: {pooling}")
        self.pooling = pooling
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x: Tensor, observed_mask: Tensor, delta_hours: Tensor, lengths: Tensor) -> Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp_min(1), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.encoder(packed)
        if self.pooling == "final_hidden":
            return self.head(torch.cat([hidden[-2], hidden[-1]], dim=-1)).squeeze(-1)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.head(_last_valid(output, lengths)).squeeze(-1)


class GRUDClassifier(nn.Module):
    """A compact GRU-D variant using event-to-event time gaps.

    Numeric measurement masks are supplied by the feature schema.  Ontology
    states and relation counts are treated as observed at each event; their
    values are still allowed to change across events.  Missing numeric values
    are exponentially decayed toward the training mean (zero after scaling).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.decay_rate = nn.Parameter(torch.full((input_dim,), -1.5))
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, observed_mask: Tensor, delta_hours: Tensor, lengths: Tensor) -> Tensor:
        batch, steps, dim = x.shape
        hidden = torch.zeros((batch, self.cell.hidden_size), dtype=x.dtype, device=x.device)
        previous_x = torch.zeros((batch, dim), dtype=x.dtype, device=x.device)
        rates = torch.nn.functional.softplus(self.decay_rate).view(1, 1, dim)
        for step in range(steps):
            valid = (step < lengths).to(x.dtype).view(batch, 1)
            observed = observed_mask[:, step] * valid
            delta = delta_hours[:, step].clamp_min(0.0).view(batch, 1, 1)
            decay = torch.exp(-(delta * rates).clamp(max=20.0)).squeeze(1)
            current = observed * x[:, step] + (1.0 - observed) * (
                decay * previous_x
            )
            next_hidden = self.cell(current, hidden)
            hidden = valid * next_hidden + (1.0 - valid) * hidden
            previous_x = observed * x[:, step] + (1.0 - observed) * previous_x
        return self.head(hidden).squeeze(-1)


class TimeAwareTransformerClassifier(nn.Module):
    """Transformer with continuous time-gap and admission-time embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, observed_mask: Tensor, delta_hours: Tensor, lengths: Tensor) -> Tensor:
        # The final two channels are scaled delta_hours and hours_from_admission.
        time_values = x[..., -2:]
        non_time = torch.cat([x[..., :-2], torch.zeros_like(time_values)], dim=-1)
        hidden = self.input_projection(non_time) + self.time_projection(time_values)
        positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, -1)
        padding = positions >= lengths.to(hidden.device).view(-1, 1)
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        return self.head(_last_valid(hidden, lengths)).squeeze(-1)

    def forward_with_attention(
        self,
        x: Tensor,
        observed_mask: Tensor,
        delta_hours: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return logits and per-layer/per-head event attention in evaluation mode.

        The calculation mirrors ``TransformerEncoderLayer`` with ``norm_first``
        enabled. Attention is an auxiliary description of information flow;
        feature attribution should additionally use gradients or perturbations.
        """
        time_values = x[..., -2:]
        non_time = torch.cat([x[..., :-2], torch.zeros_like(time_values)], dim=-1)
        hidden = self.input_projection(non_time) + self.time_projection(time_values)
        positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, -1)
        padding = positions >= lengths.to(hidden.device).view(-1, 1)
        attentions: list[Tensor] = []
        for layer in self.encoder.layers:
            normalised = layer.norm1(hidden)
            attended, weights = layer.self_attn(
                normalised,
                normalised,
                normalised,
                key_padding_mask=padding,
                need_weights=True,
                average_attn_weights=False,
            )
            hidden = hidden + layer.dropout1(attended)
            feedforward_input = layer.norm2(hidden)
            feedforward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(feedforward_input)))
            )
            hidden = hidden + layer.dropout2(feedforward)
            attentions.append(weights)
        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        logits = self.head(_last_valid(hidden, lengths)).squeeze(-1)
        return logits, torch.stack(attentions, dim=1)


class EnhancedTimeAwareTransformerClassifier(nn.Module):
    """Time-aware Transformer with explicit missingness and position encoding.

    This is kept separate from the locked primary model so older checkpoints
    remain loadable.  Hyperparameters must be selected on validation data only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
        max_seq_len: int = 24,
        pooling: str = "attention",
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if pooling not in {"last", "attention", "multiscale"}:
            raise ValueError("pooling must be last, attention, or multiscale")
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.observed_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.time_projection = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool_score = (
            nn.Linear(hidden_dim, 1) if pooling in {"attention", "multiscale"} else None
        )
        pooled_dim = hidden_dim * 4 if pooling == "multiscale" else hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim), nn.Dropout(dropout), nn.Linear(pooled_dim, 1)
        )

    def forward(self, x: Tensor, observed_mask: Tensor, delta_hours: Tensor, lengths: Tensor) -> Tensor:
        time_values = x[..., -2:]
        non_time = torch.cat([x[..., :-2], torch.zeros_like(time_values)], dim=-1)
        positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
        padding = positions >= lengths.to(x.device).view(-1, 1)
        hidden = (
            self.input_projection(non_time)
            + self.observed_projection(observed_mask)
            + self.time_projection(time_values)
            + self.position_embedding(positions)
        )
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        if self.pooling == "last":
            pooled = _last_valid(hidden, lengths)
        else:
            scores = self.pool_score(hidden).squeeze(-1).masked_fill(padding, -1e9)
            weights = torch.softmax(scores, dim=1)
            attention_pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
            if self.pooling == "attention":
                pooled = attention_pooled
            else:
                valid = (~padding).unsqueeze(-1)
                count = valid.sum(dim=1).clamp_min(1).to(hidden.dtype)
                mean_pooled = (hidden * valid).sum(dim=1) / count
                max_pooled = hidden.masked_fill(~valid, -torch.inf).max(dim=1).values
                last_pooled = _last_valid(hidden, lengths)
                pooled = torch.cat(
                    [last_pooled, mean_pooled, max_pooled, attention_pooled], dim=-1
                )
        return self.head(pooled).squeeze(-1)

    def forward_with_attention(
        self, x: Tensor, observed_mask: Tensor, delta_hours: Tensor, lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, encoder attention, and decision-pooling weights.

        Pooling weights are the direct event-level weights used by the enhanced
        model. Encoder attention is retained as an auxiliary diagnostic only.
        """
        time_values = x[..., -2:]
        non_time = torch.cat([x[..., :-2], torch.zeros_like(time_values)], dim=-1)
        positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
        padding = positions >= lengths.to(x.device).view(-1, 1)
        hidden = (
            self.input_projection(non_time)
            + self.observed_projection(observed_mask)
            + self.time_projection(time_values)
            + self.position_embedding(positions)
        )
        attentions: list[Tensor] = []
        for layer in self.encoder.layers:
            normalised = layer.norm1(hidden)
            attended, weights = layer.self_attn(
                normalised, normalised, normalised, key_padding_mask=padding,
                need_weights=True, average_attn_weights=False,
            )
            hidden = hidden + layer.dropout1(attended)
            ff_input = layer.norm2(hidden)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
            hidden = hidden + layer.dropout2(ff)
            attentions.append(weights)
        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        if self.pooling == "last":
            pooled = _last_valid(hidden, lengths)
            pool_weights = torch.zeros_like(padding, dtype=hidden.dtype)
            pool_weights.scatter_(1, (lengths - 1).clamp_min(0).view(-1, 1), 1.0)
        else:
            scores = self.pool_score(hidden).squeeze(-1).masked_fill(padding, -1e9)
            pool_weights = torch.softmax(scores, dim=1)
            attention_pooled = torch.sum(hidden * pool_weights.unsqueeze(-1), dim=1)
            if self.pooling == "attention":
                pooled = attention_pooled
            else:
                valid = (~padding).unsqueeze(-1)
                count = valid.sum(dim=1).clamp_min(1).to(hidden.dtype)
                mean_pooled = (hidden * valid).sum(dim=1) / count
                max_pooled = hidden.masked_fill(~valid, -torch.inf).max(dim=1).values
                last_pooled = _last_valid(hidden, lengths)
                pooled = torch.cat(
                    [last_pooled, mean_pooled, max_pooled, attention_pooled], dim=-1
                )
        return self.head(pooled).squeeze(-1), torch.stack(attentions, dim=1), pool_weights


MODEL_NAMES = ("lstm", "gru", "gru_d", "transformer")


def make_sequence_model(name: str, input_dim: int, hidden_dim: int = 128,
                        recurrent_pooling: str = "legacy_last_output") -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMClassifier(input_dim, hidden_dim=hidden_dim, pooling=recurrent_pooling)
    if name == "gru":
        return GRUClassifier(input_dim, hidden_dim=hidden_dim, pooling=recurrent_pooling)
    if name == "gru_d":
        return GRUDClassifier(input_dim, hidden_dim=hidden_dim)
    if name in {"transformer", "time_transformer", "time-aware-transformer"}:
        transformer_hidden = max(96, hidden_dim)
        return TimeAwareTransformerClassifier(input_dim, hidden_dim=transformer_hidden)
    raise ValueError(f"Unknown sequence model: {name}")


@dataclass
class TrainState:
    epoch: int
    train_loss: float
    validation_auprc: float
