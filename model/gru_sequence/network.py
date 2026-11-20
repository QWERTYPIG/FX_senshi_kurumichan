"""The shared sequence encoder and cutoff-aware per-stock head."""

import torch
from global_history import GLOBAL_COLUMNS, US_COLUMNS
from market import MARKET_COLUMNS, DateBatch
from runtime import Config
from torch import nn


class GRUSequence(nn.Module):
    def __init__(self, config: Config, num_stocks: int):
        super().__init__()
        self.gru = nn.GRU(len(MARKET_COLUMNS) + len(GLOBAL_COLUMNS), config.hidden_size,
                          config.gru_layers, batch_first=True)
        self.attention = (
            nn.MultiheadAttention(config.hidden_size, num_heads=1, batch_first=True)
            if config.attention else None
        )
        self.embedding = nn.Embedding(num_stocks, config.embedding_size) if config.embedding else None
        head_inputs = config.hidden_size + len(US_COLUMNS) + (config.embedding_size if config.embedding else 0)
        self.head = nn.Sequential(
            nn.Linear(head_inputs, config.head_hidden_size),
            nn.ReLU(),
            nn.Linear(config.head_hidden_size, 1),
        )

    def forward(self, batch: DateBatch) -> torch.Tensor:
        # Only present stocks enter the graph; scatter scores back to the universe axis.
        (positions,) = batch.present.nonzero(as_tuple=True)
        scores = batch.sequence.new_zeros(len(batch.present))
        if positions.numel() == 0:
            return scores

        _, states = self.gru(batch.sequence[positions])
        summary = states[-1]
        if self.attention is not None:
            cross_section = summary.unsqueeze(0)
            context, _ = self.attention(cross_section, cross_section, cross_section, need_weights=False)
            summary = summary + context.squeeze(0)
        parts = [summary]
        if self.embedding is not None:
            parts.append(self.embedding(positions))
        parts.append(batch.today.expand(positions.numel(), -1))
        scores[positions] = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        return scores
