"""The shared market sequence encoder and per-stock head."""

import torch
from market import MARKET_COLUMNS, DateBatch
from runtime import Config
from torch import nn


class GRUSequence(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gru = nn.GRU(len(MARKET_COLUMNS), config.hidden_size, config.gru_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.head_hidden_size),
            nn.ReLU(),
            nn.Linear(config.head_hidden_size, 1),
        )

    def forward(self, batch: DateBatch) -> torch.Tensor:
        _, states = self.gru(batch.market)
        return self.head(states[-1]).squeeze(-1)
