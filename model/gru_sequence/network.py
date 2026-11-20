"""The shared sequence encoder and cutoff-aware per-stock head."""

import torch
from global_history import GLOBAL_COLUMNS, US_COLUMNS
from market import MARKET_COLUMNS, DateBatch
from runtime import Config
from torch import nn


class GRUSequence(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gru = nn.GRU(len(MARKET_COLUMNS) + len(GLOBAL_COLUMNS), config.hidden_size,
                          config.gru_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size + len(US_COLUMNS), config.head_hidden_size),
            nn.ReLU(),
            nn.Linear(config.head_hidden_size, 1),
        )

    def forward(self, batch: DateBatch) -> torch.Tensor:
        _, states = self.gru(batch.sequence)
        today = batch.today.expand(len(batch.present), -1)
        return self.head(torch.cat((states[-1], today), dim=-1)).squeeze(-1)
