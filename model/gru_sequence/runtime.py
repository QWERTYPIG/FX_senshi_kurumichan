"""Shared manager paths, configuration and CPU runtime setup."""

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent


def iso_date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"expected YYYY-MM-DD date, got {value!r}")
    return parsed


@dataclass(frozen=True)
class Config:
    window_length: int
    hidden_size: int
    gru_layers: int
    head_hidden_size: int
    learning_rate: float
    epochs: int
    seed: int

    def __post_init__(self):
        for field in ("window_length", "hidden_size", "gru_layers", "head_hidden_size", "epochs"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if (type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be positive and finite")

    @classmethod
    def read(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ManagerPaths:
    history: Path
    update: Path
    template: Path
    output: Path

    @classmethod
    def read(cls):
        root = Path(os.environ.get("MANAGER_ROOT", MODEL_DIR.parents[1]))
        return cls(
            Path(os.environ.get("MANAGER_BASIC_DATA_DIR", root / "basic_data")),
            Path(os.environ.get("MANAGER_DATA_DIR", root / "data")),
            Path(os.environ.get("MANAGER_TEMPLATE", root / "model/all_down/all_down.csv")),
            Path(os.environ.get("MANAGER_OUTPUT", MODEL_DIR / "prediction.csv")),
        )


def command_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    return parser


def configure_cpu():
    # Small date batches do not benefit from a large BLAS thread pool.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
