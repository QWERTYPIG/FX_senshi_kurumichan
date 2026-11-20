"""Shared manager paths, configuration and CPU runtime setup."""

import argparse
import math
import os
import platform
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
    attention: bool = True
    embedding: bool = True
    embedding_size: int = 4

    def __post_init__(self):
        for field in ("attention", "embedding"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be a boolean")
        for field in ("window_length", "hidden_size", "gru_layers", "head_hidden_size", "epochs", "embedding_size"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        validate_seed(self.seed)
        if (type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be positive and finite")

    @classmethod
    def from_settings(cls, settings):
        sections = {"evaluation", "search", "deployment_seeds"}
        return cls(**{key: value for key, value in settings.items() if key not in sections})


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")


def read_seeds(values, count, field):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{field} must contain exactly {count} seeds")
    for seed in values:
        validate_seed(seed)
    if len(set(values)) != count:
        raise ValueError(f"{field} seeds must be distinct")
    return tuple(values)


def deployment_seeds(settings, config):
    if "deployment_seeds" not in settings:
        return (config.seed,)  # Preserve single-seed configs for isolated runs.
    return read_seeds(settings["deployment_seeds"], 5, "deployment_seeds")


def package_versions():
    return {"python": platform.python_version(), "torch": torch.__version__}


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
