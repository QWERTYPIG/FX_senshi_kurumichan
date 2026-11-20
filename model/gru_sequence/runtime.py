"""Shared manager paths, configuration and CPU runtime setup."""

import argparse
import math
import os
import platform
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_NAME = "checkpoint.pt"
DEPLOYMENT_SEED_COUNT = 5
SEARCH_SEED_COUNT = 3


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
    input_noise_std: float = 0.1

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
        if (type(self.input_noise_std) not in (int, float)
                or not math.isfinite(self.input_noise_std) or self.input_noise_std < 0):
            raise ValueError("input_noise_std must be nonnegative and finite")

    @classmethod
    def from_settings(cls, settings):
        sections = {"evaluation", "search", "deployment_mode", "deployment_seeds", "epoch_selection"}
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


def deployment_seeds(settings, config, *, allow_standalone=True):
    mode = settings.get("deployment_mode", "ensemble")
    if mode == "standalone" and allow_standalone:
        if "deployment_seeds" in settings:
            raise ValueError("standalone uses seed, not deployment_seeds")
        return (config.seed,)
    if mode != "ensemble":
        raise ValueError("deployment_mode must be ensemble" + (" or standalone" if allow_standalone else ""))
    return read_seeds(settings.get("deployment_seeds"), DEPLOYMENT_SEED_COUNT, "deployment_seeds")


def run_metadata(config, history):
    return {
        "config": asdict(config),
        "data_date_range": [history.dates[0].isoformat(), history.dates[-1].isoformat()],
        "global_data_date_range": history.globals.source_date_range(),
        "package_versions": {"python": platform.python_version(), "torch": str(torch.__version__)},
    }


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
