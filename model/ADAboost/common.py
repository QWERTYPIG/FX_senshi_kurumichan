"""Shared paths, validation, time splits, and scoring for the AdaBoost model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MANAGER_ROOT", MODEL_DIR.parents[1]))
DATA_DIR = Path(os.environ.get("MANAGER_DATA_DIR", ROOT / "data"))
HISTORY_DIR = Path(os.environ.get("MANAGER_BASIC_DATA_DIR", ROOT / "basic_data"))
TEMPLATE = Path(os.environ.get("MANAGER_TEMPLATE", ROOT / "manager_data/template.csv"))


def read_template(path: Path | str) -> pd.DataFrame:
    """Read a competition template without losing leading zeroes in stock IDs."""
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str})
    if frame.empty or frame.columns[0] != "id" or len(frame.columns) < 2:
        raise ValueError("Template must start with id and contain at least one prediction date")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Template stock IDs must be present and unique")
    for value in frame.columns[1:]:
        try:
            parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid template date {value!r}; expected YYYY-MM-DD") from error
        if parsed.strftime("%Y-%m-%d") != value:
            raise ValueError(f"Invalid template date {value!r}; expected YYYY-MM-DD")
    return frame


def walk_forward_splits(
    dates: Iterable[pd.Timestamp], *, folds: int, valid_days: int,
    test_days: int, min_train_days: int,
) -> tuple[list[dict[str, Any]], pd.Timestamp]:
    """Create expanding-window folds and an untouched final test period."""
    if min(folds, valid_days, test_days, min_train_days) < 1:
        raise ValueError("All split sizes must be positive")
    unique = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(dates)))))
    first_valid = len(unique) - test_days - folds * valid_days
    if first_valid < min_train_days:
        required = min_train_days + folds * valid_days + test_days
        raise ValueError(f"Need at least {required} unique trading dates; found {len(unique)}")
    splits: list[dict[str, Any]] = []
    for index in range(folds):
        start = first_valid + index * valid_days
        splits.append({
            "fold": index + 1,
            "train_end": unique[start - 1],
            "valid_start": unique[start],
            "valid_end": unique[start + valid_days - 1],
        })
    return splits, unique[-test_days]


def evaluate(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    """Compute accuracy and the official mean signed-return score."""
    actual = frame["intraday_return"].to_numpy(dtype=float)
    binary = np.asarray(prediction, dtype=int)
    if len(actual) == 0 or len(binary) != len(actual):
        raise ValueError("Evaluation requires one prediction per non-empty input row")
    if not np.isfinite(actual).all() or not np.isin(binary, [0, 1]).all():
        raise ValueError("Evaluation values must be finite returns and binary predictions")
    contribution = (2 * binary - 1) * actual
    return {
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "accuracy": float(np.mean(binary == (actual > 0))),
        "score": float(np.mean(contribution)),
        "all_up_score": float(np.mean(actual)),
        "all_down_score": float(-np.mean(actual)),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
