"""Walk-forward splits on unique trading dates, never individual stock rows."""
import numpy as np
import pandas as pd


def walk_forward_splits(dates, *, folds, valid_days, test_days, early_stop_days, min_train_days):
    if min(folds, valid_days, test_days, early_stop_days, min_train_days) < 1:
        raise ValueError("All cross-validation sizes must be positive")
    dates = pd.DatetimeIndex(sorted(set(dates)))
    first_valid = len(dates) - test_days - folds * valid_days
    if first_valid - early_stop_days < min_train_days:
        required = test_days + folds * valid_days + early_stop_days + min_train_days
        raise ValueError(f"Need at least {required} unique trading dates; found {len(dates)}")
    splits = []
    for fold in range(folds):
        start = first_valid + fold * valid_days
        splits.append({
            "fold": fold + 1,
            "inner_train_end": dates[start - early_stop_days - 1],
            "early_stop_start": dates[start - early_stop_days],
            "train_end": dates[start - 1],
            "valid_start": dates[start],
            "valid_end": dates[start + valid_days - 1],
        })
    return splits, dates[-test_days]


def select_iterations(iterations):
    """A fixed robust rule, selected before looking at the final test."""
    return max(1, int(np.median(iterations)))
