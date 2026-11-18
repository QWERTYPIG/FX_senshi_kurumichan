"""Leakage-conscious stock features for AdaBoost training and inference."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd


FEATURE_VERSION = 1
FEATURES = [
    "intraday_lag1",
    "close_return_1",
    "close_return_2",
    "close_return_5",
    "close_return_10",
    "close_return_20",
    "range_lag1",
    "gap_lag1",
    "intraday_mean_5",
    "intraday_mean_20",
    "intraday_std_5",
    "intraday_std_20",
    "close_return_mean_5",
    "close_return_std_5",
    "close_return_mean_20",
    "close_return_std_20",
    "volume_change_1",
    "volume_ratio_5",
    "volume_ratio_20",
    "turnover_lag1",
    "rsi_lag1",
    "close_to_sma",
    "bbands_position",
    "bbands_width",
    "macd_ratio",
    "atr_ratio",
    "foreign_holding_ratio",
    "weekday",
    "month",
    "stock_age_days",
]

REQUIRED_COLUMNS = [
    "date", "etl:adj_open", "etl:adj_close", "etl:adj_high", "etl:adj_low",
    "price:成交股數",
]


def discover_stock_ids(data_dir: Path | str, history_dir: Path | str | None = None) -> list[str]:
    """Return the union of market CSV names from historical and live data."""
    found: set[str] = set()
    searched: list[Path] = []
    for root in (history_dir, data_dir):
        if root is None:
            continue
        market = Path(root) / "market"
        searched.append(market.resolve())
        if market.is_dir():
            found.update(path.stem for path in market.glob("*.csv"))
    if not found:
        locations = ", ".join(str(path) for path in searched)
        raise FileNotFoundError(
            "No stock CSV files were found. "
            f"Searched: {locations}. "
            "Place the historical training set under basic_data/market and the daily update "
            "under data/market, or pass --history-dir and --data-dir explicitly. "
            "When using the manager, configure data_access_key in the repo .env and synchronize data first."
        )
    return sorted(found)


def load_stock_history(live_path: Path, history_path: Path | None = None) -> pd.DataFrame:
    """Merge static and live rows; a live row replaces an overlapping static row."""
    pieces: list[pd.DataFrame] = []
    for path in (history_path, live_path):
        if path is None or not path.is_file():
            continue
        part = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS) - set(part.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        part["date"] = pd.to_datetime(part["date"], errors="coerce")
        if part["date"].isna().any() or part["date"].duplicated().any():
            raise ValueError(f"Invalid or duplicate dates in {path}")
        pieces.append(part)
    if not pieces:
        raise ValueError(f"No history found for stock {live_path.stem}")
    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def stock_features(live_path: Path, history_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build end-of-day feature states and same-day labels for one stock."""
    raw = load_stock_history(live_path, history_path)
    for name in REQUIRED_COLUMNS[1:]:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")

    price_columns = ["etl:adj_open", "etl:adj_close", "etl:adj_high", "etl:adj_low"]
    valid = raw[price_columns].gt(0).all(axis=1) & raw["price:成交股數"].ge(0)
    raw = raw.loc[valid].copy().reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"No valid positive-price rows in {live_path}")

    open_ = raw["etl:adj_open"]
    close = raw["etl:adj_close"]
    high = raw["etl:adj_high"]
    low = raw["etl:adj_low"]
    volume = raw["price:成交股數"]
    intraday = close / open_ - 1
    close_return = close.pct_change(fill_method=None)
    bb_upper = _numeric(raw, "BBANDS_upper")
    bb_lower = _numeric(raw, "BBANDS_lower")
    bb_width = bb_upper - bb_lower

    state = pd.DataFrame({
        "feature_date": raw["date"],
        "intraday_lag1": intraday,
        "close_return_1": close_return,
        "close_return_2": close.pct_change(2, fill_method=None),
        "close_return_5": close.pct_change(5, fill_method=None),
        "close_return_10": close.pct_change(10, fill_method=None),
        "close_return_20": close.pct_change(20, fill_method=None),
        "range_lag1": (high - low) / open_,
        "gap_lag1": open_ / close.shift(1) - 1,
        "intraday_mean_5": intraday.rolling(5).mean(),
        "intraday_mean_20": intraday.rolling(20).mean(),
        "intraday_std_5": intraday.rolling(5).std(),
        "intraday_std_20": intraday.rolling(20).std(),
        "close_return_mean_5": close_return.rolling(5).mean(),
        "close_return_std_5": close_return.rolling(5).std(),
        "close_return_mean_20": close_return.rolling(20).mean(),
        "close_return_std_20": close_return.rolling(20).std(),
        "volume_change_1": volume.pct_change(fill_method=None),
        "volume_ratio_5": volume / volume.rolling(5).mean().replace(0, np.nan),
        "volume_ratio_20": volume / volume.rolling(20).mean().replace(0, np.nan),
        "turnover_lag1": _numeric(raw, "price:成交金額") / 1e8,
        "rsi_lag1": _numeric(raw, "RSI") / 100,
        "close_to_sma": close / _numeric(raw, "SMA") - 1,
        "bbands_position": (close - bb_lower) / bb_width.replace(0, np.nan),
        "bbands_width": bb_width / close,
        "macd_ratio": _numeric(raw, "MACD") / close,
        "atr_ratio": _numeric(raw, "ATR") / close,
        "foreign_holding_ratio": _numeric(
            raw, "foreign_investors_shareholding:全體外資及陸資持股比率"
        ) / 100,
    }).replace([np.inf, -np.inf], np.nan)
    labels = pd.DataFrame({"date": raw["date"], "intraday_return": intraday})
    return state, labels


def build_dataset(
    data_dir: Path | str,
    stock_ids: Iterable[str],
    dates: Iterable[str | pd.Timestamp] | None = None,
    *,
    history_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Build rows whose inputs always come from a date strictly before the target."""
    data_dir = Path(data_dir)
    history_root = Path(history_dir) if history_dir is not None else None
    requested_dates = None if dates is None else pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
    frames: list[pd.DataFrame] = []
    for stock_id in stock_ids:
        live_path = data_dir / "market" / f"{stock_id}.csv"
        history_path = (
            history_root / "market" / f"{stock_id}.csv" if history_root is not None else None
        )
        state, labels = stock_features(live_path, history_path)
        queries = labels if requested_dates is None else pd.DataFrame({"date": requested_dates})
        joined = pd.merge_asof(
            queries.sort_values("date"),
            state.sort_values("feature_date"),
            left_on="date",
            right_on="feature_date",
            direction="backward",
            allow_exact_matches=False,
        )
        joined["stock_id"] = str(stock_id)
        joined["weekday"] = joined["date"].dt.day_of_week
        joined["month"] = joined["date"].dt.month
        joined["stock_age_days"] = (joined["date"] - joined["feature_date"]).dt.days
        frames.append(joined)
    if not frames:
        raise ValueError("At least one stock ID is required")
    result = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return result.sort_values(["date", "stock_id"]).reset_index(drop=True)
