"""Shared causal features: a date's input uses only strictly earlier stock rows."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
DATA_DIR = Path(os.environ.get("MANAGER_DATA_DIR", ROOT / "data"))
HISTORY_DIR = ROOT / "basic_data"
# Standalone execution should reuse the manager's latest generated template.
# The canonical all_down file is a stock-list / Phase 1 reference, not today's template.
TEMPLATE = Path(os.environ.get("MANAGER_TEMPLATE", ROOT / "manager_data/template.csv"))
FEATURE_VERSION = 3
FEATURES = [
    "stock_id", "intraday_lag1", "return_lag1", "range_lag1", "gap_lag1",
    "volume_ratio_5", "return_mean_5", "return_std_5", "return_mean_10",
    "rsi_lag1", "close_to_sma", "atr_ratio", "macd_ratio", "weekday",
]


def read_template(path):
    template = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str})
    if template.columns[0] != "id" or template["id"].duplicated().any():
        raise ValueError("Template must start with id and contain unique stock IDs")
    if len(template.columns) < 2 or template.empty:
        raise ValueError("Template must contain stocks and prediction dates")
    for date in template.columns[1:]:
        pd.to_datetime(date, format="%Y-%m-%d", errors="raise")
    return template


def discover_stock_ids(data_dir, history_dir=None):
    """Discover the full stock universe on every invocation, without a cached list."""
    directories = [Path(data_dir)]
    if history_dir is not None:
        directories.append(Path(history_dir))
    stocks = set()
    for directory in directories:
        market = directory / "market"
        if not market.is_dir():
            raise ValueError(f"Missing market directory: {market}")
        stocks.update(path.stem for path in market.glob("*.csv"))
    if not stocks:
        raise ValueError("No stock CSV files found")
    return sorted(stocks)


def load_stock_history(path, history_path=None):
    """Combine static history with live updates; a live row replaces a static row."""
    parts = []
    for source in (history_path, path):
        if source is None or not source.is_file():
            continue
        part = pd.read_csv(source, parse_dates=["date"])
        if part["date"].isna().any() or part["date"].duplicated().any():
            raise ValueError(f"Invalid or duplicated dates in {source}")
        parts.append(part)
    if not parts:
        raise ValueError(f"No history found for {path.stem}")
    return (pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last")
            .sort_values("date").reset_index(drop=True))


def stock_features(path, history_path=None):
    df = load_stock_history(path, history_path)
    # Do not use the supplied y column; derive the PDF's target from prices.
    required = ["etl:adj_open", "etl:adj_close", "etl:adj_high", "etl:adj_low",
                "price:成交股數"]
    for name in required:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    # Remove invalid market rows BEFORE computing returns and rolling indicators.
    # The supplied data uses -1 for missing observations (e.g. 2026-10-23).
    valid_prices = df[["etl:adj_open", "etl:adj_close", "etl:adj_high", "etl:adj_low"]].gt(0).all(axis=1)
    valid_volume = df["price:成交股數"].ge(0)
    df = df.loc[valid_prices & valid_volume].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid price history in {path}")

    close = df["etl:adj_close"]
    open_ = df["etl:adj_open"]
    volume = df["price:成交股數"]
    ret = close.pct_change(fill_method=None)

    def optional(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df else np.nan

    # Each row describes information available AFTER this source date's close.
    # build_dataset joins it to strictly later target dates (equivalent to shift(1)
    # on the observed trading calendar, and also works for an unseen future date).
    result = pd.DataFrame({
        "feature_date": df["date"],
        "intraday_lag1": close / open_ - 1,
        "return_lag1": ret,
        "range_lag1": (df["etl:adj_high"] - df["etl:adj_low"]) / open_,
        "gap_lag1": open_ / close.shift(1) - 1,
        "volume_ratio_5": volume / volume.rolling(5).mean().replace(0, np.nan),
        "return_mean_5": ret.rolling(5).mean(),
        "return_std_5": ret.rolling(5).std(),
        "return_mean_10": ret.rolling(10).mean(),
        "rsi_lag1": optional("RSI"),
        "close_to_sma": close / optional("SMA") - 1,
        "atr_ratio": optional("ATR") / close,
        "macd_ratio": optional("MACD") / close,
    }).replace([np.inf, -np.inf], np.nan)
    labels = pd.DataFrame({"date": df["date"], "intraday_return": close / open_ - 1})
    return result, labels


def build_dataset(data_dir, stock_ids, dates=None, *, history_dir=None, category_ids=None):
    """Training uses observed target dates; inference can request future dates."""
    frames = []
    for stock_id in stock_ids:
        history_path = (Path(history_dir) / "market" / f"{stock_id}.csv"
                        if history_dir is not None else None)
        source, labels = stock_features(Path(data_dir) / "market" / f"{stock_id}.csv", history_path)
        queries = (labels if dates is None else
                   pd.DataFrame({"date": pd.to_datetime(list(dates))}).sort_values("date"))
        joined = pd.merge_asof(
            queries, source, left_on="date", right_on="feature_date",
            direction="backward", allow_exact_matches=False,
        )
        joined["stock_id"] = stock_id
        joined["weekday"] = joined["date"].dt.day_of_week
        frames.append(joined)
    result = pd.concat(frames, ignore_index=True)
    categories = stock_ids if category_ids is None else category_ids
    if not set(stock_ids).issubset(categories):
        raise ValueError("Requested stocks are absent from the trained category universe; retrain")
    result["stock_id"] = pd.Categorical(result["stock_id"], categories=categories)
    return result.sort_values(["date", "stock_id"]).reset_index(drop=True)
