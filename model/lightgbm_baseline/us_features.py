"""US-session signals available before a Taiwan 08:25 forecast cutoff."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
US_DATA_DIR = Path(os.environ.get("MANAGER_US_DATA_DIR", ROOT / "us_data"))
US_FEATURE_VERSION = 1
SYMBOLS = ("SPY", "QQQ", "SMH", "TSM", "NVDA")
US_FEATURES = [
    *[f"us_{s.lower()}_{suffix}" for s in SYMBOLS for suffix in ("ret1", "ret5", "since_tw_close")],
    "us_smh_vol20", "us_smh_volume_ratio20", "us_tech_relative", "us_semis_relative",
    "us_tsm_relative", "us_nvda_relative", "us_age_days",
]
MAX_AGE_DAYS = 7


def load_us_session_features(directory, symbol):
    path = Path(directory) / "prices" / f"{symbol}.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError(f"Invalid/duplicated US session dates: {path}")
    if "stock_id" in frame and not frame["stock_id"].astype(str).eq(symbol).all():
        raise ValueError(f"Unexpected stock_id in {path}")
    frame = frame.sort_values("date").reset_index(drop=True)
    for column in ("Open", "High", "Low", "Close", "Volume", "Adj_Close"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    price = frame["Adj_Close"] if "Adj_Close" in frame and frame["Adj_Close"].gt(0).any() else frame["Close"]
    valid = frame[["Open", "High", "Low", "Close"]].gt(0).all(axis=1) & price.gt(0) & frame["Volume"].ge(0)
    frame, price = frame.loc[valid].copy(), price.loc[valid]
    if frame.empty:
        raise ValueError(f"No valid US prices in {path}")
    returns = price.pct_change(fill_method=None)
    # Date is treated as the NY session date. Vendor documents daily updates at
    # 08:00 Taiwan time; use 08:10 after session close as a publication ASSUMPTION.
    close = (frame["date"] + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York")
    available = close.dt.tz_convert("Asia/Taipei").dt.normalize() + pd.Timedelta(hours=8, minutes=10)
    return pd.DataFrame({
        "available_at": available, "source_date": frame["date"], "price": price,
        "ret1": returns, "ret5": price / price.shift(5) - 1,
        "vol20": returns.rolling(20).std(),
        "volume_ratio20": frame["Volume"] / frame["Volume"].rolling(20).mean().replace(0, np.nan),
    }).reset_index(drop=True)


def add_us_features(frame, directory=US_DATA_DIR):
    """Join by information availability, never by a same-date CSV join."""
    queries = frame[["date", "feature_date"]].drop_duplicates().copy()
    queries["_forecast_at"] = queries["date"].dt.tz_localize("Asia/Taipei") + pd.Timedelta(hours=8, minutes=25)
    queries["_previous_tw_close"] = queries["feature_date"].dt.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)
    queries = queries.sort_values("_forecast_at").reset_index(drop=True)
    ages = []
    for symbol in SYMBOLS:
        source = load_us_session_features(directory, symbol).sort_values("available_at")
        latest = pd.merge_asof(queries[["_forecast_at"]], source, left_on="_forecast_at",
                               right_on="available_at", direction="backward")
        previous_queries = queries[["_previous_tw_close"]].copy()
        previous_queries["_row"] = previous_queries.index
        missing = previous_queries["_previous_tw_close"].isna()
        previous = pd.merge_asof(
            previous_queries.loc[~missing].sort_values("_previous_tw_close"),
            source[["available_at", "price"]], left_on="_previous_tw_close",
            right_on="available_at", direction="backward",
        ).set_index("_row")["price"].reindex(queries.index)
        prefix = f"us_{symbol.lower()}"
        queries[f"{prefix}_ret1"] = latest["ret1"].to_numpy()
        queries[f"{prefix}_ret5"] = latest["ret5"].to_numpy()
        queries[f"{prefix}_since_tw_close"] = latest["price"].to_numpy() / previous - 1
        if symbol == "SMH":
            queries["us_smh_vol20"] = latest["vol20"].to_numpy()
            queries["us_smh_volume_ratio20"] = latest["volume_ratio20"].to_numpy()
        ages.append((queries["date"] - latest["source_date"]).dt.days)
    queries["us_age_days"] = pd.concat(ages, axis=1).max(axis=1, skipna=False)
    queries["us_tech_relative"] = queries["us_qqq_ret1"] - queries["us_spy_ret1"]
    queries["us_semis_relative"] = queries["us_smh_ret1"] - queries["us_spy_ret1"]
    queries["us_tsm_relative"] = queries["us_tsm_ret1"] - queries["us_smh_ret1"]
    queries["us_nvda_relative"] = queries["us_nvda_ret1"] - queries["us_smh_ret1"]
    return frame.merge(queries[["date", "feature_date", *US_FEATURES]], on=["date", "feature_date"],
                       how="left", validate="many_to_one").replace([np.inf, -np.inf], np.nan)


def us_coverage(frame):
    return frame[US_FEATURES].notna().all(axis=1) & frame["us_age_days"].le(MAX_AGE_DAYS)
