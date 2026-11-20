"""Market history and time-causal cross-sections on a stable universe stock axis."""

import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

ADJUSTED_OHLC = ("etl:adj_close", "etl:adj_open", "etl:adj_high", "etl:adj_low")
PRICE_COLUMNS = (*ADJUSTED_OHLC, "SMA", "MACD", "ATR", "BBANDS_upper", "BBANDS_mid", "BBANDS_lower")
OTHER_COLUMNS = (
    "price:成交筆數", "price:成交金額", "price:成交股數",
    "foreign_investors_shareholding:全體外資及陸資持股比率",
    "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)",
    "institutional_investors_trading_summary:投信買賣超股數",
    "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)",
    "institutional_investors_trading_summary:自營商買賣超股數(避險)",
    "RSI",
)
MARKET_COLUMNS = PRICE_COLUMNS + OTHER_COLUMNS
CLOSE, OPEN = map(MARKET_COLUMNS.index, ADJUSTED_OHLC[:2])


def iso_date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"expected YYYY-MM-DD date, got {value!r}")
    return parsed


def _stock_ids(history: Path) -> tuple[str, ...]:
    info = json.loads((history / "info.json").read_text(encoding="utf-8-sig"))
    stocks = tuple(map(str, info["Top100@Y"]))
    if not stocks or len(stocks) != len(set(stocks)) or not all(stock.isascii() and stock.isdigit() for stock in stocks):
        raise ValueError("Top100@Y in historical info.json must contain unique stock IDs")
    return stocks


def _read_stock(stock: str, history: Path, update: Path, before: date | None) -> dict[date, list[float]]:
    rows = {}
    expected = {"date", *MARKET_COLUMNS}
    for directory in (history, update):
        path = directory / "market" / f"{stock}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if len(set(fields)) != len(fields) or set(fields) - {"y"} != expected:
                raise ValueError(f"{path}: unexpected market columns")
            for row in reader:
                day = iso_date(row["date"])
                if before is None or day < before:
                    rows[day] = row
    return {day: [float(row[col]) if row[col] else float("nan") for col in MARKET_COLUMNS]
            for day, row in rows.items()}


@dataclass(frozen=True)
class DateBatch:
    """One stock axis for features, absence mask, provenance and optional training labels."""

    market: torch.Tensor
    present: torch.Tensor
    latest_market_date: tuple[date | None, ...]
    intraday_return: torch.Tensor | None


class MarketHistory:
    def __init__(self, history: Path, update: Path, *, before: date | None = None):
        self.stock_ids = _stock_ids(history)
        self.series = []
        trading_dates = set()
        for stock in self.stock_ids:
            rows = _read_stock(stock, history, update, before)
            days = sorted(rows)
            values = torch.tensor([rows[day] for day in days], dtype=torch.float64).reshape(-1, len(MARKET_COLUMNS))
            valid = torch.isfinite(values).all(dim=1) & (values[:, :len(ADJUSTED_OHLC)] > 0).all(dim=1)
            days = [day for day, keep in zip(days, valid.tolist()) if keep]
            self.series.append((days, values[valid]))
            trading_dates.update(days)
        self.dates = tuple(sorted(trading_dates))

    def batch(self, day: date, length: int, *, labels: bool = False) -> DateBatch:
        present = torch.zeros(len(self.stock_ids), dtype=torch.bool)
        market = torch.zeros(len(self.stock_ids), length, len(MARKET_COLUMNS), dtype=torch.float64)
        intraday_return = torch.zeros(len(self.stock_ids), dtype=torch.float32) if labels else None
        latest = []
        for i, (days, values) in enumerate(self.series):
            end = bisect_left(days, day)
            latest.append(days[end - 1] if end else None)
            if end < length or (labels and (end == len(days) or days[end] != day)):
                continue
            present[i] = True
            market[i] = values[end - length:end]
            if intraday_return is not None:
                intraday_return[i] = values[end, CLOSE] / values[end, OPEN] - 1

        # A missing stock has inert zero inputs; the mask determines whether it contributes.
        price = market[:, :, :len(PRICE_COLUMNS)]
        last_close = torch.where(present, market[:, -1, CLOSE], 1)
        price /= last_close[:, None, None]
        other = market[:, :, len(PRICE_COLUMNS):]
        std, mean = torch.std_mean(other, dim=1, correction=0, keepdim=True)
        other -= mean
        other /= torch.where(std > 0, std, 1)
        return DateBatch(market.float(), present, tuple(latest), intraday_return)
