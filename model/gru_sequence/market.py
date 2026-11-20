"""Market history and time-causal cross-sections on a stable universe stock axis."""

import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch
from global_history import GLOBAL_COLUMNS, US_COLUMNS, GlobalHistory
from runtime import iso_date

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

    sequence: torch.Tensor
    today: torch.Tensor
    present: torch.Tensor
    latest_market_date: tuple[date | None, ...]
    latest_global_dates: tuple[tuple[date | None, ...], ...]
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
        self.globals = GlobalHistory(history, update, through=before)
        global_rows = []
        self.global_sources = []
        previous = None
        for day in self.dates:
            values, sources = self.globals.at_cutoff(day, previous)
            global_rows.append(values)
            self.global_sources.append(sources)
            previous = day
        self.global_values = torch.tensor(global_rows, dtype=torch.float64).reshape(-1, len(GLOBAL_COLUMNS))
        self.tx_observed = torch.tensor([sources[-1] is not None for sources in self.global_sources],
                                        dtype=torch.bool)
        self.calendar_positions = {day: i for i, day in enumerate(self.dates)}
        self.global_indices = [torch.tensor([self.calendar_positions[day] for day in days], dtype=torch.long)
                               for days, _ in self.series]

    def batch(self, day: date, length: int, *, labels: bool = False,
              market_not_before: date | None = None) -> DateBatch:
        present = torch.zeros(len(self.stock_ids), dtype=torch.bool)
        market = torch.zeros(len(self.stock_ids), length, len(MARKET_COLUMNS), dtype=torch.float64)
        global_window = torch.zeros(len(self.stock_ids), length, len(GLOBAL_COLUMNS), dtype=torch.float64)
        tx_observed = torch.zeros(len(self.stock_ids), length, dtype=torch.bool)
        intraday_return = torch.zeros(len(self.stock_ids), dtype=torch.float32) if labels else None
        # A requested prediction date need not already have its own market rows.
        end = bisect_left(self.dates, day)
        previous = self.dates[end - 1] if end else None
        today, us_sources = self.globals.at_cutoff(day, previous, US_COLUMNS)
        latest = []
        global_sources = []
        for i, (days, values) in enumerate(self.series):
            end = bisect_left(days, day)
            last_row = days[end - 1] if end else None
            latest.append(last_row)
            stale = market_not_before is not None and (last_row is None or last_row < market_not_before)
            if stale or end < length or (labels and (end == len(days) or days[end] != day)):
                global_sources.append((None,) * len(GLOBAL_COLUMNS))
                continue
            present[i] = True
            market[i] = values[end - length:end]
            positions = self.global_indices[i][end - length:end]
            global_window[i] = self.global_values[positions]
            tx_observed[i] = self.tx_observed[positions]
            # As-of source dates never decrease; today's US levels supersede
            # the window dates. Non-US inputs appear only in the window.
            window_sources = self.global_sources[self.calendar_positions[days[end - 1]]]
            global_sources.append((*us_sources, *window_sources[len(US_COLUMNS):]))
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
        # For TX_OI, leading gaps are neutral and do not enter the window statistics.
        tx = global_window[:, :, -1]
        count = tx_observed.sum(dim=1, keepdim=True).clamp_min(1)
        tx -= tx.sum(dim=1, keepdim=True) / count
        tx.masked_fill_(~tx_observed, 0)
        std = (tx.square().sum(dim=1, keepdim=True) / count).sqrt()
        tx /= torch.where(std > 0, std, 1)

        sequence = torch.cat((market, global_window), dim=-1).float()
        return DateBatch(sequence, torch.tensor(today, dtype=torch.float32), present,
                         tuple(latest), tuple(global_sources), intraday_return)
