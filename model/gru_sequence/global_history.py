"""Global observations at Taiwan's information cutoff."""

import csv
import math
from bisect import bisect_right
from datetime import date
from pathlib import Path

import torch
from features import zscore_window_
from runtime import iso_date

US_COLUMNS = ("^GSPC", "^IXIC", "^VIX")
INDEX_COLUMNS = (*US_COLUMNS, "^TWII", "^KS11")
GLOBAL_COLUMNS = (*INDEX_COLUMNS, "TX_OI")
TX_OI = GLOBAL_COLUMNS.index("TX_OI")


class GlobalHistory:
    def __init__(self, history: Path, update: Path, *, through: date | None,
                 prediction_date: date | None):
        rows = {}
        found = False
        for directory in (history, update):
            path = directory / "global" / "global_data.csv"
            if not path.exists():
                continue
            found = True
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                if len(fields) != len(GLOBAL_COLUMNS) + 1 or set(fields) != {"date", *GLOBAL_COLUMNS}:
                    raise ValueError(f"{path}: unexpected global columns")
                for row in reader:
                    day = iso_date(row["date"])
                    if through is None or day <= through:
                        rows[day] = row
        if not found:
            raise FileNotFoundError("global/global_data.csv not found in historical or update data")

        self.series = {}
        ordered_days = sorted(rows)
        for column in GLOBAL_COLUMNS:
            dates, values = [], []
            is_index = column in INDEX_COLUMNS
            eligible_days = ordered_days
            if column not in US_COLUMNS and ordered_days and ordered_days[-1] == prediction_date:
                eligible_days = ordered_days[:-1]
            for day in eligible_days:
                cell = rows[day][column]
                value = float(cell) if cell and cell.strip() else math.nan
                if not math.isfinite(value) or (is_index and value <= 0):
                    continue
                dates.append(day)
                values.append(math.log(value) if is_index else value)
            self.series[column] = (dates, values)

    def source_date_range(self):
        spans = [(days[0], days[-1]) for days, _ in self.series.values() if days]
        if not spans:
            return None
        return [min(start for start, _ in spans).isoformat(), max(end for _, end in spans).isoformat()]

    def _latest(self, day: date, column: str) -> tuple[float, date | None]:
        days, values = self.series[column]
        end = bisect_right(days, day)
        return (values[end - 1], days[end - 1]) if end else (0.0, None)

    def align(self, trading_dates):
        """Align inputs through each completed Taiwan market row and retain real source dates."""
        rows, sources = [], []
        previous = None
        for day in trading_dates:
            row, provenance = self._through(day, previous, GLOBAL_COLUMNS)
            rows.append(row)
            sources.append(provenance)
            previous = day
        values = torch.tensor(rows, dtype=torch.float64).reshape(-1, len(GLOBAL_COLUMNS))
        oi_available = torch.tensor([row[TX_OI] is not None for row in sources], dtype=torch.bool)
        return values, sources, oi_available

    @staticmethod
    def normalize_window_(values, oi_available):
        # TX_OI is a count (not an index). Carry after the first observation;
        # neutral placeholders before that date never enter the window statistics.
        zscore_window_(values[:, :, TX_OI], oi_available)

    def at_cutoff(self, day: date, previous: date | None):
        """Use only US series, including observations dated on the prediction date."""
        return self._through(day, previous, US_COLUMNS)

    def _through(self, day: date, previous: date | None, columns: tuple[str, ...]):
        values, sources = [], []
        for column in columns:
            value, source = self._latest(day, column)
            if column in INDEX_COLUMNS:
                old, old_source = self._latest(previous, column) if previous is not None else (0.0, None)
                value = value - old if source is not None and old_source is not None else 0.0
            values.append(value)
            sources.append(source)
        return values, tuple(sources)
