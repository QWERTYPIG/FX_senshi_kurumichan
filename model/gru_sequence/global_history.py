"""Global observations at Taiwan's information cutoff."""

import csv
import math
from bisect import bisect_left, bisect_right
from datetime import date
from pathlib import Path

from runtime import iso_date

US_COLUMNS = ("^GSPC", "^IXIC", "^VIX")
GLOBAL_COLUMNS = (*US_COLUMNS, "^TWII", "^KS11", "TX_OI")
INDEX_COLUMNS = GLOBAL_COLUMNS[:-1]


class GlobalHistory:
    def __init__(self, history: Path, update: Path, *, through: date | None):
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
            for day in ordered_days:
                if day == through and column not in US_COLUMNS:
                    break
                cell = rows[day][column]
                value = float(cell) if cell and cell.strip() else math.nan
                if not math.isfinite(value) or (column in INDEX_COLUMNS and value <= 0):
                    continue
                dates.append(day)
                values.append(math.log(value) if column in INDEX_COLUMNS else value)
            self.series[column] = (dates, values)

    def _asof(self, day: date, column: str) -> tuple[float, date | None]:
        days, values = self.series[column]
        end = (bisect_right if column in US_COLUMNS else bisect_left)(days, day)
        return (values[end - 1], days[end - 1]) if end else (0.0, None)

    def at_cutoff(
        self, day: date, previous: date | None, columns: tuple[str, ...] = GLOBAL_COLUMNS,
    ) -> tuple[list[float], tuple[date | None, ...]]:
        values, sources = [], []
        for column in columns:
            value, source = self._asof(day, column)
            if column in INDEX_COLUMNS:
                old, old_source = self._asof(previous, column) if previous is not None else (0.0, None)
                value = value - old if source is not None and old_source is not None else 0.0
            values.append(value)
            sources.append(source)
        return values, tuple(sources)
