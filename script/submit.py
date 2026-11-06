import argparse
import csv
import json
from pathlib import Path

from auth import BASE_URL, login


def parse_csv(path: Path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.reader(f)
        header = next(reader)

        if header[0].strip() != "id":
            raise ValueError("First column must be 'id'")

        dates = [x.strip() for x in header[1:]]
        stock_ids = []
        predictions = []

        for row in reader:
            if not row:
                continue

            if len(row) != len(header):
                raise ValueError(
                    f"Invalid column count: {row[0]}"
                )

            stock_ids.append(row[0].strip())

            values = []
            for cell in row[1:]:
                cell = cell.strip()

                if cell == "":
                    values.append(None)
                elif cell in ("0", "1"):
                    values.append(int(cell))
                else:
                    raise ValueError(
                        f"Invalid prediction: {cell}"
                    )

            predictions.append(values)

    if not stock_ids or not dates:
        raise ValueError("Empty submission")

    if len(set(stock_ids)) != len(stock_ids):
        raise ValueError("Duplicate stock IDs")

    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate dates")

    return {
        "stockIds": stock_ids,
        "dates": dates,
        "predictions": predictions,
    }


def submit(path: Path, phase: int, dry_run=True):
    payload = {
        "phase": phase,
        "fileName": path.name,
        "payload": parse_csv(path),
        "note": "",
    }

    if dry_run:
        print(json.dumps({
            "phase": phase,
            "stockIds": len(payload["payload"]["stockIds"]),
            "dates": len(payload["payload"]["dates"]),
            "firstRow": payload["payload"]["predictions"][0],
        }, indent=2))
        print("Dry run: no submission sent")
        return

    with login() as session:
        response = session.post(
            f"{BASE_URL}/api/submissions",
            json=payload,
            headers={
                "Referer": f"{BASE_URL}/",
            },
            timeout=30,
        )

        response.raise_for_status()
        print("Response:", response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    submit(
        args.file,
        args.phase,
        dry_run=not args.submit,
    )
