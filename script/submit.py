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
    if phase not in (1, 2):
        raise ValueError("Phase must be 1 or 2")

    data = parse_csv(path)

    if phase == 2:
        # Phase 2: 50 stocks, 1 trading date
        if len(data["stockIds"]) != 50:
            raise ValueError("Phase 2 requires 50 stocks")

        if len(data["dates"]) != 1:
            raise ValueError("Phase 2 requires exactly one date")

        # Convert (50, 1) into (50,)
        predictions = [
            row[0] for row in data["predictions"]
        ]

        if any(
            type(x) is not int or x not in (0, 1)
            for x in predictions
        ):
            raise ValueError(
                "Phase 2 predictions must be 0 or 1"
            )

        data["predictions"] = predictions

    payload = {
        "phase": phase,
        "fileName": path.name,
        "payload": data,
        "note": "",
    }

    # Preview
    print(json.dumps({
        "phase": phase,
        "fileName": path.name,
        "stockIds": len(data["stockIds"]),
        "dates": data["dates"],
        "predictionShape": (
            [len(data["predictions"]), len(data["dates"])]
            if phase == 1
            else [len(data["predictions"])]
        ),
        "sample": data["predictions"][:5],
    }, indent=2, ensure_ascii=False))

    if dry_run:
        print("Dry run: no submission sent")
        return

    # Submit exactly once
    with login() as session:
        response = session.post(
            f"{BASE_URL}/api/submissions",
            json=payload,
            headers={
                "Referer": f"{BASE_URL}/",
            },
            timeout=30,
        )

        print("Status:", response.status_code)

        try:
            print(
                "Response:",
                json.dumps(
                    response.json(),
                    indent=2,
                    ensure_ascii=False,
                ),
            )
        except ValueError:
            print("Response:", response.text)

        response.raise_for_status()


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
