"""Populate the manager's Phase 1 or session template with GRU directions."""

import csv
import json

import torch
from global_history import GLOBAL_COLUMNS
from market import MarketHistory
from network import GRUSequence
from runtime import Config, ManagerPaths, command_parser, configure_cpu, iso_date


def read_template(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header, *rows = csv.reader(handle)
    if len(header) < 2 or header[0] != "id" or len(set(header)) != len(header):
        raise ValueError("template must start with id followed by unique YYYY-MM-DD dates")
    dates = [iso_date(value) for value in header[1:]]
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("template must have one full row per target stock")
    stock_ids = [row[0] for row in rows]
    if len(set(stock_ids)) != len(stock_ids):
        raise ValueError("template stock IDs must be unique")
    return dates, stock_ids


@torch.inference_mode()
def main():
    args = command_parser(__doc__).parse_args()
    paths = ManagerPaths.read()
    dates, target_ids = read_template(paths.template)
    configure_cpu()
    metadata = json.loads((args.artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    config = Config(**metadata["config"])
    history = MarketHistory(paths.history, paths.update, before=max(dates))
    if metadata["stock_ids"] != list(history.stock_ids):
        raise ValueError("stock IDs in historical info.json differ from the training universe")
    positions = {stock: i for i, stock in enumerate(history.stock_ids)}
    if not set(target_ids) <= positions.keys():
        raise ValueError("template contains stock IDs outside the training universe")
    index = [positions[stock] for stock in target_ids]

    model = GRUSequence(config, len(history.stock_ids))
    model.load_state_dict(torch.load(args.artifacts_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    columns = []
    details = []
    for day in dates:
        batch = history.batch(day, config.window_length)
        scores = model(batch)
        directions = ((scores > 0) & batch.present).int()
        columns.append(directions[index].tolist())
        for stock, i in zip(target_ids, index):
            if batch.present[i]:
                details.append((stock, day, scores[i].item(), batch.latest_market_date[i],
                                *batch.latest_global_dates[i]))
            else:
                details.append((stock, day, "", "", *batch.latest_global_dates[i]))

    detail_path = paths.output.with_name(f"{paths.output.stem}_detail.csv")
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        global_headers = [f"latest_{column.removeprefix('^').lower()}_date" for column in GLOBAL_COLUMNS]
        writer.writerow(("id", "date", "f", "latest_market_date", *global_headers))
        writer.writerows(details)

    with paths.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", *dates))
        writer.writerows(zip(target_ids, *columns))
    print(f"Wrote {paths.output}")


if __name__ == "__main__":
    main()
