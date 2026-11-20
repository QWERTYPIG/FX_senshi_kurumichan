"""Populate the manager's Phase 1 or session template with GRU directions."""

import csv
import logging

import torch
from global_history import GLOBAL_COLUMNS
from market import MarketHistory
from network import GRUSequence, mean_f
from runtime import CHECKPOINT_NAME, Config, ManagerPaths, command_parser, configure_cpu, iso_date


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
    checkpoint = args.artifacts_dir / CHECKPOINT_NAME
    try:
        snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise SystemExit(f"No checkpoint at {checkpoint}; run train.py to create it") from None
    metadata = snapshot["metadata"]
    last_label = iso_date(metadata["last_training_label_date"])
    for day in dates:
        if day <= last_label:
            raise SystemExit(f"prediction date {day} must be after the last training label date {last_label}")

    configure_cpu()
    config = Config(**metadata["config"])
    history = MarketHistory(paths.history, paths.update, before=max(dates))
    if metadata["stock_ids"] != history.stock_ids:
        raise ValueError("stock IDs in historical info.json differ from the training universe")
    index = history.stock_positions(target_ids, source="template")

    models = []
    for seed in metadata["deployment_seeds"]:
        model = GRUSequence(config, len(history.stock_ids))
        model.load_state_dict(snapshot["states"][seed])
        models.append(model.eval())

    columns = []
    details = []
    for day in dates:
        batch = history.batch(day, config.window_length)
        if batch.all_market_stale:
            raise SystemExit(f"{day}: all universe stocks have stale market data; no predictions written")
        for position in batch.stale_market_positions:
            logging.warning("%s: stale market data for stock %s (latest valid row: %s); treating as absent",
                            day, history.stock_ids[position], batch.latest_market_date[position] or "none")
        scores = mean_f(models, batch)
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
