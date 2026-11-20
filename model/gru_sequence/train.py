"""Fit one fixed configuration on all available universe stock labels."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from market import MarketHistory
from network import GRUSequence
from runtime import MODEL_DIR, Config, ManagerPaths, command_parser, configure_cpu


def training_batches(history: MarketHistory, dates, config: Config):
    return (history.batch(day, config.window_length, labels=True) for day in dates)


def initialize_training(config: Config, num_stocks: int):
    torch.manual_seed(config.seed)
    model = GRUSequence(config, num_stocks)
    return model, torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def smooth_loss(model, batch):
    f = model(batch)[batch.present]
    return -(f.tanh() * batch.intraday_return[batch.present]).mean()


def train_epoch(model, optimizer, batches):
    model.train()
    for batch in batches:
        optimizer.zero_grad()
        smooth_loss(model, batch).backward()
        optimizer.step()


def fit_fixed(config: Config, num_stocks: int, make_batches):
    model, optimizer = initialize_training(config, num_stocks)
    for _ in range(config.epochs):
        train_epoch(model, optimizer, make_batches())
    return model


def main():
    parser = command_parser(__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.json")
    args = parser.parse_args()
    config = Config.read(args.config)
    configure_cpu()
    paths = ManagerPaths.read()
    history = MarketHistory(paths.history, paths.update)
    dates = sorted(history.labeled_dates(config.window_length))
    if not dates:
        raise ValueError("no training labels with a complete market window")
    model = fit_fixed(config, len(history.stock_ids), lambda: training_batches(history, dates, config))

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.artifacts_dir / "model.pt")
    metadata = {
        "config": asdict(config),
        "stock_ids": history.stock_ids,
        "first_training_label_date": dates[0].isoformat(),
        "last_training_label_date": dates[-1].isoformat(),
        "torch_version": torch.__version__,
    }
    (args.artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
