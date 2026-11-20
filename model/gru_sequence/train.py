"""Fit one fixed configuration on all available universe stock labels."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from market import MarketHistory
from network import GRUSequence
from runtime import MODEL_DIR, Config, ManagerPaths, command_parser, configure_cpu


def train_epoch(model, optimizer, history: MarketHistory, dates, config: Config):
    trained_dates = []
    model.train()
    for day in dates:
        batch = history.batch(day, config.window_length, labels=True)
        if not batch.present.any():
            continue
        optimizer.zero_grad()
        f = model(batch)[batch.present]
        intraday_return = batch.intraday_return[batch.present]
        loss = -(f.tanh() * intraday_return).mean()
        loss.backward()
        optimizer.step()
        trained_dates.append(day)
    if not trained_dates:
        raise ValueError("no training labels with a complete market window")
    return min(trained_dates), max(trained_dates)


def main():
    parser = command_parser(__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.json")
    args = parser.parse_args()
    config = Config.read(args.config)
    configure_cpu()
    torch.manual_seed(config.seed)

    paths = ManagerPaths.read()
    history = MarketHistory(paths.history, paths.update)
    model = GRUSequence(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        first_label, last_label = train_epoch(model, optimizer, history, history.dates, config)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.artifacts_dir / "model.pt")
    metadata = {
        "config": asdict(config),
        "stock_ids": history.stock_ids,
        "first_training_label_date": first_label.isoformat(),
        "last_training_label_date": last_label.isoformat(),
        "torch_version": torch.__version__,
    }
    (args.artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
