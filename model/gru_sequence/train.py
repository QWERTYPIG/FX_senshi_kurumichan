"""Fit one fixed configuration on all available universe stock labels."""

import json
import logging
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from market import MarketHistory
from network import GRUSequence
from runtime import (
    CHECKPOINT_NAME, MODEL_DIR, Config, ManagerPaths, command_parser, configure_cpu, deployment_seeds, run_metadata,
)


BATCH_CACHE_BYTES = 256 * 1024**2


class TrainingBatches:
    """Reuse labeled batches across fits and epochs within a memory budget."""

    def __init__(self, history: MarketHistory, config: Config):
        self.history = history
        self.length = config.window_length
        self.cached = {}
        self.remaining_bytes = BATCH_CACHE_BYTES

    @property
    def num_stocks(self):
        return len(self.history.stock_ids)

    def for_dates(self, dates):
        for day in dates:
            batch = self.cached.get(day)
            if batch is None:
                batch = self.history.batch(day, self.length, labels=True)
                # The sequence dominates memory; include the other tensors and provenance pointers.
                size = batch.sequence.nbytes + batch.today.nbytes + batch.present.nbytes + batch.intraday_return.nbytes
                size += 1024 + 128 * len(batch.present)
                if size <= self.remaining_bytes:
                    self.cached[day] = batch
                    self.remaining_bytes -= size
            yield batch


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


def fit_fixed(config: Config, batches: TrainingBatches, dates):
    model, optimizer = initialize_training(config, batches.num_stocks)
    for _ in range(config.epochs):
        train_epoch(model, optimizer, batches.for_dates(dates))
    return model


def fit_ensemble(config, seeds, batches, dates):
    return [fit_fixed(replace(config, seed=seed), batches, dates) for seed in seeds]


def save_ensemble(directory, models, seeds, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory, prefix=".fit-") as tmp:
        staging = Path(tmp)
        metadata = {**metadata, "deployment_seeds": seeds}
        states = {seed: model.state_dict() for seed, model in zip(seeds, models, strict=True)}
        checkpoint = staging / CHECKPOINT_NAME
        torch.save({"metadata": metadata, "states": states}, checkpoint)
        manifest = staging / "metadata.json"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # One replace publishes a complete fit. A concurrent reader with the
        # previous file already open keeps that snapshot, including its metadata.
        checkpoint.replace(directory / CHECKPOINT_NAME)
        manifest.replace(directory / "metadata.json")


def main():
    parser = command_parser(__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.json")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = Config.from_settings(settings)
    provisional = args.config.resolve() == MODEL_DIR / "config.json"
    seeds = deployment_seeds(settings, config, allow_standalone=not provisional)
    if provisional:
        logging.warning("Using provisional GRU defaults; walk-forward evidence for adoption is pending (#16)")
    configure_cpu()
    paths = ManagerPaths.read()
    history = MarketHistory(paths.history, paths.update)
    dates = sorted(history.labeled_dates(config.window_length))
    if not dates:
        raise ValueError("no training labels with a complete market window")
    epoch_selection = settings.get("epoch_selection")
    epoch_source = "fixed"
    if epoch_selection is not None:
        epoch_source = "validation" if config.epochs == epoch_selection["refit_epochs"] else "fixed_override"
    metadata = {
        **run_metadata(config, history),
        "stock_ids": history.stock_ids,
        "deployment_mode": settings.get("deployment_mode", "ensemble"),
        "configuration_status": "provisional_default" if provisional else "supplied",
        "first_training_label_date": dates[0].isoformat(),
        "last_training_label_date": dates[-1].isoformat(),
        "epoch_source": epoch_source,
        "epoch_selection": epoch_selection,
    }
    batches = TrainingBatches(history, config)
    models = fit_ensemble(config, seeds, batches, dates)
    save_ensemble(args.artifacts_dir, models, seeds, metadata)


if __name__ == "__main__":
    main()
