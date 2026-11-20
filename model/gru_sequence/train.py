"""Fit one fixed configuration on all available universe stock labels."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from market import MarketHistory
from network import GRUSequence
from runtime import MODEL_DIR, Config, ManagerPaths, command_parser, configure_cpu, deployment_seeds, package_versions


class TrainingBatches:
    """Reuse labeled batches across fits and epochs within a memory budget."""

    def __init__(self, history: MarketHistory, config: Config):
        self.history = history
        self.length = config.window_length
        self.cached = {}
        self.remaining_bytes = 256 * 1024**2

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


def fit_fixed(config: Config, num_stocks: int, make_batches):
    model, optimizer = initialize_training(config, num_stocks)
    for _ in range(config.epochs):
        train_epoch(model, optimizer, make_batches())
    return model


def fit_ensemble(config, seeds, num_stocks, make_batches):
    return [fit_fixed(replace(config, seed=seed), num_stocks, make_batches) for seed in seeds]


def save_ensemble(directory, models, seeds, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory, prefix=".fit-") as tmp:
        staging = Path(tmp)
        if len(seeds) == 1:
            torch.save(models[0].state_dict(), staging / "model.pt")
            (staging / "model.pt").replace(directory / "model.pt")
            files = ["model.pt"]
        else:
            # Keep weight sets immutable so readers with older metadata cannot mix runs.
            name = "ensemble-" + staging.name.removeprefix(".fit-")
            weights = staging / name
            weights.mkdir()
            for seed, model in zip(seeds, models):
                torch.save(model.state_dict(), weights / f"model-{seed}.pt")
            weights.rename(directory / name)
            files = [f"{name}/model-{seed}.pt" for seed in seeds]
        payload = {**metadata, "deployment_seeds": seeds, "model_files": files}
        manifest = staging / "metadata.json"
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest.replace(directory / "metadata.json")


def main():
    parser = command_parser(__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.json")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = Config.from_settings(settings)
    seeds = deployment_seeds(settings, config)
    configure_cpu()
    paths = ManagerPaths.read()
    history = MarketHistory(paths.history, paths.update)
    dates = sorted(history.labeled_dates(config.window_length))
    if not dates:
        raise ValueError("no training labels with a complete market window")
    batches = TrainingBatches(history, config)
    models = fit_ensemble(config, seeds, len(history.stock_ids), lambda: batches.for_dates(dates))
    metadata = {
        "config": asdict(config),
        "stock_ids": history.stock_ids,
        "first_training_label_date": dates[0].isoformat(),
        "last_training_label_date": dates[-1].isoformat(),
        "torch_version": torch.__version__,
        "package_versions": package_versions(),
    }
    save_ensemble(args.artifacts_dir, models, seeds, metadata)


if __name__ == "__main__":
    main()
