"""Evaluate the reviewed ZIP in isolation, keeping production code/artifacts intact."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(paths):
    result = hashlib.sha256()
    for path in sorted(paths):
        result.update(str(path.relative_to(ROOT)).encode())
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()


def paired_predictions(path_a, path_b):
    a, b = (pd.read_csv(path) for path in (path_a, path_b))
    keys = ["date", "id"]
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    pair = a.merge(b, on=keys, suffixes=("_A", "_B"), validate="one_to_one")
    assert len(pair) == len(a) == len(b)
    np.testing.assert_allclose(pair["actual_return_A"], pair["actual_return_B"], rtol=0, atol=0)
    for name in ("A", "B"):
        pair[f"score_{name}"] = (2 * pair[f"prediction_{name}"] - 1) * pair["actual_return_A"]
    pair["score_delta"] = pair["score_B"] - pair["score_A"]
    return pair


def compare(folder):
    a, b = (json.loads((folder / name / "metrics.json").read_text()) for name in ("A", "B"))
    assert a["train_rows"] == b["train_rows"]
    fold_rows = []
    for fa, fb in zip(a["cross_validation"]["folds"], b["cross_validation"]["folds"], strict=True):
        for key in ("train_start", "train_end", "valid_start", "valid_end", "train_rows"):
            assert fa[key] == fb[key], key
        fold_rows.append({
            "fold": fa["fold"], "start": fa["valid_start"], "end": fa["valid_end"],
            "score_A": fa["validation"]["score"], "score_B": fb["validation"]["score"],
            "delta": fb["validation"]["score"] - fa["validation"]["score"],
            "iterations_A": fa["best_iteration"], "iterations_B": fb["best_iteration"],
        })
    pd.DataFrame(fold_rows).to_csv(folder / "paired_folds.csv", index=False)
    paired = paired_predictions(folder / "A/validation_predictions.csv", folder / "B/validation_predictions.csv")
    daily = paired.groupby("date")[["score_A", "score_B", "score_delta"]].mean()
    daily.to_csv(folder / "paired_validation_daily.csv")
    return {
        "validation_A": a["validation"], "validation_B": b["validation"],
        "validation_score_delta": b["validation"]["score"] - a["validation"]["score"],
        "fold_wins": sum(f["delta"] > 1e-12 for f in fold_rows),
        "fold_ties": sum(abs(f["delta"]) <= 1e-12 for f in fold_rows),
        "fold_losses": sum(f["delta"] < -1e-12 for f in fold_rows),
        "folds": fold_rows, "daily_delta_mean": float(daily["score_delta"].mean()),
        "daily_delta_std": float(daily["score_delta"].std()),
        "changed_predictions": int((paired["prediction_A"] != paired["prediction_B"]).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-end", default="2026-11-06")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "results" / args.train_end
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    archive_path = ROOT / "lightgbm_ab_experiment.zip"
    # Extract only these reviewed, flat files; never execute arbitrary ZIP entries.
    with ZipFile(archive_path) as archive:
        for name in ("features.py", "train.py", "predict.py", "validation.py", "README.md"):
            (source / name).write_bytes(archive.read(name))
    inputs = [archive_path, ROOT / "model/all_down/all_down.csv",
              *ROOT.glob("basic_data/market/*.csv"), *ROOT.glob("data/market/*.csv")]
    initial_hash = fingerprint(inputs)
    config = {
        "cutoff": args.train_end, "input_sha256": initial_hash,
        "primary": {"cv_folds": 5, "valid_days": 20},
        "robustness": {"cv_folds": 10, "valid_days": 20},
        "test_days": 20, "early_stop_days": 20, "min_train_days": 252,
        "decision_rule": "B must improve CV mean score and win a strict majority of all folds "
                         "in both 5-fold and 10-fold evaluations. Test is disclosed after selection.",
        "note": "5-fold validation is a subset of 10-fold validation; these are not independent replications.",
    }
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    results = {}
    for name, count in (("primary", 5), ("robustness", 10)):
        folder = output / name
        folder.mkdir(exist_ok=True)
        for feature_set in ("A", "B"):
            print(f"Running {name}, {feature_set}, {count} folds", flush=True)
            command = [
                sys.executable, str(source / "train.py"), "--feature-set", feature_set,
                "--data-dir", str(ROOT / "data"), "--history-dir", str(ROOT / "basic_data"),
                "--stock-template", str(ROOT / "model/all_down/all_down.csv"),
                "--artifacts-dir", str(folder / feature_set), "--train-end", args.train_end,
                "--cv-folds", str(count), "--valid-days", "20", "--test-days", "20",
                "--early-stop-days", "20", "--min-train-days", "252",
            ]
            with (folder / f"{feature_set}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        results[name] = compare(folder)
        print(json.dumps({key: value for key, value in results[name].items()
                          if key in ("validation_score_delta", "fold_wins", "fold_ties", "fold_losses")}), flush=True)
    results["selection"] = "B" if all(
        result["validation_score_delta"] > 0 and result["fold_wins"] > len(result["folds"]) / 2
        for result in results.values()) else "A"
    # Lock CV selection before disclosing the final test comparison.
    (output / "selection.json").write_text(json.dumps(results, indent=2) + "\n")
    primary = output / "primary"
    test_pair = paired_predictions(primary / "A/test_predictions.csv", primary / "B/test_predictions.csv")
    test_pair.groupby("date")[["score_A", "score_B", "score_delta"]].mean().to_csv(output / "paired_test_daily.csv")
    results["test"] = {
        feature_set: json.loads((primary / feature_set / "metrics.json").read_text())["test"]
        for feature_set in ("A", "B")
    }
    results["configuration"] = config
    if fingerprint(inputs) != initial_hash:
        raise RuntimeError("Source data changed during the experiment; rerun on a fixed snapshot")
    (output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"CV-based selection: {results['selection']}. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
