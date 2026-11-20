"""Matched-window ablation of US features, with a full-history reference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model/lightgbm_baseline"
sys.path.insert(0, str(MODEL))
from features import DATA_DIR, HISTORY_DIR
from us_features import US_DATA_DIR, SYMBOLS, US_FEATURES


def digest():
    paths = [*DATA_DIR.glob("market/*.csv"), *HISTORY_DIR.glob("market/*.csv"),
             *[US_DATA_DIR / "prices" / f"{symbol}.csv" for symbol in SYMBOLS],
             MODEL / "train.py", MODEL / "features.py", MODEL / "us_features.py", MODEL / "validation.py",
             ROOT / "model/all_down/all_down.csv"]
    result = hashlib.sha256()
    for path in sorted(paths):
        result.update(str(path.resolve()).encode())
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()


def main():
    output = Path(__file__).resolve().parent / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True)
    best = MODEL / "artifacts/best_params.json"
    params = json.loads(best.read_text()).get("params", {}) if best.exists() else {}
    (output / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    before = digest()
    configuration = {
        "input_sha256": before, "params": params, "us_symbols": SYMBOLS, "us_features": US_FEATURES,
        "cv_folds": 5, "robustness_cv_folds": 10, "valid_days": 20, "early_stop_days": 20, "test_days": 20,
        "selection": "Assess US addition on identical covered stock-dates; use validation only. Test is diagnostic.",
        "publication_assumption": "US NY session close followed by 08:10 Taipei availability; forecast at 08:25",
    }
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    cases = {
        "baseline_full": ["--feature-set", "baseline"],
        "baseline_matched": ["--feature-set", "baseline", "--match-us-coverage"],
        "us": ["--feature-set", "us"],
    }
    metrics = {}
    for name, flags in cases.items():
        print(f"Validation: {name}", flush=True)
        command = [sys.executable, "-u", str(MODEL / "train.py"),
                   "--data-dir", str(DATA_DIR), "--history-dir", str(HISTORY_DIR),
                   "--us-data-dir", str(US_DATA_DIR), "--params", str(output / "params.json"),
                   "--artifacts-dir", str(output / name), *flags]
        with (output / f"{name}.log").open("w") as log:
            subprocess.run([*command, "--validation-only"], stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics[name] = json.loads((output / name / "metrics.json").read_text())
        if digest() != before:
            raise RuntimeError("Inputs changed during experiment; rerun on stable data")
    for name in ("baseline_matched", "us"):
        frame = pd.read_csv(output / name / "validation_predictions.csv", dtype={"id": str})
        if name == "baseline_matched":
            reference = frame
        else:
            pd.testing.assert_frame_equal(reference[["date", "id", "fold", "actual_return"]],
                                          frame[["date", "id", "fold", "actual_return"]])
    a, b = metrics["baseline_matched"], metrics["us"]
    folds = [{"fold": fa["fold"], "A": fa["validation"]["score"], "B": fb["validation"]["score"],
              "delta": fb["validation"]["score"] - fa["validation"]["score"]}
             for fa, fb in zip(a["cross_validation"]["folds"], b["cross_validation"]["folds"], strict=True)]
    selection = {
        "validation_score_delta": b["validation"]["score"] - a["validation"]["score"],
        "fold_comparison": folds,
        "fold_wins": sum(fold["delta"] > 1e-12 for fold in folds),
        "fold_losses": sum(fold["delta"] < -1e-12 for fold in folds),
        "production_changed": False,
    }
    robustness = {}
    for name in ("baseline_matched", "us"):
        print(f"Longer validation (10 folds): {name}", flush=True)
        with (output / f"robustness_{name}.log").open("w") as log:
            subprocess.run([sys.executable, "-u", str(MODEL / "train.py"),
                            "--data-dir", str(DATA_DIR), "--history-dir", str(HISTORY_DIR),
                            "--us-data-dir", str(US_DATA_DIR), "--params", str(output / "params.json"),
                            "--artifacts-dir", str(output / f"robustness_{name}"),
                            "--cv-folds", "10", "--validation-only", *cases[name]],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        robustness[name] = json.loads((output / f"robustness_{name}" / "metrics.json").read_text())
    a_long, b_long = [robustness[name] for name in ("baseline_matched", "us")]
    selection["robustness_score_delta"] = b_long["validation"]["score"] - a_long["validation"]["score"]
    selection["robustness_fold_wins"] = sum(
        fb["validation"]["score"] > fa["validation"]["score"]
        for fa, fb in zip(a_long["cross_validation"]["folds"], b_long["cross_validation"]["folds"], strict=True))
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    # Refit with the fixed settings and only now disclose final test diagnostics.
    for name, flags in cases.items():
        print(f"Final fit and test: {name}", flush=True)
        with (output / f"{name}_final.log").open("w") as log:
            subprocess.run([sys.executable, "-u", str(MODEL / "train.py"),
                            "--data-dir", str(DATA_DIR), "--history-dir", str(HISTORY_DIR),
                            "--us-data-dir", str(US_DATA_DIR), "--params", str(output / "params.json"),
                            "--artifacts-dir", str(output / name), *flags],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        final = json.loads((output / name / "metrics.json").read_text())
        assert final["validation"] == metrics[name]["validation"]
        metrics[name] = final
    if digest() != before:
        raise RuntimeError("Inputs changed during experiment; results invalid")
    comparison = {"configuration": configuration, "selection": selection, "models": metrics, "robustness": robustness}
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    pd.DataFrame([{"model": name, "cv_score": value["validation"]["score"],
                   "cv_accuracy": value["validation"]["accuracy"], "test_score": value["test"]["score"],
                   "test_accuracy": value["test"]["accuracy"], "best_iteration": value["best_iteration"]}
                  for name, value in metrics.items()]).to_csv(output / "comparison.csv", index=False)
    print(json.dumps(selection, indent=2), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
