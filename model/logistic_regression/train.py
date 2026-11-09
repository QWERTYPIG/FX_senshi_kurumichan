"""Tune weighted logistic regression chronologically and refit before the cutoff."""
import argparse
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from common import ROOT, DATA_DIR, MODEL_DIR, read_template, encode
from features import build_data, fit, predict, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--stock-template', type=Path, default=ROOT / 'model/all_down/all_down.csv')
    parser.add_argument('--artifacts-dir', type=Path, default=MODEL_DIR / 'artifacts')
    parser.add_argument('--train-end', default='2026-05-31', help='Inclusive label cutoff; default preserves Phase 1 holdout')
    parser.add_argument('--validation-start', default='2026-01-01')
    parser.add_argument('--penalties', type=float, nargs='+', default=[1e-6,1e-5,1e-4,.001,.01,.1,1,10,100])
    args = parser.parse_args()
    if any(not np.isfinite(p) or p <= 0 for p in args.penalties):
        parser.error('Penalties must be finite and positive')
    ids = read_template(args.stock_template).id.tolist()
    cutoff = pd.Timestamp(args.train_end)
    template = pd.DataFrame({'id':ids, cutoff.strftime('%Y-%m-%d'):0})
    panel = build_data(args.data_dir, template)
    features = encode(panel, ids)
    train = panel[panel.target.notna() & panel.stock_source_date.notna() & (panel.date <= cutoff)]
    early = train[train.date < pd.Timestamp(args.validation_start)]
    val = train[train.date >= pd.Timestamp(args.validation_start)]
    if early.empty or val.empty:
        raise ValueError('Need labeled history both before and after validation-start')
    trials = []
    for penalty in sorted(set(args.penalties)):
        model = fit(early, features, penalty)
        trial = dict(penalty=penalty, **metrics(val, predict(model,val)[0]))
        trials.append(trial)
        print(json.dumps(trial), flush=True)
    best = max(trials, key=lambda t:(t['score'],t['penalty']))
    model = fit(train, features, best['penalty'])
    metadata = dict(feature_version=1, stock_ids=ids, trained_through=str(train.date.max().date()), selected_penalty=best['penalty'])
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with (args.artifacts_dir/'model.pkl').open('wb') as f:
        pickle.dump(model,f)
    for name, value in [('metadata.json',metadata),('metrics.json',dict(validation=best,regularization_trials=trials,training_rows=len(train)))]:
        (args.artifacts_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    pd.DataFrame(trials).to_csv(args.artifacts_dir/'regularization_search.csv',index=False)
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
