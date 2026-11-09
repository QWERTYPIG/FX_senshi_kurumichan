"""Leakage-conscious Phase 1 weighted logistic regression baseline (NumPy/SciPy)."""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def load_history(data_dir, update_dir, relative_path, end_date):
    paths = [data_dir / relative_path]
    if update_dir is not None:
        paths.append(update_dir / relative_path)
    pieces = []
    for path in paths:
        piece = pd.read_csv(path, parse_dates=['date'])
        if piece.date.duplicated().any():
            raise ValueError(f'Duplicate dates: {path}')
        pieces.append(piece)
    # Updated records take precedence on overlaps. Later placeholder rows are unused.
    combined = pd.concat(pieces, ignore_index=True).drop_duplicates('date', keep='last')
    return combined[combined.date <= end_date].set_index('date').sort_index()


def build_data(data_dir, template, update_dir=None):
    dates = pd.to_datetime(template.columns[1:])
    global_raw = load_history(data_dir, update_dir, Path('global/global_data.csv'), dates.max())
    for c in global_raw:
        if c != 'TX_OI':
            global_raw[c] = global_raw[c].where(global_raw[c] > 0)
    gf = pd.DataFrame(index=global_raw.index)
    for c in global_raw:
        gf['global_' + c + '_change'] = global_raw[c].pct_change(fill_method=None)
    gf['global_vix'] = global_raw['^VIX']
    # Exact matches excluded: all global observations must predate prediction day.
    frames = []
    for sid in template.iloc[:, 0].astype(str):
        raw = load_history(data_dir, update_dir, Path('market') / f'{sid}.csv', dates.max())
        # Reject unavailable-price placeholder rows as feature sources and labels.
        raw = raw[(raw['etl:adj_close'] > 0) & (raw['etl:adj_open'] > 0)].copy()
        if raw.empty:
            raise ValueError(f'No valid price history: {sid}')
        if raw.index.has_duplicates:
            raise ValueError(f'Duplicate dates: {sid}')
        close, op = raw['etl:adj_close'], raw['etl:adj_open']
        f = pd.DataFrame(index=raw.index)
        ret = close.pct_change(fill_method=None)
        f['intraday'] = close / op - 1
        f['range'] = (raw['etl:adj_high'] - raw['etl:adj_low']) / close
        for n in [1, 5, 20]:
            f[f'return_{n}'] = close.pct_change(n, fill_method=None)
        for n in [5, 20]:
            f[f'volatility_{n}'] = ret.rolling(n).std()
            f[f'momentum_{n}'] = close / close.rolling(n).mean() - 1
        volume = raw['price:成交股數']
        f['volume_ratio'] = volume / volume.rolling(20).mean().replace(0, np.nan)
        f['log_volume'] = np.log1p(volume)
        for c in raw:
            if c.startswith('institutional_'):
                f[c] = raw[c] / volume.replace(0, np.nan)
        f['foreign_holding'] = raw['foreign_investors_shareholding:全體外資及陸資持股比率'] / 100
        f['rsi'] = raw['RSI'] / 100
        for c in ['SMA', 'BBANDS_upper', 'BBANDS_mid', 'BBANDS_lower']:
            f[c] = raw[c] / close - 1
        for c in ['MACD', 'ATR']:
            f[c] = raw[c] / close
        # As-of join handles missing future dates without inventing market observations.
        history = f.reset_index().rename(columns={'date': 'stock_source_date'})
        target_dates = raw.index.union(dates).sort_values()
        joined = pd.merge_asof(pd.DataFrame({'date': target_dates}), history,
                               left_on='date', right_on='stock_source_date', direction='backward', allow_exact_matches=False)
        joined = pd.merge_asof(joined, gf.reset_index().rename(columns={'date': 'global_source_date'}),
                               left_on='date', right_on='global_source_date', direction='backward', allow_exact_matches=False)
        joined['stock_age_days'] = (joined.date - joined.stock_source_date).dt.days
        joined['global_age_days'] = (joined.date - joined.global_source_date).dt.days
        joined['weekday'] = joined.date.dt.dayofweek
        joined['stock_id'] = sid
        joined['target'] = joined.date.map(close / op - 1)
        frames.append(joined)
    panel = pd.concat(frames, ignore_index=True)
    return panel.replace([np.inf, -np.inf], np.nan)


def fit(frame, features, penalty=0.01):
    x = frame[features].to_numpy(float)
    median = np.nanmedian(x, axis=0)
    median = np.nan_to_num(median)
    x = np.where(np.isnan(x), median, x)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-10] = 1
    x = np.column_stack([np.ones(len(x)), np.clip((x - mean) / std, -10, 10)])
    r = frame.target.to_numpy()
    y = (r > 0).astype(float)
    weights = np.abs(r) / np.mean(np.abs(r))
    def objective(beta):
        z = x @ beta
        loss = np.mean(weights * (np.logaddexp(0, z) - y * z)) + penalty / 2 * np.sum(beta[1:] ** 2)
        grad = x.T @ (weights * (expit(z) - y)) / len(x)
        grad[1:] += penalty * beta[1:]
        return loss, grad
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    # Ridge predicts E[|return| | features] on its original fractional-return scale.
    reg = np.eye(x.shape[1]) * len(x) * penalty
    reg[0, 0] = 0
    magnitude_beta = np.linalg.solve(x.T @ x + reg, x.T @ np.abs(r))
    return dict(features=features, median=median, mean=mean, std=std, beta=result.x,
                magnitude_beta=magnitude_beta, magnitude_floor=float(np.mean(np.abs(r)) * .01),
                optimizer_iterations=int(result.nit), penalty=penalty)


def predict(model, frame):
    x = frame[model['features']].to_numpy(float)
    x = np.where(np.isnan(x), model['median'], x)
    x = np.column_stack([np.ones(len(x)), np.clip((x - model['mean']) / model['std'], -10, 10)])
    weighted_p = expit(x @ model['beta'])
    magnitude = np.maximum(x @ model['magnitude_beta'], model['magnitude_floor'])
    return magnitude * (2 * weighted_p - 1), weighted_p


def metrics(frame, prediction):
    r = frame.target.to_numpy()
    return {'rows': len(r), 'first_date': str(frame.date.min().date()), 'last_date': str(frame.date.max().date()),
            'accuracy': float(np.mean((prediction > 0) == (r > 0))),
            'score': float(np.mean(np.where(prediction > 0, 1, -1) * r)),
            'return_mae': float(np.mean(np.abs(prediction - r))),
            'always_up_score': float(np.mean(r)), 'always_down_score': float(-np.mean(r))}

