"""Resumable FinMind US daily prices, shared by the download CLI and manager."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
API_URL = 'https://api.finmindtrade.com/api/v4/data'
START_DATE = '2022-01-01'
DEFAULT_SYMBOLS = ['^DJI', '^GSPC', '^IXIC', '^SOX', '^VIX', 'SPY', 'QQQ', 'SMH',
                   'AAPL', 'MSFT', 'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'AMAT',
                   'LRCX', 'MU', 'INTC', 'QCOM']
COLUMNS = ['date', 'stock_id', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']


class FinMindError(RuntimeError):
    pass


class RequestBudgetReached(FinMindError):
    pass


def read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_text(content, encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def price_path(directory, symbol):
    return directory / 'prices' / (quote(symbol, safe='^') + '.csv')


def data_hash(directory):
    """Hash model inputs only; timestamps and request progress never change the version."""
    digest = hashlib.sha256()
    for path in sorted((directory / 'prices').glob('*.csv')):
        digest.update(path.name.encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def year_windows(start, end):
    while start <= end:
        boundary = min(date(start.year, 12, 31), end)
        yield start, boundary
        start = boundary + timedelta(days=1)


def normalize_rows(rows, symbol, start, end):
    result = []
    aliases = {column.lower(): column for column in COLUMNS}
    for original in rows:
        if not isinstance(original, dict):
            raise FinMindError('FinMind returned a non-object price row')
        row = {aliases.get(key.lower(), key): value for key, value in original.items()}
        if str(row.get('stock_id')) != symbol:
            raise FinMindError(f'{symbol}: response contains a different stock_id')
        trading_date = date.fromisoformat(str(row.get('date')))
        if not start <= trading_date <= end:
            raise FinMindError(f'{symbol}: response date is outside requested range')
        normalized = {'date': trading_date.isoformat(), 'stock_id': symbol}
        for column in COLUMNS[2:]:
            value = row.get(column)
            if value is None and column == 'Adj_Close':
                normalized[column] = ''
                continue
            try:
                number = Decimal(str(value))
            except InvalidOperation as exc:
                raise FinMindError(f'{symbol}: invalid {column}') from exc
            if not number.is_finite():
                raise FinMindError(f'{symbol}: non-finite {column}')
            normalized[column] = format(number.normalize(), 'f')
        result.append(normalized)
    return result


def merge_prices(path, rows):
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    combined = {row['date']: row for row in csv.DictReader(io.StringIO(previous))}
    combined.update({row['date']: row for row in rows})
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(combined[key] for key in sorted(combined))
    if output.getvalue() != previous:
        atomic_write(path, output.getvalue())
    return sorted(combined)


class FinMindClient:
    def __init__(self, session, directory, interval=13, max_requests=None):
        self.session = session
        self.directory = directory
        self.interval = interval
        self.max_requests = max_requests
        self.requests = 0
        token = os.environ.get('FINMIND_TOKEN')
        self.headers = {'Authorization': f'Bearer {token}'} if token else {}

    def rate_limit(self, response):
        value = response.headers.get('Retry-After', '')
        try:
            seconds = int(value)
        except ValueError:
            try:
                seconds = int(parsedate_to_datetime(value).timestamp() - time.time())
            except (ValueError, OverflowError):
                seconds = 3600
        atomic_write(self.directory / '.retry_after', str(time.time() + max(1, seconds)))
        raise FinMindError(f'FinMind rate limit (429); retry after {max(1, seconds)} seconds, progress saved')

    def get(self, dataset, **params):
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise RequestBudgetReached('Request budget reached; progress saved')
        retry_file = self.directory / '.retry_after'
        if retry_file.exists() and time.time() < float(retry_file.read_text()):
            raise FinMindError('FinMind rate limit cooldown is active; progress saved')
        clock_file = self.directory / '.last_request'
        previous = float(clock_file.read_text()) if clock_file.exists() else 0
        delay = max(0, previous + self.interval - time.time())
        if delay:
            time.sleep(delay)
        atomic_write(clock_file, str(time.time()))
        self.requests += 1
        response = self.session.get(API_URL, headers=self.headers,
                                    params={'dataset': dataset, **params}, timeout=(10, 60))
        if response.status_code == 429:
            self.rate_limit(response)
        if response.status_code != 200:
            raise FinMindError(f'FinMind HTTP {response.status_code}; check FINMIND_TOKEN or service status')
        payload = response.json()
        if isinstance(payload, dict) and str(payload.get('status')) == '429':
            self.rate_limit(response)
        if not isinstance(payload, dict) or str(payload.get('status')) != '200':
            raise FinMindError('FinMind API rejected the request; check token, permissions and symbol')
        if not isinstance(payload.get('data'), list):
            raise FinMindError('FinMind response has no data array')
        return payload['data']


def sync_data(directory=ROOT / 'us_data', *, symbols=None, start_date=None, end_date=None,
              full=False, all_symbols=False, interval=13, max_requests=None, overlap_days=7):
    """Save each successful year separately, so interrupted downloads can resume."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / '.env')
    if interval < 0 or overlap_days < 0 or (max_requests is not None and max_requests < 1):
        raise ValueError('Invalid request interval, overlap or request budget')
    with (directory / '.sync.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FinMindError('Another FinMind download/update is running') from exc
        config = read_json(directory / 'config.json', {})
        metadata = read_json(directory / 'metadata.json', {'symbols': {}, 'cursor': 0})
        start = date.fromisoformat(start_date or config.get('start_date', START_DATE))
        end = date.fromisoformat(end_date) if end_date else datetime.now(ZoneInfo('America/New_York')).date()
        if start > end:
            raise ValueError('start_date must be on or before end_date')
        errors = {}
        pending = False
        with requests.Session() as session:
            client = FinMindClient(session, directory, interval, max_requests)
            if all_symbols:
                info = client.get('USStockInfo')
                symbols = sorted({str(row['stock_id']) for row in info if row.get('stock_id')})
                save_json(directory / 'stock_info.json', info)
            symbols = list(dict.fromkeys(symbols if symbols is not None else config.get('symbols', DEFAULT_SYMBOLS)))
            if not symbols or any(not isinstance(symbol, str) or not symbol.strip() or
                                  any(ord(char) < 32 for char in symbol) for symbol in symbols):
                raise ValueError('symbols must be non-empty stock IDs')
            config = {'dataset': 'USStockPrice', 'start_date': start.isoformat(), 'symbols': symbols}
            save_json(directory / 'config.json', config)
            cursor = metadata.get('cursor', 0) % len(symbols) if max_requests is not None else 0
            order = list(range(cursor, len(symbols))) + list(range(cursor))
            for index in order:
                symbol = symbols[index]
                record = metadata['symbols'].setdefault(symbol, {})
                metadata['cursor'] = index
                path = price_path(directory, symbol)
                first = start
                if not full and record.get('checked_through') and path.exists() and record.get('history_start') == start.isoformat():
                    checked = date.fromisoformat(record['checked_through'])
                    if record.get('history_complete', True):
                        first = max(start, checked - timedelta(days=overlap_days))
                    else:
                        first = max(start, checked + timedelta(days=1))
                try:
                    for begin, finish in year_windows(first, end):
                        rows = client.get('USStockPrice', data_id=symbol,
                                          start_date=begin.isoformat(), end_date=finish.isoformat())
                        rows = normalize_rows(rows, symbol, begin, finish)
                        dates = merge_prices(path, rows)
                        record.update(checked_through=finish.isoformat(), history_start=start.isoformat(),
                                      latest_date=dates[-1] if dates else None, rows=len(dates),
                                      history_complete=finish == end)
                        record.pop('error', None)
                        save_json(directory / 'metadata.json', metadata)
                    record['last_success'] = datetime.now(ZoneInfo('UTC')).isoformat()
                    metadata['cursor'] = (index + 1) % len(symbols)
                    print(f'{symbol}: {record.get("rows", 0)} rows, latest {record.get("latest_date")}', flush=True)
                except RequestBudgetReached:
                    pending = True
                    break
                except (FinMindError, requests.RequestException, ValueError) as exc:
                    # Atomic per-window writes leave earlier successful prices intact.
                    record['error'] = str(exc)
                    errors[symbol] = str(exc)
                    print(f'{symbol}: {exc}', file=sys.stderr, flush=True)
                    # Network/API failures commonly affect every symbol; resume from here next time.
                    break
                finally:
                    save_json(directory / 'metadata.json', metadata)
        metadata['last_attempt'] = datetime.now(ZoneInfo('UTC')).isoformat()
        metadata['pending'] = pending or bool(errors)
        save_json(directory / 'metadata.json', metadata)
        return {'dataHash': data_hash(directory), 'errors': errors, 'pending': metadata['pending'],
                'symbols': len(symbols), 'requests': client.requests}


def main():
    parser = argparse.ArgumentParser(description='FinMind USStockPrice daily history from 2022, resumable by year')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'us_data')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--symbols', nargs='+', help='Stock IDs (quote ^IXIC, ^GSPC, etc.)')
    selection.add_argument('--all', action='store_true', help='Use every USStockInfo stock ID; may take many API requests')
    parser.add_argument('--start-date', default=None)
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--interval', type=float, default=13, help='Minimum seconds between API requests')
    parser.add_argument('--max-requests', type=int)
    parser.add_argument('--overlap-days', type=int, default=7)
    parser.add_argument('--full', action='store_true', help='Re-fetch the full requested date range')
    args = parser.parse_args()
    try:
        result = sync_data(args.output_dir, symbols=args.symbols, start_date=args.start_date,
                           end_date=args.end_date, full=args.full, all_symbols=args.all,
                           interval=args.interval, max_requests=args.max_requests, overlap_days=args.overlap_days)
    except (FinMindError, requests.RequestException, ValueError) as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['errors']:
        raise SystemExit(1)
