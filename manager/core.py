import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TAIPEI = ZoneInfo('Asia/Taipei')
sys.path.insert(0, str(ROOT / 'script'))


def now():
    return datetime.now(TAIPEI).isoformat(timespec='seconds')


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def unpack(archive, destination):
    """Validate before extracting; compare member contents, not ZIP timestamps."""
    with zipfile.ZipFile(archive) as z:
        members = sorted((i for i in z.infolist() if not i.is_dir()), key=lambda i: i.filename)
        digest = hashlib.sha256()
        for item in members:
            path = Path(item.filename)
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'data':
                raise ValueError(f'Unsafe archive path: {item.filename}')
            digest.update(item.filename.encode())
            digest.update(z.read(item))
        # Build a fresh tree, so files removed upstream do not linger.
        staging = destination.parent / 'extracting'
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        z.extractall(staging)
        if not (staging / 'data' / 'market').is_dir():
            raise ValueError('Archive has no data/market directory')
        old = destination.with_name('previous_data')
        if old.exists():
            shutil.rmtree(old)
        if destination.exists():
            destination.rename(old)
        (staging / 'data').rename(destination)
        shutil.rmtree(staging)
        if old.exists():
            shutil.rmtree(old)
        return digest.hexdigest()


def returns(data):
    result = {}
    for path in sorted((data / 'market').glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                try:
                    opening, close = float(row['etl:adj_open']), float(row['etl:adj_close'])
                    if opening > 0 and math.isfinite(opening) and math.isfinite(close):
                        result[(path.stem, row['date'])] = (close - opening) / opening
                except (KeyError, ValueError, TypeError):
                    continue
    return result


def score_prediction(payload, realized, created_at):
    """Only publish complete daily scores; separate retrospective from pre-deadline runs."""
    scores = []
    for col, date in enumerate(payload['dates']):
        values = []
        for stock, row in zip(payload['stockIds'], payload['predictions']):
            value = realized.get((stock, date))
            if value is None or row[col] is None:
                break
            values.append((2 * row[col] - 1) * value)
        if len(values) == len(payload['stockIds']):
            cutoff = datetime.fromisoformat(date + 'T09:00:00').replace(tzinfo=TAIPEI)
            scores.append({'date': date, 'score': sum(values) / len(values),
                           'count': len(values), 'prospective': datetime.fromisoformat(created_at) < cutoff})
    return scores


class Manager:
    def __init__(self, root=ROOT, state_interval=300, command_timeout=7200):
        self.root = Path(root)
        self.store = self.root / 'manager_data'
        self.store.mkdir(exist_ok=True)
        (self.store / 'runs').mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.tasks = queue.Queue(maxsize=32)
        self.submit_tasks = queue.Queue(maxsize=8)
        self.submission_lock = threading.Lock()
        from auth import CompetitionSession
        self.competition_session = CompetitionSession()
        self.stop = threading.Event()
        self.state_interval = state_interval
        self.command_timeout = command_timeout
        self.path = self.store / 'manager.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'selected': None, 'autoSubmit': False, 'dataHash': None, 'runs': [],
            'models': {}, 'events': [], 'competition': None, 'rawState': {},
            'submissions': [], 'lastCheck': None, 'lastStateCheck': None,
        }
        # Interrupted runs must be retried instead of remaining permanently 'running'.
        for record in self.state['runs']:
            if record['status'] == 'running':
                record.update(status='failed', error='Manager 在執行期間停止，請重新執行')
                self.state['models'][record['model']] = {
                    'status': 'failed', 'run': record['id'], 'error': record['error']}
        self.save()
        self.busy = None
        self.submission_busy = None
        self.pending = set()

    def save(self):
        with self.lock:
            atomic_json(self.path, self.state)

    def event(self, message, *, submission=False):
        with self.lock:
            event = {'at': now(), 'message': message}
            if submission:
                event['submission'] = True
            self.state['events'] = ([event] + self.state['events'])[:100]
            self.save()

    def discover(self):
        return {p.parent.name: p for p in sorted((self.root / 'model').glob('*/usage.txt'))
                if not p.parent.is_symlink() and p.parent.name not in {'manager_data', 'data', '.venv'}}

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            result.pop('rawState', None)
            result['availableModels'] = list(self.discover())
            result['busy'] = self.submission_busy or self.busy
            result['pending'] = sorted(self.pending)
            result['schedule'] = ['05:10', '22:10']
            result['submissionTime'] = '08:30'
            return result

    def enqueue(self, action, model=None):
        if model is not None and model not in self.discover():
            raise ValueError('Unknown model')
        key = action + ':' + (model or '')
        with self.lock:
            if key in self.pending:
                return False
            tasks = self.submit_tasks if action in {'submit', 'scheduled-submit'} else self.tasks
            tasks.put_nowait((key, action, model))
            self.pending.add(key)
        return True

    def configure(self, selected=None, auto=None):
        with self.lock:
            if selected is not None:
                if selected not in self.discover():
                    raise ValueError('Unknown model')
                self.state['selected'] = selected
            if auto is not None:
                if not isinstance(auto, bool):
                    raise ValueError('autoSubmit must be boolean')
                self.state['autoSubmit'] = auto
            self.save()

    def fetch_state(self):
        from state import fetch_state, summarize
        raw = fetch_state(self.competition_session)
        compact = summarize(raw)
        with self.lock:
            self.state.update(rawState=raw, competition=compact, lastStateCheck=now())
            self.save()
        self.event('比賽資訊已更新')

    def refresh(self):
        from downloads import download_data
        try:
            self.fetch_state()
        except Exception as exc:
            self.event(f'比賽資訊更新失敗：{exc}')
        archive = self.store / 'data.zip'
        download_data(archive)
        digest = unpack(archive, self.root / 'data')
        with self.lock:
            changed = digest != self.state['dataHash']
            self.state.update(dataHash=digest, lastCheck=now())
            if changed:
                self.state['dataUpdatedAt'] = now()
            self.save()
        self.rescore()
        for name in self.discover():
            status = self.state['models'].get(name, {})
            # Also retry models that failed on an otherwise unchanged data revision.
            phase = (self.state.get('competition') or {}).get('phase2') or {}
            latest = next((r for r in reversed(self.state['runs']) if r['model'] == name), {})
            new_session = phase.get('isOpen') and latest.get('dates') != [phase.get('tradingDate')]
            if changed or new_session or status.get('dataHash') != digest or status.get('error'):
                try:
                    self.run_model(name)
                except Exception as exc:
                    self.event(f'{name} 執行失敗：{exc}')
        self.event('資料已更新' if changed else '資料內容未變更')

    def template(self):
        from submit import parse_csv
        sample = self.root / 'model' / 'all_down' / 'all_down.csv'
        base = parse_csv(sample)
        phase = (self.state.get('competition') or {}).get('phase2') or {}
        path = self.store / 'template.csv'
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if phase.get('tradingDate') and phase.get('isOpen'):
                writer.writerow(['id', phase['tradingDate']])
                writer.writerows([stock, 0] for stock in base['stockIds'])
            else:
                writer.writerow(['id', *base['dates']])
                writer.writerows([stock, *row] for stock, row in zip(base['stockIds'], base['predictions']))
        return path

    def command(self, command, cwd, env, log):
        # usage.txt is trusted local code, like running the model from a terminal.
        with log.open('a', encoding='utf-8') as output:
            process = subprocess.Popen(command, shell=True, cwd=cwd, env=env,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=self.command_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError(f'指令超過 {self.command_timeout} 秒')
        if code:
            raise RuntimeError(f'指令結束碼 {code}，請查看執行紀錄')

    def run_model(self, name, train=False):
        from submit import parse_csv
        usage = self.discover()[name]
        lines = usage.read_text(encoding='utf-8').splitlines()
        run_id = datetime.now(TAIPEI).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        folder = self.store / 'runs' / run_id
        folder.mkdir()
        started = now()
        record = {'id': run_id, 'model': name, 'createdAt': started, 'dataHash': self.state['dataHash'],
                  'trained': train, 'scores': [], 'error': None, 'status': 'running'}
        with self.lock:
            self.state['runs'].append(record)
            self.state['models'][name] = {'status': 'running', 'run': run_id}
            self.save()
        try:
            if len(lines) < 2 or not all(line.strip() for line in lines[:2]):
                raise ValueError('usage.txt 必須有兩行：預測指令、訓練指令')
            output = usage.parent / 'prediction.csv'
            env = os.environ.copy()
            env.update(MANAGER_ROOT=str(self.root), MANAGER_DATA_DIR=str(self.root / 'data'),
                       MANAGER_BASIC_DATA_DIR=str(self.root / 'basic_data'),
                       MANAGER_OUTPUT=str(output), MANAGER_TEMPLATE=str(self.template()))
            env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
            if train:
                self.command(lines[1], usage.parent, env, folder / 'run.log')
            before = output.stat() if output.exists() else None
            self.command(lines[0], usage.parent, env, folder / 'run.log')
            if not output.is_file():
                raise ValueError('模型未建立工作目錄中的 prediction.csv')
            after = output.stat()
            if before is not None and (before.st_ino, before.st_mtime_ns, before.st_ctime_ns, before.st_size) == (
                    after.st_ino, after.st_mtime_ns, after.st_ctime_ns, after.st_size):
                raise ValueError('模型未更新 prediction.csv，拒絕使用上次的預測')
            payload = parse_csv(output)
            base = parse_csv(self.root / 'model' / 'all_down' / 'all_down.csv')
            if payload['stockIds'] != base['stockIds'] or any(v is None for row in payload['predictions'] for v in row):
                raise ValueError('預測必須包含模板中依序排列的 50 支股票，且不可留白')
            for date in payload['dates']:
                datetime.strptime(date, '%Y-%m-%d')
            shutil.copyfile(output, folder / 'prediction.csv')
            # Timestamp AFTER generation, so training across a deadline cannot count as live.
            created_at = now()
            scores = score_prediction(payload, returns(self.root / 'data'), created_at)
            with self.lock:
                record.update(createdAt=created_at, status='done', scores=scores, dates=payload['dates'])
            self.event(f'{name} 預測完成')
        except Exception as exc:
            with self.lock:
                record.update(error=str(exc), status='failed')
            raise
        finally:
            with self.lock:
                self.state['models'][name] = {'status': record['status'], 'run': run_id,
                    'error': record['error'], 'dataHash': record['dataHash']}
                self.save()

    def rescore(self):
        from submit import parse_csv
        realized = returns(self.root / 'data')
        with self.lock:
            for record in self.state['runs']:
                if record['status'] == 'done':
                    record['scores'] = score_prediction(parse_csv(self.store / 'runs' / record['id'] / 'prediction.csv'),
                                                        realized, record['createdAt'])
            self.save()

    def submit_selected(self, dry_run=False, trading_date=None):
        # Manual submissions and the daily timer share a lock and deduplication.
        with self.submission_lock:
            return self._submit_selected(dry_run=dry_run, trading_date=trading_date)

    def queue_submission_prediction(self, model):
        with self.lock:
            if 'refresh:' in self.pending or f'train:{model}' in self.pending:
                return '資料同步或訓練正在處理，稍後再檢查預測'
            if self.enqueue('predict', model):
                return '已排入重新預測，完成後由排程重試'
            return '預測工作已排隊或執行中'

    def _submit_selected(self, dry_run=False, trading_date=None):
        from auth import RequestNotAccepted
        from submit import parse_csv, submit
        # Recheck the active session and quota immediately before a real upload.
        self.fetch_state()
        with self.lock:
            phase = copy.deepcopy(self.state['competition']['phase2'])
        if phase.get('canSubmit') is not True:
            if not phase.get('isOpen'):
                raise ValueError('官方 Phase 2 場次目前未開放' +
                                 (f'，截止時間 {phase["deadline"]}' if phase.get('deadline') else ''))
            if phase.get('remaining') == 0:
                raise ValueError('目前 Phase 2 場次的提交額度已用完')
            raise ValueError('官方 API 未提供可確認的 Phase 2 提交額度')
        if trading_date is not None and phase.get('tradingDate') != trading_date:
            raise ValueError('08:30 排程僅提交今天的場次，官方目前場次日期不符')
        with self.lock:
            name = self.state['selected']
            latest = copy.deepcopy(next((r for r in reversed(self.state['runs']) if r['model'] == name), None))
            data_hash = self.state['dataHash']
        if name is None:
            raise ValueError('請先選擇提交模型')
        if not latest or latest['status'] != 'done':
            error = latest.get('error') if latest else None
            raise ValueError(f'{name} 沒有成功的最新預測' + (f'，{error}' if error else ''))
        if latest['dataHash'] != data_hash:
            if trading_date is not None and not dry_run:
                raise ValueError('模型預測使用舊資料，' + self.queue_submission_prediction(name))
            raise ValueError('模型預測使用舊資料，請先重新產生預測')
        path = self.store / 'runs' / latest['id'] / 'prediction.csv'
        payload = parse_csv(path)
        if payload['dates'] != [phase['tradingDate']]:
            recovery = '請先產生該場次的單日預測'
            if trading_date is not None and not dry_run:
                recovery = self.queue_submission_prediction(name)
            raise ValueError(f'官方場次為 {phase["tradingDate"]}，'
                             f'{name} 的預測從 {payload["dates"][0]} 起有 {len(payload["dates"])} 個日期欄，'
                             f'{recovery}')
        key = phase['tradingDate'] + ':' + hashlib.sha256(path.read_bytes()).hexdigest()
        previous = next((s for s in self.state['submissions'] if s['key'] == key), None)
        if dry_run:
            return {'status': 'dry-run', 'wouldSubmit': previous is None, 'model': name,
                    'tradingDate': phase['tradingDate'], 'file': str(path),
                    'stocks': len(payload['stockIds']), 'dates': payload['dates'],
                    'previousStatus': previous['status'] if previous else None,
                    'submissionTime': '08:30', 'timezone': 'Asia/Taipei'}
        if previous:
            if previous['status'] == 'sent':
                self.event('相同日期與預測已有成功提交紀錄', submission=True)
            else:
                self.event('相同預測的提交結果未確認，請核對官方 Submission Status', submission=True)
            return previous['status']
        # Save intent first: ambiguous network failure must not trigger automatic duplicates.
        entry = {'key': key, 'at': now(), 'model': name, 'run': latest['id'], 'status': 'sending'}
        with self.lock:
            self.state['submissions'].append(entry)
            if trading_date is not None:
                self.state['lastScheduledSubmissionDate'] = trading_date
            self.save()
        try:
            submit(path, 2, dry_run=False, client=self.competition_session)
            with self.lock:
                entry['status'] = 'sent'
            self.event(f'{name} 已提交 {phase["tradingDate"]}', submission=True)
        except RequestNotAccepted:
            # The server did not accept this upload. Release its reservation.
            with self.lock:
                self.state['submissions'].remove(entry)
                if trading_date is not None:
                    self.state.pop('lastScheduledSubmissionDate', None)
            raise
        except Exception as exc:
            with self.lock:
                entry.update(status='uncertain', error=str(exc))
            self.event('提交結果未確認，請至官方網站檢查紀錄；不會自動重送')
            raise
        finally:
            self.save()
        return entry['status']

    def scheduled_submission_due(self, current):
        current = current.astimezone(TAIPEI)
        start = current.replace(hour=8, minute=30, second=0, microsecond=0)
        deadline = current.replace(hour=9, minute=0, second=0, microsecond=0)
        with self.lock:
            return (self.state['autoSubmit'] and self.state['selected'] is not None
                    and start <= current < deadline
                    and self.state.get('lastScheduledSubmissionDate') != current.date().isoformat())

    def queue_scheduled_submission(self, current):
        current = current.astimezone(TAIPEI)
        if not self.scheduled_submission_due(current):
            return False
        if current < getattr(self, '_next_scheduled_attempt', current):
            return False
        if self.enqueue('scheduled-submit'):
            self._next_scheduled_attempt = current + timedelta(minutes=1)
            self.event('08:30 自動提交已排入佇列')
            return True
        return False

    def scheduled_submit(self, current=None):
        current = (current or datetime.now(TAIPEI)).astimezone(TAIPEI)
        # Recheck on execution: disable/cancel, queue delay, or deadline may have intervened.
        if not self.scheduled_submission_due(current):
            return
        date = current.date().isoformat()
        try:
            status = self.submit_selected(trading_date=date)
        except Exception:
            # An ambiguous upload must not be retried with a different CSV after restart.
            with self.lock:
                uncertain = any(s['key'].startswith(date + ':') and s['status'] in {'sending', 'uncertain'}
                                for s in self.state['submissions'])
                if uncertain:
                    self.state['lastScheduledSubmissionDate'] = date
                    self.save()
            raise
        with self.lock:
            self.state['lastScheduledSubmissionDate'] = date
            self.save()
        self.event(f'08:30 排程完成：{date}（{status}）')

    def worker(self, tasks=None):
        tasks = self.tasks if tasks is None else tasks
        is_submission = tasks is self.submit_tasks
        while not self.stop.is_set():
            try:
                key, action, name = tasks.get(timeout=1)
            except queue.Empty:
                continue
            with self.lock:
                if is_submission:
                    self.submission_busy = key
                else:
                    self.busy = key
            try:
                if action == 'refresh':
                    self.refresh()
                elif action == 'state':
                    self.fetch_state()
                elif action in {'train', 'predict'}:
                    self.run_model(name, train=action == 'train')
                elif action == 'scheduled-submit':
                    self.scheduled_submit()
                elif action == 'submit':
                    self.submit_selected()
            except Exception as exc:
                label = {'submit': '手動提交', 'scheduled-submit': '08:30 自動提交'}.get(action, key)
                self.event(f'{label}失敗：{exc}', submission=is_submission)
            finally:
                with self.lock:
                    if is_submission:
                        self.submission_busy = None
                    else:
                        self.busy = None
                    self.pending.discard(key)
                tasks.task_done()

    def scheduler(self):
        self.enqueue('refresh')
        next_state = time.monotonic() + self.state_interval
        while not self.stop.wait(15):
            current = datetime.now(TAIPEI)
            try:
                self.queue_scheduled_submission(current)
            except queue.Full:
                self.event('自動提交佇列已滿，稍後重試')
            # Persist scheduled slots, including a same-day catch-up after a restart.
            slots = [current.replace(hour=h, minute=10, second=0, microsecond=0) for h in (5, 22)]
            slot = max((s for s in slots if s <= current), default=None)
            if slot and self.state.get('lastSlot') != slot.isoformat():
                if self.enqueue('refresh'):
                    with self.lock:
                        self.state['lastSlot'] = slot.isoformat()
                        self.save()
            # Failed network/data refreshes get another attempt after ten minutes.
            last = self.state.get('lastCheck')
            stale = not last or datetime.fromisoformat(last) < (slot or current - timedelta(days=1))
            if stale and time.monotonic() >= getattr(self, '_retry', 0):
                self.enqueue('refresh')
                self._retry = time.monotonic() + 600
            if time.monotonic() >= next_state:
                self.enqueue('state')
                next_state = time.monotonic() + self.state_interval

    def start(self):
        for target in (self.worker, self.scheduler):
            threading.Thread(target=target, daemon=True).start()
        threading.Thread(target=self.worker, args=(self.submit_tasks,), daemon=True).start()
