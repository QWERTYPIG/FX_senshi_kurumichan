"""FinMind downloads and manager refreshes without network requests."""
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import test_manager
from manager.core import Manager
from finmind_us import (FinMindClient, FinMindError, data_hash, normalize_rows,
                        price_path, save_json, sync_data)


def row(day, symbol='AAPL', close=10):
    return {'date':day, 'stock_id':symbol, 'Open':9, 'High':11, 'Low':8,
            'Close':close, 'Adj_Close':close, 'Volume':100}


def response(rows=None, status=200):
    return Mock(status_code=status, headers={}, json=Mock(return_value={'status':200, 'data':rows or []}))


class FinMindTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'us_data'

    def sync(self, session, **kwargs):
        with patch('finmind_us.requests.Session') as factory:
            factory.return_value.__enter__.return_value = session
            return sync_data(self.directory, interval=0, **kwargs)

    def test_download_uses_calendar_years_and_keeps_full_history(self):
        session = Mock()
        session.get.side_effect = [response([row('2022-12-30')]), response([row('2023-01-03')])]
        result = self.sync(session, symbols=['AAPL'], start_date='2022-01-01', end_date='2023-01-05')
        self.assertFalse(result['pending'])
        calls = session.get.call_args_list
        self.assertEqual(calls[0].kwargs['params']['start_date'], '2022-01-01')
        self.assertEqual(calls[0].kwargs['params']['end_date'], '2022-12-31')
        self.assertEqual(calls[1].kwargs['params']['start_date'], '2023-01-01')
        with price_path(self.directory, 'AAPL').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['date'] for r in rows], ['2022-12-30','2023-01-03'])

    def test_incremental_overlap_corrects_existing_rows_and_is_idempotent(self):
        session = Mock()
        session.get.return_value = response([row('2022-01-03')])
        self.sync(session, symbols=['AAPL'], start_date='2022-01-01', end_date='2022-01-10')
        session.get.return_value = response([row('2022-01-03', close=12), row('2022-01-11', close=13)])
        result = self.sync(session, end_date='2022-01-12')
        self.assertEqual(session.get.call_args.kwargs['params']['start_date'], '2022-01-03')
        with price_path(self.directory, 'AAPL').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['Close'] for r in rows], ['12','13'])
        session.get.return_value = response([row('2022-01-11', close=13)])
        again = self.sync(session, end_date='2022-01-12')
        self.assertFalse(again['pending'])
        self.assertFalse(again['errors'])
        self.assertEqual(result['dataHash'], again['dataHash'])
        self.assertEqual(len(rows), 2)

    def test_interrupted_history_resumes_without_losing_previous_year(self):
        session = Mock()
        session.get.return_value = response([row('2022-12-30')])
        partial = self.sync(session, symbols=['AAPL'], start_date='2022-01-01',
                            end_date='2023-01-05', max_requests=1)
        self.assertTrue(partial['pending'])
        session.get.side_effect = [response([row('2023-01-03')])]
        finished = self.sync(session, end_date='2023-01-05', max_requests=1)
        self.assertFalse(finished['pending'])
        self.assertEqual(session.get.call_args.kwargs['params']['start_date'], '2023-01-01')
        self.assertEqual(json.loads((self.directory / 'metadata.json').read_text())['symbols']['AAPL']['rows'], 2)

    def test_api_or_schema_errors_preserve_prices_and_retry_cursor(self):
        session = Mock()
        session.get.return_value = response([row('2022-01-03')])
        self.sync(session, symbols=['AAPL'], start_date='2022-01-01', end_date='2022-01-04')
        before = price_path(self.directory, 'AAPL').read_bytes()
        for invalid in [response(status=500), response([row('2022-01-04', symbol='WRONG')]),
                        Mock(status_code=200, json=Mock(return_value={'status':400,'data':[]}))]:
            session.get.return_value = invalid
            result = self.sync(session, end_date='2022-01-05')
            self.assertTrue(result['pending'])
            self.assertTrue(result['errors'])
            self.assertEqual(price_path(self.directory, 'AAPL').read_bytes(), before)

    def test_429_waits_across_new_client_instances(self):
        self.directory.mkdir()
        session = Mock()
        session.get.return_value = Mock(status_code=429, headers={'Retry-After':'120'})
        with patch('finmind_us.time.time', return_value=100):
            client = FinMindClient(session, self.directory, interval=0)
            with self.assertRaises(FinMindError):
                client.get('USStockPrice')
        with patch('finmind_us.time.time', return_value=160):
            with self.assertRaises(FinMindError):
                FinMindClient(session, self.directory, interval=0).get('USStockPrice')
        self.assertEqual(session.get.call_count, 1)

    def test_all_symbols_are_discovered_and_saved_for_updates(self):
        session = Mock()
        session.get.side_effect = [response([{'stock_id':'AAPL'}, {'stock_id':'MSFT'}, {'stock_id':'AAPL'}]),
                                   response([row('2022-01-03')]), response([row('2022-01-03', 'MSFT')])]
        result = self.sync(session, all_symbols=True, start_date='2022-01-01', end_date='2022-01-04')
        self.assertEqual(result['symbols'], 2)
        self.assertEqual(json.loads((self.directory / 'config.json').read_text())['symbols'], ['AAPL','MSFT'])
        self.assertTrue((self.directory / 'stock_info.json').exists())

    def test_symbol_paths_cannot_escape_prices_directory(self):
        path = price_path(self.directory, '../A/B')
        self.assertEqual(path.parent, self.directory / 'prices')
        self.assertNotIn('/', path.name)


class FinMindManagerTests(unittest.TestCase):
    setUp = test_manager.ManagerTests.setUp
    add_model = test_manager.ManagerTests.add_model
    archive = test_manager.ManagerTests.archive

    def configure(self):
        directory = self.root / 'us_data'
        directory.mkdir()
        save_json(directory / 'config.json', {'symbols':['AAPL'], 'start_date':'2022-01-01'})
        return directory

    def test_us_only_changes_regenerate_predictions_and_track_composite_version(self):
        self.add_model('sample')
        directory = self.configure()
        self.m.update_data_version('competition-version')
        self.m.run_model('sample')
        before = self.m.state['dataHash']
        def sync(*args, **kwargs):
            (directory / 'prices').mkdir()
            price_path(directory, 'AAPL').write_text('date,Close\n2026-11-20,11\n')
            return {'dataHash':data_hash(directory), 'pending':False, 'errors':{}, 'symbols':1,'requests':1}
        with patch.object(self.m, 'fetch_state'), patch('finmind_us.sync_data', side_effect=sync):
            self.m.refresh_us()
        self.assertNotEqual(self.m.state['dataHash'], before)
        self.assertEqual(len(self.m.state['runs']), 2)
        self.assertEqual(self.m.state['competitionDataHash'], 'competition-version')
        with patch.object(self.m, 'fetch_state'), patch('finmind_us.sync_data', return_value={
                'dataHash':data_hash(directory),'pending':False,'errors':{},'symbols':1,'requests':1}):
            self.m.refresh_us()
        self.assertEqual(len(self.m.state['runs']), 2)

    def test_us_failure_still_allows_competition_download_and_prediction(self):
        self.configure()
        self.add_model('sample')
        archive = self.archive('download.zip', 'date,etl:adj_open,etl:adj_close\n2026-11-06,100,110\n')
        def download(destination):
            destination.write_bytes(archive.read_bytes())
        with patch.object(self.m, 'fetch_state'), patch('downloads.download_data', side_effect=download), \
                patch('finmind_us.sync_data', side_effect=RuntimeError('offline')):
            self.m.refresh()
        self.assertEqual(self.m.state['runs'][-1]['status'], 'done')
        self.assertEqual(self.m.state['usDataStatus']['error'], 'offline')

    def test_0810_update_enters_worker_queue(self):
        self.configure()
        with patch('manager.core.datetime', wraps=datetime) as clock, \
                patch.object(self.m.stop, 'wait', side_effect=[False, True]):
            clock.now.return_value = datetime.fromisoformat('2026-11-21T08:10:00+08:00')
            self.m.scheduler()
        self.assertIn('us-update:', self.m.pending)
        self.assertTrue(self.m.submit_tasks.empty())

    def test_training_and_prediction_receive_us_directory(self):
        self.add_model('sample')
        import shutil
        observed = []
        def command(cmd, cwd, env, log):
            observed.append(env['MANAGER_US_DATA_DIR'])
            shutil.copyfile(env['MANAGER_TEMPLATE'], env['MANAGER_OUTPUT'])
        with patch.object(self.m, 'command', side_effect=command):
            self.m.run_model('sample', train=True)
        self.assertEqual(observed, [str(self.root / 'us_data')] * 2)


if __name__ == '__main__':
    unittest.main()
