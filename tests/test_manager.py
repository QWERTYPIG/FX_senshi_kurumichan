import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from manager.core import Manager, returns, score_prediction, unpack
from manager.__main__ import handler_for


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'model/all_down').mkdir(parents=True)
        with (self.root / 'model/all_down/all_down.csv').open('w') as f:
            f.write('id,2026-11-06\n2330,0\n2317,0\n')
        (self.root / 'data/market').mkdir(parents=True)
        (self.root / 'data/market/2330.csv').write_text('date,etl:adj_open,etl:adj_close\n2026-11-06,100,110\n')
        (self.root / 'data/market/2317.csv').write_text('date,etl:adj_open,etl:adj_close\n2026-11-06,200,190\n')
        self.m = Manager(self.root)

    def test_scoring_uses_returns_not_accuracy_and_waits_for_all_labels(self):
        p = {'stockIds':['2330','2317'], 'dates':['2026-11-06','2026-11-09'], 'predictions':[[1,1],[0,1]]}
        result = score_prediction(p, returns(self.root / 'data'), '2026-11-06T08:59:59+08:00')
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['score'], .075)
        self.assertTrue(result[0]['prospective'])
        self.assertFalse(score_prediction(p, returns(self.root / 'data'), '2026-11-06T09:00:00+08:00')[0]['prospective'])

    def archive(self, name, value, timestamp=(2026,11,6,5,0,0)):
        with zipfile.ZipFile(self.root / name, 'w') as z:
            item = zipfile.ZipInfo('data/market/2330.csv', date_time=timestamp)
            z.writestr(item, value)
        return self.root / name

    def test_archive_hash_ignores_zip_metadata_and_removes_old_files(self):
        a = self.archive('a.zip', 'date,value\n2026-11-06,1\n')
        b = self.archive('b.zip', 'date,value\n2026-11-06,1\n', (2026,11,7,22,0,0))
        first = unpack(a, self.root / 'data')
        self.assertFalse((self.root / 'data/market/2317.csv').exists())
        self.assertEqual(first, unpack(b, self.root / 'data'))
        c = self.archive('c.zip', 'date,value\n2026-11-06,2\n')
        self.assertNotEqual(first, unpack(c, self.root / 'data'))

    def test_reject_zip_traversal_before_extracting(self):
        archive = self.root / 'evil.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('data/../../outside', 'bad')
        with self.assertRaises(ValueError):
            unpack(archive, self.root / 'data')
        self.assertTrue((self.root / 'data/market/2330.csv').exists())

    def add_model(self, name, fail=False):
        folder = self.root / 'model' / name
        folder.mkdir()
        command = 'python -c "raise SystemExit(4)"' if fail else 'python predict.py'
        (folder / 'usage.txt').write_text(command + '\npython -c "print(123)"\n')
        (folder / 'predict.py').write_text('import os, shutil\nshutil.copyfile(os.environ["MANAGER_TEMPLATE"], os.environ["MANAGER_OUTPUT"])\n')

    def test_training_and_prediction_preserved_then_rescored(self):
        self.add_model('sample')
        self.m.run_model('sample', train=True)
        record = self.m.state['runs'][0]
        self.assertEqual(record['status'], 'done')
        self.assertAlmostEqual(record['scores'][0]['score'], -.025)
        archived = self.m.store / 'runs' / record['id'] / 'prediction.csv'
        original = archived.read_bytes()
        self.assertEqual((self.root / 'model/sample/prediction.csv').read_bytes(), original)
        self.m.run_model('sample')
        self.assertEqual(archived.read_bytes(), original)
        self.assertEqual(len(self.m.state['runs']), 2)
        self.assertEqual(Manager(self.root).state['runs'][0]['id'], record['id'])
        self.m.rescore()

    def test_relative_output_is_archived_and_stale_output_is_rejected(self):
        self.add_model('sample')
        model = self.root / 'model/sample'
        (model / 'predict.py').write_text('import os, shutil\nshutil.copyfile(os.environ["MANAGER_TEMPLATE"], "prediction.csv")\n')
        self.m.run_model('sample')
        record = self.m.state['runs'][-1]
        archived = self.m.store / 'runs' / record['id'] / 'prediction.csv'
        original = archived.read_bytes()
        (model / 'predict.py').write_text('print("No CSV generated")\n')
        with self.assertRaisesRegex(ValueError, '未更新'):
            self.m.run_model('sample')
        self.assertEqual(self.m.state['runs'][-1]['status'], 'failed')
        self.assertEqual(archived.read_bytes(), original)
        self.assertEqual((model / 'prediction.csv').read_bytes(), original)

    def test_labels_arriving_later_rescore_the_original_prediction(self):
        self.add_model('sample')
        market = self.root / 'data/market/2317.csv'
        market.write_text('date,etl:adj_open,etl:adj_close\n')
        self.m.run_model('sample')
        record = self.m.state['runs'][0]
        self.assertEqual(record['scores'], [])
        created_at = record['createdAt']
        market.write_text('date,etl:adj_open,etl:adj_close\n2026-11-06,200,190\n')
        self.m.rescore()
        self.assertAlmostEqual(record['scores'][0]['score'], -.025)
        self.assertEqual(record['createdAt'], created_at)

    def test_restart_marks_interrupted_run_as_failed(self):
        self.m.state['runs'].append({'id':'interrupted','model':'sample','status':'running'})
        self.m.save()
        recovered = Manager(self.root)
        self.assertEqual(recovered.state['runs'][0]['status'], 'failed')
        self.assertEqual(recovered.state['models']['sample']['status'], 'failed')

    def test_refresh_retries_failed_model_without_repeating_success(self):
        self.add_model('good')
        self.add_model('bad', fail=True)
        archive = self.archive('download.zip', 'date,etl:adj_open,etl:adj_close\n2026-11-06,100,110\n')
        def download(destination):
            destination.write_bytes(archive.read_bytes())
        with patch.object(self.m, 'fetch_state'), patch('downloads.download_data', side_effect=download):
            self.m.refresh()
            self.m.refresh()
        self.assertEqual(len([r for r in self.m.state['runs'] if r['model'] == 'good']), 1)
        self.assertEqual(len([r for r in self.m.state['runs'] if r['model'] == 'bad']), 2)

    def prepare_submission(self):
        self.add_model('good')
        self.m.run_model('good')
        self.m.configure(selected='good')
        self.m.state['competition'] = {'phase2':{'tradingDate':'2026-11-06', 'canSubmit':True}}

    def test_submission_deduplicates_and_rejects_wrong_date(self):
        self.prepare_submission()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            self.m.submit_selected()
            self.m.submit_selected()
            self.assertEqual(upload.call_count, 1)
            self.m.state['competition']['phase2']['tradingDate'] = '2026-11-09'
            with self.assertRaises(ValueError):
                self.m.submit_selected()
            self.assertEqual(upload.call_count, 1)

    def test_unknown_upload_result_not_retried_and_failed_run_not_submitted(self):
        self.prepare_submission()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit', side_effect=TimeoutError) as upload:
            with self.assertRaises(TimeoutError):
                self.m.submit_selected()
            self.m.submit_selected()
            self.assertEqual(upload.call_count, 1)
        self.assertEqual(self.m.state['submissions'][0]['status'], 'uncertain')
        self.m.state['runs'].append({'model':'good','status':'failed'})
        with patch.object(self.m, 'fetch_state'), self.assertRaises(ValueError):
            self.m.submit_selected()

    def request(self, method, path, body=b'', headers=None):
        # Exercise real request parsing and response handling, without binding a network port.
        handler = object.__new__(handler_for(self.m))
        handler.path = path
        handler.request_version = 'HTTP/1.1'
        handler.requestline = method + ' ' + path
        handler.client_address = ('127.0.0.1', 12345)
        from email.message import Message
        handler.headers = Message()
        for key, value in {'Host':'127.0.0.1:8000','Content-Type':'application/json','Content-Length':str(len(body)), **(headers or {})}.items():
            handler.headers[key] = value
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.log_request = lambda *args: None
        getattr(handler, 'do_' + method)()
        return handler.wfile.getvalue()

    def test_web_status_actions_and_cross_site_rejection(self):
        response = self.request('GET', '/api/status')
        self.assertIn(b'200 OK', response)
        self.assertIn(b'availableModels', response)
        self.assertIn(b'202 Accepted', self.request('POST', '/api/refresh', b'{}'))
        self.assertIn('refresh:', self.m.pending)
        self.assertIn(b'403 Forbidden', self.request('POST', '/api/submit', b'{}', {'Origin':'http://evil.example'}))
        self.assertIn(b'415 Unsupported', self.request('POST', '/api/train', b'{}', {'Content-Type':'text/plain'}))
        self.assertIn(b'404 Not Found', self.request('GET', '/.env'))
        self.assertIn(b'200 OK', self.request('GET', '/'))


if __name__ == '__main__':
    unittest.main()
