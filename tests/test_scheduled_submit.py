"""Offline tests: injected Taipei times, temporary files, no real upload or login."""
from datetime import datetime
import threading
import unittest
from unittest.mock import patch

import test_manager
from manager.core import Manager


def at(value='2026-11-06T08:30:00+08:00'):
    return datetime.fromisoformat(value)


class ScheduledSubmitTests(unittest.TestCase):
    setUp = test_manager.ManagerTests.setUp
    add_model = test_manager.ManagerTests.add_model
    prepare_submission = test_manager.ManagerTests.prepare_submission
    archive = test_manager.ManagerTests.archive
    request = test_manager.ManagerTests.request

    def ready(self):
        self.prepare_submission()
        self.m.configure(auto=True)

    def drain_scheduled(self, current):
        key, action, model = self.m.submit_tasks.get_nowait()
        self.assertEqual(action, 'scheduled-submit')
        try:
            self.m.scheduled_submit(current)
        finally:
            self.m.pending.discard(key)
            self.m.submit_tasks.task_done()

    def test_0830_trigger_uses_taipei_time_and_separate_queue(self):
        self.ready()
        self.assertFalse(self.m.queue_scheduled_submission(at('2026-11-06T08:29:59+08:00')))
        self.assertTrue(self.m.queue_scheduled_submission(at('2026-11-06T00:30:00+00:00')))
        self.assertTrue(self.m.tasks.empty())
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            self.drain_scheduled(at())
            self.assertEqual(upload.call_count, 1)
        self.assertFalse(self.m.queue_scheduled_submission(at('2026-11-06T08:31:00+08:00')))
        self.assertEqual(self.m.state['lastScheduledSubmissionDate'], '2026-11-06')
        restarted = Manager(self.root)
        self.assertFalse(restarted.queue_scheduled_submission(at('2026-11-06T08:40:00+08:00')))
        self.assertTrue(restarted.queue_scheduled_submission(at('2026-11-07T08:30:00+08:00')))

    def test_scheduler_loop_enqueues_the_daily_submission(self):
        self.ready()
        with patch('manager.core.datetime', wraps=datetime) as clock, \
                patch.object(self.m.stop, 'wait', side_effect=[False, True]):
            clock.now.return_value = at()
            self.m.scheduler()
        self.assertIn('scheduled-submit:', self.m.pending)
        self.assertIn('refresh:', self.m.pending)
        self.assertEqual(self.m.submit_tasks.qsize(), 1)
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            self.drain_scheduled(at())
            upload.assert_called_once()

    def test_manual_predict_and_training_do_not_trigger_automatic_upload(self):
        self.ready()
        for action in ['predict', 'train']:
            with self.subTest(action=action):
                self.m.stop.clear()
                self.m.enqueue(action, 'good')
                with patch.object(self.m, 'run_model', side_effect=lambda *a, **kw: self.m.stop.set()) as run, \
                        patch.object(self.m, 'submit_selected') as upload:
                    self.m.worker()
                    run.assert_called_once_with('good', train=action == 'train')
                    upload.assert_not_called()

    def test_catch_up_before_deadline_but_never_at_or_after_nine(self):
        self.ready()
        self.assertTrue(self.m.queue_scheduled_submission(at('2026-11-06T08:45:00+08:00')))
        with patch.object(self.m, 'submit_selected') as upload:
            self.drain_scheduled(at('2026-11-06T09:00:00+08:00'))
            upload.assert_not_called()
        for hour in ['09:00:00', '22:10:00']:
            self.assertFalse(self.m.queue_scheduled_submission(at(f'2026-11-06T{hour}+08:00')))

    def test_disabled_or_missing_selection_does_not_queue(self):
        self.assertFalse(self.m.queue_scheduled_submission(at()))
        self.ready()
        self.m.configure(auto=False)
        self.assertFalse(self.m.queue_scheduled_submission(at()))
        self.m.configure(auto=True)
        self.assertTrue(self.m.queue_scheduled_submission(at()))
        self.m.configure(auto=False)
        with patch.object(self.m, 'submit_selected') as upload:
            self.drain_scheduled(at())
            upload.assert_not_called()

    def test_pre_upload_failure_retries_after_one_minute(self):
        self.ready()
        self.assertTrue(self.m.queue_scheduled_submission(at()))
        with patch.object(self.m, 'fetch_state', side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError):
                self.drain_scheduled(at())
        self.assertFalse(self.m.queue_scheduled_submission(at('2026-11-06T08:30:30+08:00')))
        self.assertTrue(self.m.queue_scheduled_submission(at('2026-11-06T08:31:00+08:00')))
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            self.drain_scheduled(at('2026-11-06T08:31:00+08:00'))
            upload.assert_called_once()

    def test_closed_or_wrong_day_session_never_uploaded(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            self.m.state['competition']['phase2']['canSubmit'] = False
            with self.assertRaises(ValueError):
                self.m.scheduled_submit(at())
            self.m.state['competition']['phase2'].update(canSubmit=True, tradingDate='2026-11-09')
            with self.assertRaisesRegex(ValueError, '日期不符'):
                self.m.scheduled_submit(at())
            upload.assert_not_called()
        self.assertIsNone(self.m.state.get('lastScheduledSubmissionDate'))

    def test_uncertain_upload_blocks_automatic_retry_even_after_restart(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.m.scheduled_submit(at())
        self.assertEqual(self.m.state['submissions'][0]['status'], 'uncertain')
        restarted = Manager(self.root)
        self.assertFalse(restarted.queue_scheduled_submission(at('2026-11-06T08:40:00+08:00')))

    def test_preview_checks_real_pipeline_without_upload_or_consuming_slot(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            result = self.m.submit_selected(dry_run=True)
            self.assertEqual(result['status'], 'dry-run')
            self.assertTrue(result['wouldSubmit'])
            self.assertEqual(result['tradingDate'], '2026-11-06')
            response = self.request('POST', '/api/submit-preview', b'{}')
            self.assertIn(b'200 OK', response)
            self.assertIn(b'dry-run', response)
            upload.assert_not_called()
        self.assertEqual(self.m.state['submissions'], [])
        self.assertIsNone(self.m.state.get('lastScheduledSubmissionDate'))

    def test_refresh_does_not_upload_even_when_auto_is_enabled(self):
        self.ready()
        archive = self.archive('download.zip', 'date,etl:adj_open,etl:adj_close\n2026-11-06,100,110\n')
        def download(destination):
            destination.write_bytes(archive.read_bytes())
        with patch.object(self.m, 'fetch_state'), patch('downloads.download_data', side_effect=download), \
                patch.object(self.m, 'submit_selected') as upload:
            self.m.refresh()
            upload.assert_not_called()

    def test_submission_worker_is_not_blocked_by_prediction_worker(self):
        self.ready()
        entered, release, uploaded = threading.Event(), threading.Event(), threading.Event()
        original = self.m.scheduled_submit
        def long_prediction(*args, **kwargs):
            entered.set()
            release.wait(3)
        workers = [threading.Thread(target=self.m.worker),
                   threading.Thread(target=self.m.worker, args=(self.m.submit_tasks,))]
        with patch.object(self.m, 'run_model', side_effect=long_prediction), \
                patch.object(self.m, 'fetch_state'), \
                patch.object(self.m, 'scheduled_submit', side_effect=lambda: original(at())), \
                patch('submit.submit', side_effect=lambda *args, **kwargs: uploaded.set()):
            try:
                self.m.enqueue('predict', 'good')
                for worker in workers:
                    worker.start()
                self.assertTrue(entered.wait(1))
                self.assertTrue(self.m.queue_scheduled_submission(at()))
                self.assertTrue(uploaded.wait(1), 'Training/prediction must not block the submission queue')
            finally:
                release.set()
                self.m.stop.set()
                for worker in workers:
                    worker.join(3)


if __name__ == '__main__':
    unittest.main()
