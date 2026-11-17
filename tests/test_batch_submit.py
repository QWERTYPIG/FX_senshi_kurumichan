"""Multi-model batch order, recovery and official score mapping without network access."""
from datetime import datetime
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

import test_manager
from auth import RequestNotAccepted
from manager.core import Manager, PredictionNotReady
from submit import SubmissionRejected


AT = datetime.fromisoformat('2026-11-06T08:30:00+08:00')


def model_of(call):
    return call.kwargs['note'].split(';')[0].removeprefix('model=')


class BatchSubmitTests(unittest.TestCase):
    setUp = test_manager.ManagerTests.setUp
    add_model = test_manager.ManagerTests.add_model
    prepare_submission = test_manager.ManagerTests.prepare_submission

    def ready(self):
        self.prepare_submission()
        for name in ('alpha', 'zeta'):
            self.add_model(name)
            self.m.run_model(name)
        self.m.state['competition']['phase2'].update(isOpen=True, remaining=10)
        self.m.configure(auto=True)

    def test_all_models_with_identical_predictions_are_submitted_selected_last(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit', return_value={'ok': True}) as upload:
            self.m.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['alpha', 'zeta', 'good'])
        filenames = [call.kwargs['file_name'] for call in upload.call_args_list]
        self.assertEqual(len(set(filenames)), 3)
        self.assertEqual(self.m.state['submissionBatches']['2026-11-06']['completed'], ['alpha','zeta','good'])
        self.assertEqual(self.m.state['lastScheduledSubmissionDate'], '2026-11-06')
        restarted = Manager(self.root)
        with patch.object(restarted, 'fetch_state'), patch('submit.submit') as upload:
            restarted.scheduled_submit(AT)
            upload.assert_not_called()

    def test_selected_is_uploaded_again_after_an_earlier_manual_submission(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            self.m.submit_selected()
            self.m.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['good','alpha','zeta','good'])

    def test_rate_limit_resumes_remaining_models_after_restart(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit', side_effect=[None, RequestNotAccepted('429')]):
            with self.assertRaises(RequestNotAccepted):
                self.m.scheduled_submit(AT)
        self.assertEqual(self.m.state['submissionBatches']['2026-11-06']['completed'], ['alpha'])
        self.assertIsNone(self.m.state.get('lastScheduledSubmissionDate'))
        restarted = Manager(self.root)
        with patch.object(restarted, 'fetch_state'), patch('submit.submit') as upload:
            restarted.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['zeta','good'])

    def test_unknown_upload_is_reconciled_by_unique_filename_before_continuing(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit', side_effect=[None, TimeoutError('network')]):
            with self.assertRaises(TimeoutError):
                self.m.scheduled_submit(AT)
        ambiguous = self.m.state['submissions'][-1]
        raw = {'submissionHistory':[{'phase':2, 'id':77, 'fileName':ambiguous['fileName'],
                                    'tradingDate':'2026-11-06', 'score':None}]}
        restarted = Manager(self.root)
        with patch.object(restarted, 'fetch_state', side_effect=lambda: restarted.sync_official_scores(raw)), \
                patch('submit.submit') as upload:
            restarted.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['good'])
        self.assertEqual(restarted.state['submissions'][1]['officialId'], 77)
        self.assertEqual(restarted.state['submissions'][1]['status'], 'sent')

    def test_accepted_final_upload_recovers_progress_even_when_quota_is_exhausted(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit'):
            self.m.scheduled_submit(AT)
        self.m.state.pop('lastScheduledSubmissionDate')
        batch = self.m.state['submissionBatches']['2026-11-06']
        batch.update(status='pending', completed=['alpha', 'zeta'])
        self.m.state['competition']['phase2'].update(canSubmit=False, remaining=0)
        self.m.save()
        restarted = Manager(self.root)
        with patch.object(restarted, 'fetch_state'), patch('submit.submit') as upload:
            restarted.scheduled_submit(AT)
            upload.assert_not_called()
        self.assertEqual(restarted.state['lastScheduledSubmissionDate'], '2026-11-06')
        self.assertEqual(restarted.state['submissionBatches']['2026-11-06']['completed'], ['alpha','zeta','good'])

    def test_quota_reserves_last_slot_for_selected_model(self):
        self.ready()
        def refresh():
            count = sum(entry['status'] == 'sent' for entry in self.m.state['submissions'])
            self.m.state['competition']['phase2']['remaining'] = 2 - count
        with patch.object(self.m, 'fetch_state', side_effect=refresh), patch('submit.submit') as upload:
            self.m.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['alpha','good'])
        self.assertIn('zeta', self.m.state['submissionBatches']['2026-11-06']['skipped'])

    def test_failed_model_is_reported_and_selected_still_goes_last(self):
        self.ready()
        record = next(r for r in reversed(self.m.state['runs']) if r['model'] == 'alpha')
        record.update(status='failed', error='No trained model')
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            self.m.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['zeta','good'])
        self.assertIn('No trained model', self.m.state['submissionBatches']['2026-11-06']['skipped']['alpha'])

    def test_selected_must_be_ready_before_other_models_are_uploaded(self):
        self.ready()
        record = next(r for r in reversed(self.m.state['runs']) if r['model'] == 'good')
        record['dataHash'] = 'old'
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            with self.assertRaises(PredictionNotReady):
                self.m.scheduled_submit(AT)
            upload.assert_not_called()
        self.assertIn('predict:good', self.m.pending)

    def test_stale_non_selected_model_pauses_then_resumes_before_final_model(self):
        self.ready()
        self.m.state['competition']['phase2']['isOpen'] = True
        alpha = next(r for r in self.m.state['runs'] if r['model'] == 'alpha')
        alpha['dataHash'] = 'old'
        with patch.object(self.m, 'fetch_state'), patch('submit.submit') as upload:
            with self.assertRaises(PredictionNotReady):
                self.m.scheduled_submit(AT)
            upload.assert_not_called()
            self.assertIn('predict:alpha', self.m.pending)
            self.m.run_model('alpha')
            self.m.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['alpha','zeta','good'])

    def test_rejected_non_selected_upload_is_skipped_and_final_is_submitted(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), \
                patch('submit.submit', side_effect=[SubmissionRejected('bad payload'), None, None]) as upload:
            self.m.scheduled_submit(AT)
        self.assertEqual([model_of(call) for call in upload.call_args_list], ['alpha','zeta','good'])
        self.assertEqual([s['model'] for s in self.m.state['submissions']], ['zeta','good'])
        self.assertIn('alpha', self.m.state['submissionBatches']['2026-11-06']['skipped'])

    def test_transport_sends_the_unique_filename_and_model_metadata(self):
        (self.root / 'model/all_down/all_down.csv').write_text(
            'id,2026-11-06\n' + ''.join(f'{stock},0\n' for stock in range(1000, 1050)))
        self.ready()
        response = Mock(status_code=201, json=Mock(return_value={'id':'receipt'}))
        with patch.object(self.m, 'fetch_state'), \
                patch.object(self.m.competition_session, 'request', return_value=response) as request, \
                redirect_stdout(io.StringIO()):
            self.m.scheduled_submit(AT)
        self.assertEqual(request.call_count, 3)
        for entry, call in zip(self.m.state['submissions'], request.call_args_list):
            self.assertEqual(call.kwargs['json']['fileName'], entry['fileName'])
            self.assertIn('model=' + entry['model'], call.kwargs['json']['note'])
            self.assertEqual(entry['receipt']['id'], 'receipt')
            self.assertEqual(len(call.kwargs['json']['payload']['predictions']), 50)

    def test_scores_from_current_and_archived_records_are_assigned_to_each_model(self):
        self.ready()
        with patch.object(self.m, 'fetch_state'), patch('submit.submit'):
            self.m.scheduled_submit(AT)
        entries = self.m.state['submissions']
        def record(entry, official_id, score):
            return {'id':official_id, 'phase':2, 'tradingDate':'2026-11-06', 'fileName':entry['fileName'],
                    'submittedAt':'2026-11-06T00:30:00Z', 'score':score}
        raw = {'team':{'id':'test'}, 'mySubmissions':[record(entries[2], 'current-id', -.01)],
               'submissionHistory':[record(entries[0], 101, .02), record(entries[1], 102, 0.0)]}
        with patch('state.fetch_state', return_value=raw), patch('state.summarize', return_value=self.m.state['competition']):
            self.m.fetch_state()
        self.assertEqual([(s['model'], s['officialScore']) for s in entries],
                         [('alpha', .02), ('zeta', 0.0), ('good', -.01)])
        restored = Manager(self.root)
        self.assertEqual(restored.state['submissions'][1]['officialScore'], 0.0)
        self.assertEqual(restored.state['submissions'][0]['officialId'], 101)

    def test_official_history_does_not_guess_scores_for_legacy_or_unrelated_files(self):
        self.ready()
        self.m.state['submissions'].append({'key':'2026-11-06:legacy', 'model':'good', 'status':'sent'})
        with patch.object(self.m, 'fetch_state'), patch('submit.submit'):
            self.m.scheduled_submit(AT)
        raw = {'mySubmissions':[{'phase':2, 'id':'unrelated', 'tradingDate':'2026-11-06',
                                'fileName':'prediction.csv', 'score':999}]}
        self.m.sync_official_scores(raw)
        self.assertTrue(all(s.get('officialScore') is None for s in self.m.state['submissions']))


if __name__ == '__main__':
    unittest.main()
