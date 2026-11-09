"""Phase 2 deadline boundaries in the competition's Taipei timezone."""
from datetime import datetime
import unittest

from script.state import phase2_availability


class Phase2DeadlineTests(unittest.TestCase):
    def availability(self, at, **schedule):
        state = {
            'activePhase2Day': {'tradingDate': '2026-11-10', **schedule},
            'uploadQuota': {'phase2': {'teamLimit': 10, 'teamCount': 0}},
        }
        return phase2_availability(state, datetime.fromisoformat(at))

    def test_missing_api_deadline_defaults_to_nine_taipei(self):
        result = self.availability('2026-11-10T08:59:59+08:00')
        self.assertTrue(result['canSubmit'])
        self.assertEqual(result['deadline'], '2026-11-10T09:00:00+08:00')

    def test_closed_exactly_at_nine_and_in_the_evening(self):
        for at in ['2026-11-10T09:00:00+08:00', '2026-11-10T22:10:00+08:00',
                   '2026-11-10T01:00:00+00:00']:
            with self.subTest(at=at):
                result = self.availability(at, deadline='2026-11-10T23:59:00+08:00')
                self.assertFalse(result['isOpen'])
                self.assertFalse(result['canSubmit'])

    def test_disabled_or_not_yet_open_session_cannot_submit(self):
        self.assertFalse(self.availability('2026-11-10T08:00:00+08:00', enabled=False)['canSubmit'])
        self.assertFalse(self.availability('2026-11-09T22:10:00+08:00',
                                          openAt='2026-11-10T00:00:00+08:00')['canSubmit'])

    def test_earlier_official_deadline_is_respected(self):
        self.assertFalse(self.availability('2026-11-10T08:30:00+08:00',
                                          deadline='2026-11-10T00:30:00Z')['canSubmit'])


if __name__ == '__main__':
    unittest.main()
