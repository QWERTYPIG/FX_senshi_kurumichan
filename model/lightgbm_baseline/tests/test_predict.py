import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import prediction_template


class PredictionTemplateTests(unittest.TestCase):
    def test_explicit_date_does_not_read_historical_template(self):
        result = prediction_template(Path("does-not-exist.csv"), "2026-11-09", ["2330", "2317"])
        self.assertEqual(result.columns.tolist(), ["id", "2026-11-09"])
        self.assertEqual(result["id"].tolist(), ["2330", "2317"])

    def test_reads_selected_daily_template(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "template.csv"
            pd.DataFrame({"id": ["2330"], "2026-11-09": [0]}).to_csv(path, index=False)
            result = prediction_template(path, None, ["2330"])
            self.assertEqual(result.columns.tolist(), ["id", "2026-11-09"])

    def test_missing_template_explains_date_option(self):
        with self.assertRaisesRegex(ValueError, "Specify --date"):
            prediction_template(Path("does-not-exist.csv"), None, ["2330"])

    def test_invalid_date_is_rejected(self):
        with self.assertRaises(ValueError):
            prediction_template(Path("unused.csv"), "2026-02-30", ["2330"])


if __name__ == "__main__":
    unittest.main()
