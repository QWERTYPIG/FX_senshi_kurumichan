import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import prediction_template


class PredictionTemplateTests(unittest.TestCase):
    def test_explicit_date_uses_saved_stock_order(self):
        result = prediction_template(Path("not-used.csv"), "2026-11-19", ["2330", "2317"])
        self.assertEqual(result.columns.tolist(), ["id", "2026-11-19"])
        self.assertEqual(result["id"].tolist(), ["2330", "2317"])

    def test_template_is_read_and_invalid_date_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "template.csv"
            pd.DataFrame({"id": ["2330"], "2026-11-19": [0]}).to_csv(path, index=False)
            self.assertEqual(prediction_template(path, None, ["2330"])["id"].tolist(), ["2330"])
        with self.assertRaises(ValueError):
            prediction_template(Path("unused.csv"), "2026-02-30", ["2330"])


if __name__ == "__main__":
    unittest.main()
