import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.skillsbench.fractional_reward import fractional_reward


class FractionalRewardTests(unittest.TestCase):
    def test_counts_passed_cases_over_all_collected_cases(self):
        report = {
            "results": {
                "tests": [
                    {"status": "passed"},
                    {"status": "failed"},
                    {"status": "skipped"},
                    {"status": "passed"},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "ctrf.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            reward, counts = fractional_reward(report_path)

        self.assertEqual(reward, 0.5)
        self.assertEqual(counts["passed"], 2)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["skipped"], 1)

    def test_rejects_empty_test_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "ctrf.json"
            report_path.write_text(json.dumps({"results": {"tests": []}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                fractional_reward(report_path)


if __name__ == "__main__":
    unittest.main()
