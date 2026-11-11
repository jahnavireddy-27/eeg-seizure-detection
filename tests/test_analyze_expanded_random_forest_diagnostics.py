import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyze_expanded_random_forest_diagnostics as diagnostics  # noqa: E402


class DiagnosticMetricTests(unittest.TestCase):
    def test_fixed_threshold_metrics_and_false_positives_per_hour(self):
        result = diagnostics.calculate_diagnostic_metrics(
            [1, 1, 0, 0], [1, 0, 1, 0], duration_seconds=1800.0
        )
        self.assertEqual(
            (
                result["true_negative"],
                result["false_positive"],
                result["false_negative"],
                result["true_positive"],
            ),
            (1, 1, 1, 1),
        )
        self.assertEqual(result["sensitivity"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["false_positive_windows_per_recording_hour"], 2.0)

    def test_undefined_class_metrics_are_none(self):
        result = diagnostics.calculate_diagnostic_metrics(
            [0, 0, 0], [0, 1, 0], duration_seconds=3600.0
        )
        self.assertIsNone(result["sensitivity"])
        self.assertEqual(result["specificity"], 2 / 3)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["false_positive_windows_per_recording_hour"], 1.0)


class SeizureEventMatchingTests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "event_id": "chb02_16_event_01",
            "patient_id": "chb02",
            "recording_id": "chb02_16",
            "relative_path": "chb02/chb02_16.edf",
            "event_start_seconds": 10.0,
            "event_end_seconds": 20.0,
            "event_duration_seconds": 10.0,
        }

    def predictions(self, rows):
        return pd.DataFrame(
            [
                {
                    "patient_id": "chb02",
                    "recording_id": "chb02_16",
                    "relative_path": "chb02/chb02_16.edf",
                    "start_time_seconds": start,
                    "end_time_seconds": end,
                    "y_true": truth,
                    "y_pred": prediction,
                }
                for start, end, truth, prediction in rows
            ]
        )

    def test_event_is_detected_by_any_overlapping_positive_prediction(self):
        result = diagnostics.match_seizure_events(
            [self.event], self.predictions([(7.5, 12.5, 1, 0), (17.5, 22.5, 1, 1)])
        )[0]
        self.assertTrue(result["detected"])
        self.assertEqual(result["overlapping_ground_truth_windows"], 2)
        self.assertEqual(result["false_negative_windows"], 1)
        self.assertIn("positive-duration overlap", result["event_detection_definition"])

    def test_boundary_touch_without_positive_duration_does_not_detect_event(self):
        result = diagnostics.match_seizure_events(
            [self.event], self.predictions([(20.0, 25.0, 0, 1)])
        )[0]
        self.assertFalse(result["detected"])
        self.assertEqual(result["overlapping_ground_truth_windows"], 0)
        self.assertFalse(diagnostics._overlaps(20.0, 25.0, 10.0, 20.0))


if __name__ == "__main__":
    unittest.main()
