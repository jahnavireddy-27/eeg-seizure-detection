import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_random_forest as baseline  # noqa: E402


def synthetic_frame() -> pd.DataFrame:
    rows = []
    label_counts = {
        "chb01_02": (0, 6),
        "chb01_06": (0, 6),
        "chb01_16": (2, 4),
        "chb01_18": (3, 3),
    }
    for recording_id, (positives, negatives) in label_counts.items():
        labels = [1] * positives + [0] * negatives
        for index, label in enumerate(labels):
            rows.append(
                {
                    "patient_id": "chb01",
                    "recording_id": recording_id,
                    "recording_filename": f"{recording_id}.edf",
                    "window_index": index,
                    "start_time_seconds": index * 2.5,
                    "end_time_seconds": index * 2.5 + 5.0,
                    "seizure_label": "seizure" if label else "non_seizure",
                    "label_policy": "any_overlap",
                    "sampling_frequency_hz": 256,
                    "window_samples": 1280,
                    "feature_config_json": "{}",
                    "channel_00__C3-P3__mean": float(label),
                    "channel_00__C3-P3__maximum": float(label),
                    "channel_01__T8-P8__mean": float(index),
                }
            )
    return pd.DataFrame(rows)


class PredictorAndFoldTests(unittest.TestCase):
    def test_metadata_is_excluded_and_unexpected_columns_fail(self):
        frame = synthetic_frame()
        _, feature_columns = baseline.load_feature_matrix_from_frame(frame)
        self.assertEqual(
            feature_columns,
            [
                "channel_00__C3-P3__mean",
                "channel_00__C3-P3__maximum",
                "channel_01__T8-P8__mean",
            ],
        )
        self.assertFalse(any(column in feature_columns for column in baseline.REQUIRED_METADATA_COLUMNS))
        frame["unexpected_timestamp_copy"] = frame["start_time_seconds"]
        with self.assertRaisesRegex(ValueError, "Unexpected non-signal"):
            baseline.load_feature_matrix_from_frame(frame)
        frame = synthetic_frame()
        frame["channel_metadata"] = frame["recording_id"]
        with self.assertRaisesRegex(ValueError, "Unexpected non-signal"):
            baseline.load_feature_matrix_from_frame(frame)

    def test_p3_metadata_and_channel_aligned_features_remain_separated(self):
        frame = synthetic_frame()
        frame["relative_path"] = [
            f"chb01/{recording}.edf" for recording in frame["recording_id"]
        ]
        frame["annotation_status"] = "confirmed_no_seizure"
        frame["channel__C3-P3__instance_01__mean"] = 1.0
        _, feature_columns = baseline.load_feature_matrix_from_frame(frame)
        self.assertIn("channel__C3-P3__instance_01__mean", feature_columns)
        self.assertNotIn("relative_path", feature_columns)
        self.assertNotIn("annotation_status", feature_columns)

    def test_exact_duplicate_removal_retains_first_column(self):
        frame = pd.DataFrame(
            {
                "channel_00__first": [1.0, 2.0, -0.0],
                "channel_01__duplicate": [1.0, 2.0, 0.0],
                "channel_02__different": [1.0, 2.0, 3.0],
            }
        )
        retained, excluded = baseline.deduplicate_training_features(frame)
        self.assertEqual(retained, ["channel_00__first", "channel_02__different"])
        self.assertEqual(
            excluded,
            [("channel_01__duplicate", "channel_00__first")],
        )

    def test_recording_groups_do_not_cross_fold_partitions(self):
        frame = synthetic_frame()
        folds = baseline.make_recording_folds(frame)
        self.assertEqual([fold.test_recording for fold in folds], ["chb01_16", "chb01_18"])
        recording_ids = frame["recording_id"].to_numpy()
        for fold in folds:
            train_groups = set(recording_ids[fold.train_indices])
            test_groups = set(recording_ids[fold.test_indices])
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(test_groups, {fold.test_recording})
            self.assertEqual(len(fold.test_indices), 6)
            self.assertTrue({"chb01_02", "chb01_06"} <= train_groups)

    def test_fold_class_counts_are_enforced(self):
        frame = synthetic_frame()
        folds = baseline.make_recording_folds(frame)
        counts = []
        labels = frame["seizure_label"].map(baseline.LABEL_TO_INT).to_numpy()
        for fold in folds:
            counts.append(
                (
                    np.bincount(labels[fold.train_indices], minlength=2).tolist(),
                    np.bincount(labels[fold.test_indices], minlength=2).tolist(),
                )
            )
        self.assertEqual(counts[0], ([15, 3], [4, 2]))
        self.assertEqual(counts[1], ([16, 2], [3, 3]))

    def test_missing_test_class_fails_without_dropping_fold(self):
        frame = synthetic_frame()
        frame.loc[
            (frame["recording_id"] == "chb01_16")
            & (frame["seizure_label"] == "seizure"),
            "seizure_label",
        ] = "non_seizure"
        with self.assertRaisesRegex(ValueError, "test partition lacks a class"):
            baseline.make_recording_folds(frame)

    def test_baseline_writes_predictions_metrics_and_duplicate_reports(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "features.csv"
            output_dir = root / "evaluation"
            frame.to_csv(matrix_path, index=False)
            results = baseline.run_baseline(
                matrix_path,
                output_dir,
                baseline.ForestConfiguration(n_estimators=5),
            )
            self.assertEqual(len(results), 2)
            for fold in ("fold_1", "fold_2"):
                fold_dir = output_dir / fold
                with (fold_dir / "predictions.csv").open(
                    newline="", encoding="utf-8"
                ) as file:
                    predictions = list(csv.DictReader(file))
                self.assertEqual(len(predictions), 6)
                self.assertEqual(
                    {row["recording_id"] for row in predictions},
                    {results[int(fold[-1]) - 1]["test_recording"]},
                )
                self.assertTrue((fold_dir / "metrics.json").is_file())
                self.assertTrue((fold_dir / "confusion_matrix.png").is_file())
                self.assertTrue(
                    (fold_dir / "excluded_duplicate_features.csv").is_file()
                )
            self.assertTrue((output_dir / "fold_comparison.csv").is_file())
            self.assertTrue((output_dir / "run_configuration.json").is_file())


class RandomForestAndMetricsTests(unittest.TestCase):
    def test_configuration_is_fixed_balanced_and_threshold_half(self):
        config = baseline.ForestConfiguration()
        model = baseline.build_random_forest(config)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_estimators, 300)
        self.assertEqual(config.probability_threshold, 0.5)
        with self.assertRaisesRegex(ValueError, "threshold at 0.5"):
            baseline.ForestConfiguration(probability_threshold=0.4).validate()

    def test_fixed_seed_is_reproducible_on_imbalanced_data(self):
        frame = synthetic_frame()
        x = frame[["channel_00__C3-P3__mean", "channel_01__T8-P8__mean"]]
        y = frame["seizure_label"].map(baseline.LABEL_TO_INT).to_numpy()
        configuration = baseline.ForestConfiguration(n_estimators=25)
        first = baseline.build_random_forest(configuration).fit(x, y)
        second = baseline.build_random_forest(configuration).fit(x, y)
        self.assertTrue(
            np.array_equal(
                first.predict_proba(x),
                second.predict_proba(x),
            )
        )
        self.assertEqual(first.class_weight, "balanced")

    def test_known_metric_values(self):
        y_true = [0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 1, 0, 1, 1, 0]
        probability = [0.1, 0.2, 0.8, 0.4, 0.7, 0.9, 0.6]
        result = baseline.calculate_metrics(y_true, y_pred, probability)
        self.assertEqual(result["confusion_matrix"]["values"], [[2, 1], [2, 2]])
        self.assertAlmostEqual(result["accuracy"], 4 / 7)
        self.assertAlmostEqual(result["balanced_accuracy"], (2 / 3 + 1 / 2) / 2)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall_sensitivity"], 1 / 2)
        self.assertAlmostEqual(result["specificity"], 2 / 3)
        self.assertAlmostEqual(result["f1_score"], 4 / 7)
        self.assertAlmostEqual(result["roc_auc"], 9 / 12)
        self.assertAlmostEqual(result["average_precision"], (1 + 2 / 3 + 3 / 4 + 4 / 5) / 4)

    def test_undefined_precision_and_f1_are_explicit_nulls(self):
        result = baseline.calculate_metrics(
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0.1, 0.2, 0.3, 0.4],
        )
        self.assertIsNone(result["precision"])
        self.assertEqual(result["f1_score"], 0.0)
        self.assertEqual(result["undefined_metrics"], ["precision"])


if __name__ == "__main__":
    unittest.main()
