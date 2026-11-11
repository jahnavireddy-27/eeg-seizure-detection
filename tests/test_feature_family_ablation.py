import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import feature_family_ablation as ablation  # noqa: E402
import train_random_forest as baseline  # noqa: E402


def synthetic_frame() -> pd.DataFrame:
    rows = []
    recording_labels = {
        "chb01_02": [0] * 6,
        "chb01_06": [0] * 6,
        "chb01_16": [1, 1, 1, 0, 0, 0],
        "chb01_18": [1, 1, 1, 0, 0, 0],
    }
    for recording_id, labels in recording_labels.items():
        for index, label in enumerate(labels):
            time_value = float(label + index / 100)
            frequency_value = float(2 * label + index / 10)
            wavelet_value = float(3 * label + index / 5)
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
                    "channel_00__C3-P3__mean": time_value,
                    "channel_00__C3-P3__maximum": time_value + 1,
                    "channel_01__T8-P8__mean": time_value,
                    "channel_00__C3-P3__dominant_frequency_hz": frequency_value,
                    "channel_00__C3-P3__band_power_alpha": frequency_value + 1,
                    "channel_00__C3-P3__wavelet_A3_energy": wavelet_value,
                    "channel_01__T8-P8__wavelet_A3_energy": wavelet_value,
                }
            )
    return pd.DataFrame(rows)


class FeatureFamilySelectionTests(unittest.TestCase):
    def test_feature_family_selection_uses_feature_suffixes(self):
        frame = synthetic_frame()
        _, signal_columns = baseline.load_feature_matrix_from_frame(frame)
        selected = {
            family: ablation.select_feature_family_columns(signal_columns, family)
            for family in ("time", "frequency", "wavelet")
        }
        self.assertEqual([len(selected[name]) for name in selected], [3, 2, 2])
        self.assertEqual(
            set().union(*(set(columns) for columns in selected.values())),
            set(signal_columns),
        )
        self.assertEqual(
            len(sum((columns for columns in selected.values()), [])),
            len(signal_columns),
        )
        self.assertEqual(
            ablation.select_feature_family_columns(signal_columns, "all"),
            signal_columns,
        )
        with self.assertRaisesRegex(ValueError, "family must be one of"):
            ablation.select_feature_family_columns(signal_columns, "metadata")

    def test_unknown_channel_feature_name_fails_instead_of_being_misclassified(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized feature family"):
            ablation.select_feature_family_columns(
                ["channel_00__C3-P3__unexpected"], "all"
            )


class FeatureFamilyExperimentTests(unittest.TestCase):
    def test_duplicate_handling_is_training_only_and_family_specific(self):
        frame = synthetic_frame()
        feature_columns = [
            column
            for column in frame
            if column.startswith("channel_")
        ]
        folds = baseline.make_recording_folds(frame)
        fold = folds[0]
        for family in ("time", "frequency", "wavelet"):
            family_columns = ablation.select_feature_family_columns(
                feature_columns, family
            )
            train = frame.iloc[fold.train_indices][family_columns]
            retained, excluded = baseline.deduplicate_training_features(train)
            self.assertEqual(len(retained) + len(excluded), len(family_columns))
            self.assertTrue(all(name in family_columns for name, _ in excluded))
            self.assertTrue(all(name in family_columns for _, name in excluded))
        time_columns = ablation.select_feature_family_columns(feature_columns, "time")
        _, time_excluded = baseline.deduplicate_training_features(
            frame.iloc[fold.train_indices][time_columns]
        )
        self.assertEqual(
            time_excluded,
            [("channel_01__T8-P8__mean", "channel_00__C3-P3__mean")],
        )

    def test_complete_run_preserves_folds_and_writes_consistent_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_path = root / "features.csv"
            baseline_dir = root / "baseline"
            output_dir = root / "ablation"
            frame = synthetic_frame()
            frame.to_csv(feature_path, index=False)
            baseline.run_baseline(
                feature_path,
                baseline_dir,
                baseline.ForestConfiguration(n_estimators=7),
            )
            original_config = (baseline_dir / "run_configuration.json").read_bytes()
            original_predictions = {
                fold: (baseline_dir / fold / "predictions.csv").read_bytes()
                for fold in ("fold_1", "fold_2")
            }

            results = ablation.run_ablation(feature_path, baseline_dir, output_dir)

            self.assertEqual(len(results), 8)
            self.assertEqual(
                set(results["experiment"]),
                {"time", "frequency", "wavelet", ablation.COMBINED_REFERENCE},
            )
            reference_durations = results.loc[
                results["experiment"] == ablation.COMBINED_REFERENCE,
                "training_duration_seconds",
            ]
            self.assertTrue(reference_durations.isna().all())
            for row in results.itertuples(index=False):
                self.assertEqual(row.test_windows, 6)
                self.assertEqual(row.train_windows, 18)
                self.assertGreater(row.train_seizure, 0)
                self.assertGreater(row.train_non_seizure, 0)
                self.assertGreater(row.test_seizure, 0)
                self.assertGreater(row.test_non_seizure, 0)
                self.assertNotIn(row.test_recording, json.loads(row.train_recordings))
                fold_dir = output_dir / row.experiment / row.fold
                self.assertTrue((fold_dir / "confusion_matrix.png").is_file())
                self.assertTrue((fold_dir / "roc_curve.png").is_file())
                self.assertTrue((fold_dir / "precision_recall_curve.png").is_file())
                if row.experiment != ablation.COMBINED_REFERENCE:
                    self.assertTrue(
                        (fold_dir / "excluded_duplicate_features.csv").is_file()
                    )

            self.assertEqual(
                original_config,
                (baseline_dir / "run_configuration.json").read_bytes(),
            )
            for fold, original in original_predictions.items():
                self.assertEqual(
                    original,
                    (baseline_dir / fold / "predictions.csv").read_bytes(),
                )

            with (output_dir / "per_fold_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                metric_rows = list(csv.DictReader(stream))
            with (output_dir / "experiment_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                summary_rows = list(csv.DictReader(stream))
            self.assertEqual(len(metric_rows), 8)
            self.assertEqual(len(summary_rows), 4)
            for row in metric_rows:
                self.assertEqual(row["train_windows"], "18")
                self.assertEqual(row["test_windows"], "6")
                self.assertTrue(row["confusion_matrix"])
            self.assertTrue((output_dir / "report.md").is_file())
            config = json.loads((output_dir / "run_configuration.json").read_text())
            self.assertEqual(config["feature_selection"], "none")
            self.assertEqual(config["threshold_rule"], "positive when P(seizure) >= 0.5")
            self.assertIn("existing saved", config["combined_reference"])


if __name__ == "__main__":
    unittest.main()
