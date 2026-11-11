import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_expanded_random_forest as expanded  # noqa: E402
import train_random_forest as baseline  # noqa: E402


def synthetic_expanded_frame() -> pd.DataFrame:
    rows = []
    seizure_recordings = {
        "chb01": {"chb01_16", "chb01_18"},
        "chb02": {"chb02_16", "chb02_16+", "chb02_19"},
        "chb03": {"chb03_01", "chb03_02", "chb03_34"},
    }
    recordings = []
    for patient, seizure_ids in seizure_recordings.items():
        count = 7 if patient == "chb01" else 6
        ids = list(sorted(seizure_ids)) + [f"{patient}_neg_{i:02d}" for i in range(count - len(seizure_ids))]
        recordings.extend((patient, recording) for recording in ids)

    for group_index, (patient, recording_id) in enumerate(recordings):
        positive_recording = recording_id in seizure_recordings[patient]
        labels = [1, 0, 0, 0] if positive_recording else [0, 0, 0, 0]
        for window_index, label in enumerate(labels):
            value = float(group_index + window_index)
            rows.append(
                {
                    "patient_id": patient,
                    "recording_id": recording_id,
                    "relative_path": f"{patient}/{recording_id}.edf",
                    "recording_filename": f"{recording_id}.edf",
                    "window_index": window_index,
                    "start_time_seconds": window_index * 2.5,
                    "end_time_seconds": window_index * 2.5 + 5.0,
                    "seizure_label": "seizure" if label else "non_seizure",
                    "annotation_status": (
                        "confirmed_seizure" if positive_recording else "confirmed_no_seizure"
                    ),
                    "label_policy": "any_overlap",
                    "sampling_frequency_hz": 256,
                    "window_samples": 1280,
                    "feature_config_json": "{}",
                    "channel__C3-P3__instance_01__mean": float(label) + value,
                    "channel__C3-P3__instance_01__maximum": float(label) + value,
                    "channel__F3-C3__instance_01__mean": float((group_index * 3 + window_index) % 7),
                }
            )
    return pd.DataFrame(rows)


class RecordingFoldTests(unittest.TestCase):
    def test_four_fold_integrity_and_recording_level_stratification(self):
        frame = synthetic_expanded_frame()
        folds = expanded.make_recording_grouped_folds(frame)
        labels = frame["seizure_label"].map(baseline.LABEL_TO_INT).to_numpy()
        identities = [expanded._recording_identity(row) for _, row in frame.iterrows()]
        appearances = Counter()
        self.assertEqual(len(folds), 4)
        for fold in folds:
            train_groups = {identities[index] for index in fold.train_indices}
            test_groups = {identities[index] for index in fold.test_indices}
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(test_groups, set(fold.test_recordings))
            self.assertEqual(
                set(labels[fold.train_indices]),
                {0, 1},
            )
            self.assertEqual(set(labels[fold.test_indices]), {0, 1})
            self.assertEqual(
                sum(
                    bool((labels[np.asarray([i for i, identity in enumerate(identities) if identity == group])] == 1).any())
                    for group in fold.test_recordings
                ),
                2,
            )
            appearances.update(fold.test_recordings)
        self.assertEqual(set(appearances.values()), {1})
        self.assertEqual(len(appearances), 19)

    def test_fold_assignment_is_deterministic(self):
        frame = synthetic_expanded_frame()
        first = expanded.make_recording_grouped_folds(frame)
        second = expanded.make_recording_grouped_folds(frame)
        self.assertEqual(
            [fold.test_recordings for fold in first],
            [fold.test_recordings for fold in second],
        )

    def test_split_fails_if_each_fold_cannot_receive_both_recording_classes(self):
        frame = synthetic_expanded_frame()
        positive_groups = set(
            frame.loc[frame["seizure_label"] == "seizure", "recording_id"]
        )
        keep = frame["recording_id"].isin(positive_groups | {"chb01_neg_00", "chb02_neg_00", "chb03_neg_00"})
        with self.assertRaisesRegex(ValueError, "Cannot create 4 folds"):
            expanded.make_recording_grouped_folds(frame.loc[keep].reset_index(drop=True))

    def test_out_of_fold_prediction_coverage_and_separate_outputs(self):
        frame = synthetic_expanded_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "assembled.csv"
            output_path = root / "expanded_eval"
            frame.to_csv(matrix_path, index=False)
            result = expanded.run_expanded_baseline(
                matrix_path,
                output_path,
                configuration=baseline.ForestConfiguration(n_estimators=5),
            )
            with (output_path / "out_of_fold_predictions.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), len(frame))
            expected_windows = {
                (
                    expanded._recording_identity(row),
                    str(int(row["window_index"])),
                )
                for _, row in frame.iterrows()
            }
            predicted_windows = {
                (
                    (row["patient_id"], row["recording_id"], row["relative_path"]),
                    row["window_index"],
                )
                for row in predictions
            }
            self.assertEqual(len(predicted_windows), len(predictions))
            self.assertEqual(predicted_windows, expected_windows)
            self.assertEqual(
                {(row["patient_id"], row["recording_id"], row["relative_path"]) for row in predictions},
                {expanded._recording_identity(row) for _, row in frame.iterrows()},
            )
            self.assertTrue(all(row["test_fold"] for row in predictions))
            self.assertEqual(len(result["folds"]), 4)
            self.assertEqual(result["pooled"]["windows"], len(frame))
            self.assertTrue((output_path / "recording_patient_diagnostics.csv").is_file())
            self.assertTrue((output_path / "pooled_confusion_matrix.png").is_file())
            with (output_path / "recording_patient_diagnostics.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                diagnostic = next(csv.DictReader(stream))
            self.assertEqual(
                set(json.loads(diagnostic["confusion_matrix"])),
                {"labels", "values"},
            )


if __name__ == "__main__":
    unittest.main()
