import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import segment_windows as segmentation  # noqa: E402


class WindowConfigurationTests(unittest.TestCase):
    def test_default_window_size_and_stride_at_256_hz(self):
        configuration = segmentation.WindowConfiguration()
        self.assertEqual(configuration.sample_counts(256), (1280, 640, 640))

    def test_full_hour_window_count_and_endpoints(self):
        windows = list(
            segmentation.generate_window_records(
                patient_id="chb01",
                recording_filename="chb01_02.edf",
                sample_count=3600 * 256,
                sampling_frequency_hz=256,
                channel_labels=["FP1-F7", "T8-P8", "T8-P8"],
                annotation=segmentation.RecordingAnnotation(
                    "confirmed_no_seizure", ()
                ),
            )
        )
        self.assertEqual(len(windows), 1439)
        self.assertEqual((windows[0].start_sample, windows[0].end_sample_exclusive), (0, 1280))
        self.assertEqual(
            (windows[-1].start_sample, windows[-1].end_sample_exclusive),
            (920320, 921600),
        )
        self.assertEqual(
            (windows[-1].start_time_seconds, windows[-1].end_time_seconds),
            (3595.0, 3600.0),
        )
        self.assertEqual(windows[-1].window_index, 1438)

    def test_configurable_duration_and_overlap(self):
        config = segmentation.WindowConfiguration(
            window_seconds=2.0, overlap_fraction=0.25
        )
        self.assertEqual(config.sample_counts(256), (512, 128, 384))


class SeizureLabellingTests(unittest.TestCase):
    def setUp(self):
        self.annotation = segmentation.RecordingAnnotation(
            "confirmed_seizure",
            (segmentation.SeizureInterval(10.0, 20.0),),
        )

    def label(self, start, end, policy="any_overlap"):
        return segmentation.label_window(start, end, self.annotation, policy)

    def test_half_open_onset_and_offset_boundaries(self):
        self.assertEqual(self.label(5.0, 10.0), ("non_seizure", 0.0))
        self.assertEqual(self.label(10.0, 15.0), ("seizure", 5.0))
        self.assertEqual(self.label(20.0, 25.0), ("non_seizure", 0.0))

    def test_windows_entirely_before_during_and_after(self):
        self.assertEqual(self.label(0.0, 5.0)[0], "non_seizure")
        self.assertEqual(self.label(12.0, 17.0)[0], "seizure")
        self.assertEqual(self.label(25.0, 30.0)[0], "non_seizure")

    def test_partial_overlap_and_policy_differences(self):
        self.assertEqual(self.label(7.5, 12.5, "any_overlap"), ("seizure", 2.5))
        self.assertEqual(self.label(7.5, 12.5, "majority_overlap"), ("non_seizure", 2.5))
        self.assertEqual(self.label(7.5, 12.5, "center_in_seizure"), ("seizure", 2.5))
        self.assertEqual(self.label(7.5, 12.5, "majority_overlap")[0], "non_seizure")
        self.assertEqual(self.label(7.5, 12.5, "any_overlap")[0], "seizure")

    def test_majority_is_strictly_more_than_half(self):
        self.assertEqual(self.label(7.5, 12.5, "majority_overlap")[0], "non_seizure")
        self.assertEqual(self.label(8.0, 13.0, "majority_overlap")[0], "seizure")

    def test_center_policy_uses_start_inclusive_end_exclusive_interval(self):
        self.assertEqual(self.label(7.5, 12.5, "center_in_seizure")[0], "seizure")
        self.assertEqual(self.label(17.5, 22.5, "center_in_seizure")[0], "non_seizure")

    def test_multiple_seizure_intervals_and_overlap_duration(self):
        annotation = segmentation.RecordingAnnotation(
            "confirmed_seizure",
            (
                segmentation.SeizureInterval(10.0, 12.0),
                segmentation.SeizureInterval(15.0, 17.0),
            ),
        )
        self.assertEqual(
            segmentation.label_window(9.0, 18.0, annotation, "any_overlap"),
            ("seizure", 4.0),
        )

    def test_missing_annotation_is_unknown_not_negative(self):
        annotation = segmentation.RecordingAnnotation("unknown", (), "missing audit row")
        self.assertEqual(
            segmentation.label_window(0.0, 5.0, annotation, "any_overlap"),
            ("unknown", None),
        )

    def test_invalid_intervals_and_too_short_recording(self):
        with self.assertRaises(ValueError):
            list(
                segmentation.generate_window_records(
                    patient_id="chb01",
                    recording_filename="bad.edf",
                    sample_count=2560,
                    sampling_frequency_hz=256,
                    channel_labels=["C3-P3"],
                    annotation=segmentation.RecordingAnnotation(
                        "confirmed_seizure",
                        (segmentation.SeizureInterval(-1, 2),),
                    ),
                )
            )
        with self.assertRaises(ValueError):
            list(
                segmentation.generate_window_records(
                    patient_id="chb01",
                    recording_filename="bad.edf",
                    sample_count=2560,
                    sampling_frequency_hz=256,
                    channel_labels=["C3-P3"],
                    annotation=segmentation.RecordingAnnotation(
                        "confirmed_seizure",
                        (segmentation.SeizureInterval(10, 9),),
                    ),
                ),
            )
        with self.assertRaises(ValueError):
            list(
                segmentation.generate_window_records(
                    patient_id="chb01",
                    recording_filename="bad.edf",
                    sample_count=2560,
                    sampling_frequency_hz=256,
                    channel_labels=["C3-P3"],
                    annotation=segmentation.RecordingAnnotation(
                        "confirmed_seizure",
                        (
                            segmentation.SeizureInterval(5, 10),
                            segmentation.SeizureInterval(9, 12),
                        ),
                    ),
                ),
            )
        windows = list(
            segmentation.generate_window_records(
                patient_id="chb01",
                recording_filename="short.edf",
                sample_count=1279,
                sampling_frequency_hz=256,
                channel_labels=["C3-P3"],
                annotation=segmentation.RecordingAnnotation(
                    "confirmed_no_seizure", ()
                ),
            )
        )
        self.assertEqual(windows, [])


class AnnotationAndManifestTests(unittest.TestCase):
    def test_window_manifest_preserves_relative_recording_path(self):
        windows = segmentation.generate_window_records(
            patient_id="chb02",
            recording_filename="same.edf",
            relative_path="chb02/session/same.edf",
            sample_count=1280,
            sampling_frequency_hz=256,
            channel_labels=["C3-P3"],
            annotation=segmentation.RecordingAnnotation(
                "confirmed_no_seizure", ()
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.csv"
            segmentation.write_manifest(path, windows)
            with path.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
        self.assertEqual(row["relative_path"], "chb02/session/same.edf")

    def test_corpus_manifest_annotations_use_composite_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.csv"
            fields = [
                "patient_id",
                "recording_id",
                "relative_path",
                "seizure_status",
                "seizure_intervals",
                "annotation_agreement",
                "annotation_validation_status",
                "recording_duration_seconds",
            ]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "patient_id": "chb01",
                            "recording_id": "recording",
                            "relative_path": "chb01/recording.edf",
                            "seizure_status": "seizures_annotated",
                            "seizure_intervals": '[{"start_seconds":1.125,"end_seconds":2.875}]',
                            "annotation_agreement": "not_available",
                            "annotation_validation_status": "absent",
                            "recording_duration_seconds": "120",
                        },
                        {
                            "patient_id": "chb02",
                            "recording_id": "recording",
                            "relative_path": "chb02/recording.edf",
                            "seizure_status": "no_seizures_annotated",
                            "seizure_intervals": "[]",
                            "annotation_agreement": "not_available",
                            "annotation_validation_status": "absent",
                            "recording_duration_seconds": "120",
                        },
                        {
                            "patient_id": "chb03",
                            "recording_id": "recording",
                            "relative_path": "chb03/recording.edf",
                            "seizure_status": "seizures_annotated",
                            "seizure_intervals": '[{"start_seconds":1,"end_seconds":2}]',
                            "annotation_agreement": "disagreement",
                            "annotation_validation_status": "mismatch",
                            "recording_duration_seconds": "120",
                        },
                    ]
                )

            loaded = segmentation.load_validated_annotations(path)
            patient_one = loaded[("chb01", "chb01/recording.edf")]
            patient_two = loaded[("chb02", "chb02/recording.edf")]
            patient_three = loaded[("chb03", "chb03/recording.edf")]
            selected = segmentation.load_validated_annotations(path, "chb02")

        self.assertEqual(patient_one.status, "confirmed_seizure")
        self.assertEqual(patient_one.intervals[0].start_seconds, 1.125)
        self.assertEqual(patient_two.status, "confirmed_no_seizure")
        self.assertEqual(patient_three.status, "unknown")
        self.assertEqual(set(selected), {("chb02", "chb02/recording.edf")})

    def test_validated_audit_distinguishes_confirmed_none_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            fields = [
                "recording",
                "downloaded",
                "contains_seizure",
                "summary_seizure_count",
                "seizure_intervals",
                "validation_status",
                "validation_issues",
            ]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "recording": "no.edf",
                        "downloaded": "true",
                        "contains_seizure": "false",
                        "summary_seizure_count": "0",
                        "seizure_intervals": "[]",
                        "validation_status": "PASS",
                    }
                )
                writer.writerow(
                    {
                        "recording": "yes.edf",
                        "downloaded": "true",
                        "contains_seizure": "true",
                        "summary_seizure_count": "1",
                        "seizure_intervals": json.dumps(
                            [{"start_seconds": 1, "end_seconds": 2}]
                        ),
                        "validation_status": "PASS",
                    }
                )
                writer.writerow(
                    {
                        "recording": "bad.edf",
                        "downloaded": "true",
                        "contains_seizure": "true",
                        "summary_seizure_count": "1",
                        "seizure_intervals": "[]",
                        "validation_status": "REVIEW",
                        "validation_issues": "conflict",
                    }
                )
            loaded = segmentation.load_validated_annotations(path)
            self.assertEqual(loaded["no.edf"].status, "confirmed_no_seizure")
            self.assertEqual(loaded["yes.edf"].status, "confirmed_seizure")
            self.assertEqual(loaded["bad.edf"].status, "unknown")

    def test_duplicate_channel_labels_are_preserved_in_manifest(self):
        windows = list(
            segmentation.generate_window_records(
                patient_id="chb01",
                recording_filename="chb01_16.edf",
                sample_count=1280,
                sampling_frequency_hz=256,
                channel_labels=["FP1-F7", "T8-P8", "T8-P8"],
                annotation=segmentation.RecordingAnnotation(
                    "confirmed_seizure",
                    (segmentation.SeizureInterval(0, 5),),
                ),
            )
        )
        self.assertEqual(windows[0].channel_count, 3)
        self.assertEqual(windows[0].channel_labels.count("T8-P8"), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            segmentation.write_manifest(path, windows)
            with path.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(
                json.loads(row["channel_labels"]),
                ["FP1-F7", "T8-P8", "T8-P8"],
            )

    def test_positive_windows_match_verified_seizure_intervals(self):
        expected = {
            "chb01_16.edf": (1015.0, 1066.0, 22),
            "chb01_18.edf": (1720.0, 1810.0, 37),
        }
        for recording, (onset, offset, expected_positive_count) in expected.items():
            windows = list(
                segmentation.generate_window_records(
                    patient_id="chb01",
                    recording_filename=recording,
                    sample_count=3600 * 256,
                    sampling_frequency_hz=256,
                    channel_labels=["FP1-F7"],
                    annotation=segmentation.RecordingAnnotation(
                        "confirmed_seizure",
                        (segmentation.SeizureInterval(onset, offset),),
                    ),
                )
            )
            positive = [window for window in windows if window.seizure_label == "seizure"]
            self.assertEqual(len(windows), 1439)
            self.assertEqual(len(positive), expected_positive_count)
            self.assertTrue(
                all(
                    window.end_time_seconds > onset
                    and window.start_time_seconds < offset
                    for window in positive
                )
            )
            self.assertTrue(
                all(
                    window.seizure_overlap_seconds > 0
                    for window in positive
                )
            )


if __name__ == "__main__":
    unittest.main()
