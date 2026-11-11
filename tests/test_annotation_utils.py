import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotation_utils import (  # noqa: E402
    compare_summary_sidecar,
    parse_patient_summary,
    validate_annotations,
)
import inspect_edf  # noqa: E402
import annotation_audit  # noqa: E402


class SharedAnnotationParsingTests(unittest.TestCase):
    def test_integer_decimal_and_multiple_intervals_keep_precision(self):
        content = """Data Sampling Rate: 256 Hz
File Name: chb01_01.edf
Number of Seizures in File: 3
Seizure Start Time: 10 seconds
Seizure End Time: 20 seconds
Seizure Start Time: 30.125 seconds
Seizure End Time: 40.875 seconds
Seizure Start Time: 50.0004 seconds
Seizure End Time: 60.0009 seconds
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chb01-summary.txt"
            path.write_text(content, encoding="utf-8")
            recordings, rate = parse_patient_summary(path)

        self.assertEqual(rate, 256.0)
        self.assertEqual(len(recordings), 1)
        self.assertEqual(recordings[0]["declared_count"], 3)
        self.assertEqual(
            [interval["start_seconds"] for interval in recordings[0]["intervals"]],
            [10.0, 30.125, 50.0004],
        )
        self.assertEqual(
            recordings[0]["intervals"][2]["duration_seconds"],
            60.0009 - 50.0004,
        )

    def test_inspector_uses_shared_decimal_parser(self):
        content = """File Name: chb01_01.edf
Number of Seizures in File: 1
Seizure Start Time: 1.125 seconds
Seizure End Time: 2.875 seconds
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chb01-summary.txt"
            path.write_text(content, encoding="utf-8")
            record = inspect_edf.parse_patient_summary(path)[0]

        self.assertEqual(record["starts"], [1.125])
        self.assertEqual(record["ends"], [2.875])
        self.assertEqual(record["intervals"][0]["duration_seconds"], 1.75)

    def test_no_seizure_summary_is_distinct_from_missing_information(self):
        for expected, count, interval_lines in [
            ("no_seizure", 0, ""),
            ("missing", None, ""),
            ("incomplete", 1, "Seizure Start Time: 5.5 seconds\n"),
        ]:
            count_line = (
                ""
                if count is None
                else f"Number of Seizures in File: {count}\n"
            )
            text = (
                "File Name: chb01_01.edf\n"
                + count_line
                + interval_lines
            )
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "chb01-summary.txt"
                path.write_text(text, encoding="utf-8")
                records, _ = parse_patient_summary(path)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["declared_count"], count)
                if expected == "no_seizure":
                    self.assertEqual(records[0]["intervals"], [])
                elif expected == "incomplete":
                    self.assertIsNone(records[0]["intervals"][0]["end_seconds"])

    def test_summary_sidecar_agreement_disagreement_and_unknown(self):
        summary = [
            {"start_seconds": 10.0, "end_seconds": 20.0, "duration_seconds": 10.0}
        ]
        matching = [
            {
                "start_seconds": 10.0 + 1 / 256,
                "end_seconds": 20.0,
                "duration_seconds": 10.0 - 1 / 256,
            }
        ]
        different = [
            {
                "start_seconds": 12.0,
                "end_seconds": 22.0,
                "duration_seconds": 10.0,
            }
        ]

        self.assertEqual(compare_summary_sidecar(summary, matching, 256), "agreement")
        self.assertEqual(
            compare_summary_sidecar(summary, different, 256), "disagreement"
        )
        self.assertEqual(
            compare_summary_sidecar([], different, 256), "unavailable"
        )
        self.assertEqual(compare_summary_sidecar([], [], 256, summary_declared_count=0), "agreement")

    def test_invalid_interval_validation_reports_all_requested_edge_cases(self):
        intervals = [
            {"start_seconds": -1.0, "end_seconds": 1.0},
            {"start_seconds": 8.0, "end_seconds": 7.0},
            {"start_seconds": 9.0, "end_seconds": 12.0},
        ]
        issues = validate_annotations(
            "recording.edf",
            [
                {
                    "filename": "recording.edf",
                    "declared_count": 3,
                    "intervals": intervals,
                }
            ],
            intervals,
            [3],
            10.0,
            None,
            256.0,
        )

        self.assertTrue(any("NEGATIVE_START_TIME" in issue for issue in issues))
        self.assertTrue(any("END_BEFORE_START" in issue for issue in issues))
        self.assertTrue(any("INTERVAL_EXCEEDS_EDF_DURATION" in issue for issue in issues))

    def test_invalid_sidecar_sample_interval_is_reported(self):
        issues = validate_annotations(
            "recording.edf",
            [],
            [],
            [],
            10.0,
            {
                "error": None,
                "intervals": [
                    {
                        "start_sample": -1,
                        "end_sample": 256,
                        "start_seconds": -1 / 256,
                        "end_seconds": 1.0,
                        "duration_seconds": 1 + 1 / 256,
                    },
                    {
                        "start_sample": 512,
                        "end_sample": 256,
                        "start_seconds": 2.0,
                        "end_seconds": 1.0,
                        "duration_seconds": -1.0,
                    },
                    {
                        "start_sample": 2560,
                        "end_sample": 3072,
                        "start_seconds": 10.0,
                        "end_seconds": 12.0,
                        "duration_seconds": 2.0,
                    },
                ],
            },
            256.0,
        )

        self.assertTrue(any("NEGATIVE_START_TIME: WFDB" in issue for issue in issues))
        self.assertTrue(any("END_BEFORE_START: WFDB" in issue for issue in issues))
        self.assertTrue(
            any("INTERVAL_EXCEEDS_EDF_DURATION: WFDB" in issue for issue in issues)
        )

    def test_audit_marks_duplicate_patient_paths_ambiguous(self):
        class FakeRaw:
            info = {"sfreq": 256.0}
            n_times = 2560

        fake_mne = types.SimpleNamespace(
            io=types.SimpleNamespace(read_raw_edf=lambda *_args, **_kwargs: FakeRaw())
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chb01-summary.txt").write_text(
                "File Name: chb01_01.edf\nNumber of Seizures in File: 0\n",
                encoding="utf-8",
            )
            for subdirectory in ("copy_a", "copy_b"):
                patient_dir = root / subdirectory
                patient_dir.mkdir()
                (patient_dir / "chb01_01.edf").write_bytes(b"synthetic EDF")
            output_dir = root / "audit"
            with (
                patch.dict(sys.modules, {"mne": fake_mne}),
                patch.object(annotation_audit.importlib.util, "find_spec", return_value=object()),
            ):
                csv_path, _, _ = annotation_audit.run_audit(
                    root, "chb01", output_dir
                )

            with csv_path.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))

        self.assertEqual(row["patient_id"], "chb01")
        self.assertEqual(row["recording_id"], "chb01_01")
        self.assertEqual(row["downloaded"], "false")
        self.assertEqual(row["validation_status"], "REVIEW")
        self.assertIn("AMBIGUOUS_LOCAL_EDF_MATCH", row["validation_issues"])


if __name__ == "__main__":
    unittest.main()
