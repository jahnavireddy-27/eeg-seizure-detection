import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset_manifest as manifest  # noqa: E402


class FakeEdfReader:
    def __init__(self, path):
        self.path = Path(path)
        self.file_duration = 120.0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def getSampleFrequencies(self):
        return [256.0, 256.0]

    def readSignal(self, *_args, **_kwargs):
        raise AssertionError("Manifest generation must not read EDF signal samples.")


def summary_text(*recordings):
    lines = ["Data Sampling Rate: 256 Hz", ""]
    for filename, count, intervals in recordings:
        lines.extend(
            [
                f"File Name: {filename}",
                f"Number of Seizures in File: {count}",
            ]
        )
        for start, end in intervals:
            lines.extend(
                [
                    f"Seizure Start Time: {start} seconds",
                    f"Seizure End Time: {end} seconds",
                ]
            )
        lines.append("")
    return "\n".join(lines)


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.reader_patch = patch.object(
            manifest.pyedflib, "EdfReader", side_effect=FakeEdfReader
        )
        self.reader_patch.start()
        self.addCleanup(self.reader_patch.stop)
        self.addCleanup(self.temporary_directory.cleanup)

    def write_summary(self, patient_id, *recordings):
        patient_dir = self.root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        path = patient_dir / f"{patient_id}-summary.txt"
        path.write_text(summary_text(*recordings), encoding="utf-8")
        return path

    def write_edf(self, patient_id, filename, content=b"fake-edf"):
        patient_dir = self.root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        path = patient_dir / filename
        path.write_bytes(content)
        return path

    def read_rows(self):
        return manifest.build_dataset_manifest(self.root)

    def test_multiple_patients_keep_duplicate_basenames_patient_scoped(self):
        self.write_summary("chb01", ("recording.edf", 0, []))
        self.write_summary("chb02", ("recording.edf", 1, [(1, 2)]))
        first = self.write_edf("chb01", "recording.edf", b"first")
        second = self.write_edf("chb02", "recording.edf", b"second")

        rows = self.read_rows()

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {(row["patient_id"], row["recording_id"]) for row in rows},
            {("chb01", "recording"), ("chb02", "recording")},
        )
        self.assertEqual(
            {row["relative_path"] for row in rows},
            {"chb01/recording.edf", "chb02/recording.edf"},
        )
        self.assertEqual({row["download_status"] for row in rows}, {"downloaded_validated"})
        self.assertEqual(
            {row["seizure_status"] for row in rows},
            {"no_seizures_annotated", "seizures_annotated"},
        )
        self.assertEqual(
            {row["edf_sha256"] for row in rows},
            {
                hashlib.sha256(first.read_bytes()).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
            },
        )

    def test_summary_record_without_local_edf_is_not_downloaded(self):
        self.write_summary("chb01", ("chb01_01.edf", 0, []))

        row = self.read_rows()[0]

        self.assertEqual(row["download_status"], "not_downloaded")
        self.assertEqual(row["expected_status"], "expected_in_patient_summary")
        self.assertEqual(row["recording_duration_seconds"], "")
        self.assertEqual(row["edf_sha256"], "")
        self.assertEqual(row["seizure_status"], "no_seizures_annotated")

    def test_absent_summary_does_not_turn_missing_annotation_into_negative(self):
        self.write_edf("chb03", "chb03_01.edf")

        row = self.read_rows()[0]

        self.assertEqual(row["patient_id"], "chb03")
        self.assertEqual(row["seizure_status"], "annotation_unavailable")
        self.assertEqual(row["annotation_source"], "unavailable")
        self.assertEqual(row["expected_status"], "not_listed_in_patient_summary")

    def test_local_recording_missing_from_existing_summary_is_unknown(self):
        self.write_summary("chb01", ("chb01_01.edf", 0, []))
        self.write_edf("chb01", "chb01_99.edf")

        rows = {row["recording_id"]: row for row in self.read_rows()}

        self.assertEqual(rows["chb01_99"]["expected_status"], "not_listed_in_patient_summary")
        self.assertEqual(rows["chb01_99"]["seizure_status"], "annotation_unavailable")
        self.assertEqual(rows["chb01_99"]["annotation_source"], "unavailable")

    def test_integer_and_decimal_summary_intervals_are_preserved(self):
        self.write_summary(
            "chb01",
            ("integer.edf", 1, [(10, 20)]),
            ("decimal.edf", 1, [(10.25, 20.75)]),
        )
        self.write_edf("chb01", "integer.edf")
        self.write_edf("chb01", "decimal.edf")

        rows = {row["recording_id"]: row for row in self.read_rows()}

        self.assertEqual(
            json.loads(rows["integer"]["seizure_intervals"])[0]["start_seconds"],
            10.0,
        )
        self.assertEqual(
            json.loads(rows["decimal"]["seizure_intervals"])[0]["end_seconds"],
            20.75,
        )
        self.assertTrue(
            all(row["annotation_source"] == "patient_summary" for row in rows.values())
        )

    def test_checksum_duration_and_processing_statuses(self):
        self.write_summary("chb01", ("chb01_01.edf", 0, []))
        edf_path = self.write_edf("chb01", "chb01_01.edf", b"checksum-fixture")

        row = self.read_rows()[0]

        self.assertEqual(row["edf_size_bytes"], str(edf_path.stat().st_size))
        self.assertEqual(
            row["edf_sha256"], hashlib.sha256(b"checksum-fixture").hexdigest()
        )
        self.assertEqual(row["recording_duration_seconds"], "120.000000000")
        self.assertEqual(row["processing_conversion_status"], "not_assessed")
        self.assertEqual(row["processing_segmentation_status"], "not_assessed")
        self.assertEqual(row["processing_feature_extraction_status"], "not_assessed")

    def test_unreadable_edf_is_distinguished_from_not_downloaded(self):
        self.write_summary("chb01", ("chb01_01.edf", 0, []))
        self.write_edf("chb01", "chb01_01.edf")
        with patch.object(
            manifest.pyedflib,
            "EdfReader",
            side_effect=RuntimeError("invalid EDF header"),
        ):
            row = self.read_rows()[0]

        self.assertEqual(row["download_status"], "missing_or_unreadable")
        self.assertIn("EDF_READ_OR_HASH_ERROR", row["inventory_issues"])

    def test_sidecar_is_validated_against_summary_without_guessing_bytes(self):
        self.write_summary("chb01", ("chb01_01.edf", 1, [(10, 20)]))
        self.write_edf("chb01", "chb01_01.edf")
        sidecar_path = self.root / "chb01" / "chb01_01.edf.seizures"
        sidecar_path.write_bytes(b"binary sidecar fixture")
        sidecar_result = {
            "available": True,
            "intervals": [
                {
                    "start_sample": 2560,
                    "end_sample": 5120,
                    "start_seconds": 10.0,
                    "end_seconds": 20.0,
                    "duration_seconds": 10.0,
                }
            ],
            "samples": [],
            "error": None,
        }
        with patch.object(
            manifest, "read_wfdb_sidecar", return_value=sidecar_result
        ) as sidecar_reader:
            row = self.read_rows()[0]

        sidecar_reader.assert_called_once_with(sidecar_path.resolve(), 256.0)
        self.assertEqual(row["annotation_validation_status"], "validated_match")
        self.assertEqual(row["annotation_sidecar_present"], "true")
        self.assertEqual(
            row["annotation_sidecar_relative_path"],
            "chb01/chb01_01.edf.seizures",
        )

    def test_sidecar_only_inventory_preserves_samples_without_inventing_times(self):
        patient_dir = self.root / "chb04"
        patient_dir.mkdir()
        sidecar_path = patient_dir / "chb04_02.edf.seizures"
        sidecar_path.write_bytes(b"binary sidecar fixture")
        sidecar_result = {
            "available": True,
            "intervals": [
                {
                    "start_sample": 512,
                    "end_sample": 768,
                    "start_seconds": None,
                    "end_seconds": None,
                    "duration_seconds": None,
                }
            ],
            "samples": [],
            "error": None,
        }

        with patch.object(manifest, "read_wfdb_sidecar", return_value=sidecar_result):
            row = self.read_rows()[0]

        interval = json.loads(row["seizure_intervals"])[0]
        self.assertEqual(row["download_status"], "not_downloaded")
        self.assertEqual(row["seizure_status"], "seizures_annotated")
        self.assertEqual(
            row["annotation_validation_status"],
            "unvalidated_missing_sample_rate",
        )
        self.assertEqual(interval["start_sample"], 512)
        self.assertIsNone(interval["start_seconds"])
        self.assertIsNone(interval["duration_seconds"])

    def test_summary_sidecar_disagreement_preserves_both_and_marks_unknown(self):
        self.write_summary("chb01", ("chb01_01.edf", 1, [(10, 20)]))
        self.write_edf("chb01", "chb01_01.edf")
        sidecar_path = self.root / "chb01" / "chb01_01.edf.seizures"
        sidecar_path.write_bytes(b"binary sidecar fixture")
        sidecar_result = {
            "available": True,
            "intervals": [
                {
                    "start_sample": 7680,
                    "end_sample": 10240,
                    "start_seconds": 30.0,
                    "end_seconds": 40.0,
                    "duration_seconds": 10.0,
                }
            ],
            "samples": [],
            "error": None,
        }
        with patch.object(manifest, "read_wfdb_sidecar", return_value=sidecar_result):
            row = self.read_rows()[0]

        self.assertEqual(row["seizure_status"], "annotation_unavailable")
        self.assertEqual(
            row["annotation_source"],
            "patient_summary_and_wfdb_sidecar_conflict",
        )
        self.assertEqual(row["annotation_agreement"], "disagreement")
        self.assertEqual(json.loads(row["seizure_intervals"]), [])
        self.assertEqual(
            json.loads(row["summary_seizure_intervals"])[0]["start_seconds"], 10.0
        )
        self.assertEqual(
            json.loads(row["sidecar_seizure_intervals"])[0]["start_seconds"], 30.0
        )

    def test_generation_is_deterministic_and_atomically_rerunnable(self):
        self.write_summary("chb01", ("chb01_01.edf", 0, []))
        self.write_edf("chb01", "chb01_01.edf")
        output = self.root / "manifest.csv"

        rows = self.read_rows()
        manifest.write_dataset_manifest(output, rows)
        first_content = output.read_bytes()
        manifest.write_dataset_manifest(output, self.read_rows())

        self.assertEqual(output.read_bytes(), first_content)
        with output.open(newline="", encoding="utf-8") as stream:
            parsed = list(csv.DictReader(stream))
        self.assertEqual(len(parsed), 1)

    def test_unrelated_output_is_not_overwritten(self):
        output = self.root / "notes.csv"
        output.write_text("valuable,unrelated\n", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            manifest.write_dataset_manifest(output, [])

        self.assertEqual(output.read_text(encoding="utf-8"), "valuable,unrelated\n")

    def test_failed_atomic_replace_preserves_existing_manifest_and_cleans_temp(self):
        output = self.root / "manifest.csv"
        manifest.write_dataset_manifest(output, [])
        original_content = output.read_bytes()

        with patch.object(manifest.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                manifest.write_dataset_manifest(output, [])

        self.assertEqual(output.read_bytes(), original_content)
        self.assertEqual(list(self.root.glob(".manifest.csv.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
