import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import assemble_feature_dataset as assembly  # noqa: E402
import extract_features  # noqa: E402


class FeatureDatasetAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest_path = self.root / "corpus.csv"
        self.state_path = self.root / "state.json"
        self.output_dir = self.root / "assembled"
        self.inventory = []
        self.state_recordings = {}
        self.config = json.loads(
            extract_features.FeatureConfiguration(
                feature_families=("time",)
            ).to_json()
        )
        self.addCleanup(self.temporary_directory.cleanup)

    def add_recording(
        self,
        patient,
        relative,
        labels,
        *,
        values=None,
        output="completed",
        checksum_ok=True,
    ):
        recording_id = Path(relative).stem
        identity = (patient, recording_id, relative)
        self.inventory.append(
            {
                "patient_id": patient,
                "recording_id": recording_id,
                "relative_path": relative,
                "edf_filename": Path(relative).name,
                "seizure_status": (
                    "seizures_annotated"
                    if patient.casefold() == "chb01"
                    else "no_seizures_annotated"
                ),
            }
        )
        output_path = self.root / "features" / patient / f"{Path(relative).name}.csv"
        if output == "completed":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            feature_columns = []
            for index, label in enumerate(labels):
                feature_columns.extend(
                    f"channel_{index:02d}__{quote(label, safe='')}__{name}"
                    for name in (
                        "mean",
                        "std_population",
                        "maximum",
                        "minimum",
                    )
                )
            fields = [*extract_features.METADATA_COLUMNS, *feature_columns]
            row_values = values or [
                [float(index + channel * 10) for channel in range(len(labels))]
                for index in range(2)
            ]
            annotation_status = (
                "confirmed_seizure"
                if patient.casefold() == "chb01"
                else "confirmed_no_seizure"
            )
            with output_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for index, signal_values in enumerate(row_values):
                    row = {
                        "patient_id": patient,
                        "recording_id": recording_id,
                        "relative_path": relative,
                        "recording_filename": Path(relative).name,
                        "window_index": index,
                        "start_time_seconds": index * 2.5,
                        "end_time_seconds": index * 2.5 + 5,
                        "seizure_label": (
                            "seizure"
                            if annotation_status == "confirmed_seizure" and index == 1
                            else "non_seizure"
                        ),
                        "annotation_status": annotation_status,
                        "label_policy": "any_overlap",
                        "sampling_frequency_hz": 256,
                        "window_samples": 1280,
                        "feature_config_json": json.dumps(self.config, sort_keys=True),
                    }
                    for channel, label in enumerate(labels):
                        row[
                            f"channel_{channel:02d}__{quote(label, safe='')}__mean"
                        ] = signal_values[channel]
                        row[
                            f"channel_{channel:02d}__{quote(label, safe='')}__std_population"
                        ] = 0.5
                        row[
                            f"channel_{channel:02d}__{quote(label, safe='')}__maximum"
                        ] = signal_values[channel] + 1
                        row[
                            f"channel_{channel:02d}__{quote(label, safe='')}__minimum"
                        ] = signal_values[channel] - 1
                    writer.writerow(row)
            actual_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
            self.state_recordings[assembly._identity_key(identity)] = {
                "patient_id": patient,
                "recording_id": recording_id,
                "relative_path": relative,
                "stages": {
                    "feature_extraction": {
                        "status": "Completed",
                        "output_path": str(output_path),
                        "output_checksum": (
                            actual_hash if checksum_ok else "0" * 64
                        ),
                        "row_count": 2,
                    }
                },
            }
        elif output == "missing":
            self.state_recordings[assembly._identity_key(identity)] = {
                "patient_id": patient,
                "recording_id": recording_id,
                "relative_path": relative,
                "stages": {
                    "feature_extraction": {
                        "status": "Completed",
                        "output_path": str(output_path),
                        "output_checksum": "0" * 64,
                        "row_count": 2,
                    }
                },
            }
        self._write_inputs()

    def _write_inputs(self):
        with self.manifest_path.open("w", newline="", encoding="utf-8") as file:
            fields = [
                "patient_id",
                "recording_id",
                "relative_path",
                "edf_filename",
                "seizure_status",
            ]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.inventory)
        self.state_path.write_text(
            json.dumps({"schema_version": 1, "recordings": self.state_recordings}),
            encoding="utf-8",
        )

    def assemble(self, chunk_rows=1):
        return assembly.assemble_feature_dataset(
            corpus_manifest=self.manifest_path,
            state_path=self.state_path,
            output_dir=self.output_dir,
            chunk_rows=chunk_rows,
        )

    def test_multi_patient_union_preserves_identity_and_nan_missing_channels(self):
        self.add_recording(
            "chb01",
            "chb01/same.edf",
            ["C3-P3", "P3-C3"],
            values=[[float("nan"), 5.0], [2.0, 6.0]],
        )
        self.add_recording(
            "chb02",
            "chb02/same.edf",
            ["P3-C3", "T8-P8"],
            values=[[10.0, 20.0], [11.0, 21.0]],
        )
        self.add_recording(
            "chb03",
            "chb03/missing.edf",
            ["C3-P3"],
            output="missing",
        )
        result = self.assemble(chunk_rows=1)

        with Path(result["dataset_path"]).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        schema = json.loads(Path(result["schema_path"]).read_text(encoding="utf-8"))
        with Path(result["recordings_manifest"]).open(
            newline="", encoding="utf-8"
        ) as file:
            recordings = list(csv.DictReader(file))

        self.assertEqual(result["included_recordings"], 2)
        self.assertEqual(result["excluded_recordings"], 1)
        self.assertEqual(result["windows"], 4)
        self.assertEqual(rows[0]["patient_id"], "chb01")
        self.assertEqual(rows[0]["relative_path"], "chb01/same.edf")
        self.assertEqual(rows[2]["patient_id"], "chb02")
        self.assertEqual(rows[2]["relative_path"], "chb02/same.edf")
        self.assertEqual(rows[2]["channel__C3-P3__instance_01__mean"], "NaN")
        self.assertEqual(rows[2]["channel__P3-C3__instance_01__mean"], "10.0")
        self.assertEqual(
            rows[0]["channel__C3-P3__instance_01__mean"].casefold(), "nan"
        )
        self.assertIn("channel__C3-P3__instance_01__mean", schema["predictor_columns"])
        self.assertIn("seizure_label", schema["metadata_columns"])
        self.assertNotIn("patient_id", schema["predictor_columns"])
        self.assertNotIn("window_index", schema["predictor_columns"])
        chb02_channels = schema["recordings"][
            assembly._identity_key(("chb02", "same", "chb02/same.edf"))
        ]
        self.assertIn("P3-C3", {item["original_label"] for item in chb02_channels})
        unavailable = schema["unavailable_channels"][
            assembly._identity_key(("chb02", "same", "chb02/same.edf"))
        ]
        self.assertIn(
            "channel__C3-P3__instance_01",
            unavailable,
        )
        self.assertEqual(recordings[2]["status"], "excluded")
        self.assertIn("missing", recordings[2]["reason"])

    def test_reversed_bipolar_derivations_remain_distinct(self):
        self.add_recording("chb01", "chb01/a.edf", ["C3-P3"])
        self.add_recording("chb02", "chb02/b.edf", ["P3-C3"])
        result = self.assemble()
        schema = json.loads(Path(result["schema_path"]).read_text())

        self.assertIn("channel__C3-P3__instance_01__mean", schema["predictor_columns"])
        self.assertIn("channel__P3-C3__instance_01__mean", schema["predictor_columns"])

    def test_corrupted_output_is_excluded_and_reported(self):
        self.add_recording(
            "chb01", "chb01/good.edf", ["C3-P3"]
        )
        self.add_recording(
            "chb02",
            "chb02/corrupt.edf",
            ["C3-P3"],
            checksum_ok=False,
        )
        result = self.assemble()
        with Path(result["recordings_manifest"]).open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[1]["status"], "excluded")
        self.assertIn("checksum", rows[1]["reason"])

    def test_incomplete_feature_schema_is_excluded_and_reported(self):
        self.add_recording("chb01", "chb01/good.edf", ["C3-P3"])
        self.add_recording("chb02", "chb02/incomplete.edf", ["T8-P8"])
        identity = ("chb02", "incomplete", "chb02/incomplete.edf")
        stage = self.state_recordings[assembly._identity_key(identity)]["stages"][
            "feature_extraction"
        ]
        feature_path = Path(stage["output_path"])
        with feature_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fields = [
                field
                for field in (reader.fieldnames or [])
                if not field.endswith("__minimum")
            ]
        with feature_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        stage["output_checksum"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        self._write_inputs()

        result = self.assemble()

        with Path(result["recordings_manifest"]).open(
            newline="", encoding="utf-8"
        ) as file:
            report = list(csv.DictReader(file))
        self.assertEqual(result["included_recordings"], 1)
        self.assertEqual(report[1]["status"], "excluded")
        self.assertIn("do not match", report[1]["reason"])

    def test_deterministic_outputs_on_repeated_assembly(self):
        self.add_recording("chb01", "chb01/a.edf", ["C3-P3"])
        first = self.assemble()
        first_bytes = {
            key: Path(first[key]).read_bytes()
            for key in ("dataset_path", "recordings_manifest", "schema_path")
        }
        second = self.assemble()
        second_bytes = {
            key: Path(second[key]).read_bytes()
            for key in ("dataset_path", "recordings_manifest", "schema_path")
        }
        self.assertEqual(first_bytes, second_bytes)

    def test_duplicate_corpus_identity_is_rejected(self):
        self.add_recording("chb01", "chb01/a.edf", ["C3-P3"])
        self.inventory.append(dict(self.inventory[0]))
        self._write_inputs()

        with self.assertRaisesRegex(ValueError, "Duplicate corpus recording identity"):
            self.assemble()

    def test_feature_configuration_mismatch_is_excluded(self):
        self.add_recording("chb01", "chb01/a.edf", ["C3-P3"])
        self.add_recording("chb02", "chb02/b.edf", ["T8-P8"])
        second_identity = ("chb02", "b", "chb02/b.edf")
        stage = self.state_recordings[assembly._identity_key(second_identity)][
            "stages"
        ]["feature_extraction"]
        feature_path = Path(stage["output_path"])
        changed = dict(self.config)
        changed["wavelet"] = "sym5"
        with feature_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fields = reader.fieldnames or []
        with feature_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                row["feature_config_json"] = json.dumps(changed, sort_keys=True)
                writer.writerow(row)
        stage["output_checksum"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        self._write_inputs()

        result = self.assemble()

        self.assertEqual(result["included_recordings"], 1)
        self.assertEqual(result["excluded_recordings"], 1)


if __name__ == "__main__":
    unittest.main()
