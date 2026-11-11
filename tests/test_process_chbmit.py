import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import process_chbmit as pipeline  # noqa: E402


MANIFEST_COLUMNS = [
    "patient_id",
    "recording_id",
    "relative_path",
    "edf_filename",
    "download_status",
    "recording_duration_seconds",
    "seizure_status",
    "seizure_intervals",
    "annotation_agreement",
]


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "raw"
        self.dataset_root.mkdir()
        self.output_root = self.root / "processed"
        self.manifest_path = self.root / "corpus.csv"
        self.records = []
        self.calls = {stage: [] for stage in pipeline.STAGES}
        self._write_manifest()
        self.patches = [
            patch.object(
                pipeline.edf_to_csv,
                "convert_edf_to_csv",
                side_effect=self.fake_conversion,
            ),
            patch.object(
                pipeline.segment_windows,
                "create_recording_manifest",
                side_effect=self.fake_segmentation,
            ),
            patch.object(
                pipeline.extract_features,
                "extract_feature_matrix",
                side_effect=self.fake_feature_extraction,
            ),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temporary_directory.cleanup)

    def add_record(
        self,
        patient,
        relative,
        *,
        status="downloaded_validated",
        annotation="no_seizures_annotated",
        agreement="not_available",
    ):
        path = self.dataset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"edf:{patient}:{relative}".encode())
        self.records.append(
            {
                "patient_id": patient,
                "recording_id": Path(relative).stem,
                "relative_path": Path(relative).as_posix(),
                "edf_filename": Path(relative).name,
                "download_status": status,
                "recording_duration_seconds": "10",
                "seizure_status": annotation,
                "seizure_intervals": "[]",
                "annotation_agreement": agreement,
            }
        )
        self._write_manifest()

    def _write_manifest(self):
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(self.records)

    @staticmethod
    def _write_csv(path, rows=1, tag="row"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metadata", "value"])
            for index in range(rows):
                writer.writerow([tag, index])

    def fake_conversion(self, source_path, output_dir, *, chunk_seconds):
        self.calls["conversion"].append((Path(source_path), chunk_seconds))
        output_path = Path(output_dir) / f"{Path(source_path).name}.csv"
        self._write_csv(
            output_path,
            rows=3,
            tag=Path(source_path).read_bytes().decode("utf-8"),
        )
        return type(
            "Report",
            (),
            {
                "output_path": output_path,
                "exported_rows": 3,
                "sample_count": 3,
            },
        )()

    def fake_segmentation(self, **kwargs):
        self.calls["segmentation"].append(kwargs["edf_path"])
        configuration = kwargs["configuration"]
        tag = (
            Path(kwargs["edf_path"]).read_bytes().decode("utf-8")
            + f":{configuration.window_seconds}"
        )
        self._write_csv(kwargs["output_path"], rows=2, tag=tag)
        return 2, "synthetic recording: 2 windows"

    def fake_feature_extraction(self, **kwargs):
        self.calls["feature_extraction"].append(kwargs["manifest_path"])
        output_dir = Path(kwargs["output_dir"])
        output_path = output_dir / "patient_features.csv"
        self._write_csv(output_path, rows=kwargs["chunk_rows"] and 2)
        return output_path, 2, 4

    def run_pipeline(self, **kwargs):
        return pipeline.process_corpus(
            manifest_path=self.manifest_path,
            dataset_root=self.dataset_root,
            output_root=self.output_root,
            **kwargs,
        )

    def state(self):
        return json.loads(
            (self.output_root / pipeline.STATE_FILENAME).read_text(encoding="utf-8")
        )

    def key_for(self, patient, relative):
        return pipeline._record_key(patient, relative)


class ProcessChbmitTests(PipelineFixture):
    def test_dry_run_selects_records_without_state_or_processing(self):
        self.add_record("chb01", "chb01/a.edf")
        self.add_record("chb02", "chb02/b.edf")

        result = self.run_pipeline(patients=["chb02"], dry_run=True)

        self.assertEqual(result["selected"], 1)
        self.assertTrue(result["dry_run"])
        self.assertFalse(self.output_root.exists())
        self.assertEqual(sum(map(len, self.calls.values())), 0)

    def test_multiple_patients_and_duplicate_basenames_get_isolated_outputs(self):
        self.add_record("chb01", "chb01/record.edf")
        self.add_record("chb02", "chb02/record.edf")

        result = self.run_pipeline()

        self.assertEqual(result["selected"], 2)
        self.assertEqual(result["completed"], 6)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(len(self.calls["conversion"]), 2)
        state = self.state()
        first = state["recordings"][self.key_for("chb01", "chb01/record.edf")]
        second = state["recordings"][self.key_for("chb02", "chb02/record.edf")]
        first_paths = {
            entry["output_path"]
            for entry in first["stages"].values()
            if entry["status"] == "Completed"
        }
        second_paths = {
            entry["output_path"]
            for entry in second["stages"].values()
            if entry["status"] == "Completed"
        }
        self.assertTrue(first_paths.isdisjoint(second_paths))
        self.assertTrue(all("/chb01/" in path for path in first_paths))
        self.assertTrue(all("/chb02/" in path for path in second_paths))
        self.assertEqual(
            {entry["status"] for entry in first["stages"].values()},
            {"Completed"},
        )

    def test_specific_recording_selection_accepts_relative_path(self):
        self.add_record("chb01", "chb01/a.edf")
        self.add_record("chb01", "chb01/b.edf")

        result = self.run_pipeline(recordings=["chb01/b.edf"], dry_run=True)

        self.assertEqual(result["selected"], 1)

    def test_resume_skips_verified_completed_stages(self):
        self.add_record("chb01", "chb01/a.edf")
        self.run_pipeline()
        first_calls = {key: len(value) for key, value in self.calls.items()}

        result = self.run_pipeline()

        self.assertEqual(result["failed"], 0)
        self.assertEqual(
            {key: len(value) for key, value in self.calls.items()},
            first_calls,
        )

    def test_interrupted_conversion_is_failed_then_resumed(self):
        self.add_record("chb01", "chb01/a.edf")
        original = self.fake_conversion
        attempts = 0

        def interrupted(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("simulated interruption")
            return original(*args, **kwargs)

        with patch.object(
            pipeline.edf_to_csv, "convert_edf_to_csv", side_effect=interrupted
        ):
            first = self.run_pipeline()
            status_after_first = self.state()["recordings"][
                self.key_for("chb01", "chb01/a.edf")
            ]["stages"]["conversion"]["status"]
            second = self.run_pipeline()

        self.assertEqual(first["failed"], 1)
        self.assertEqual(status_after_first, "Failed")
        self.assertEqual(second["failed"], 0)
        self.assertEqual(attempts, 2)
        self.assertEqual(len(self.calls["segmentation"]), 1)

    def test_process_interruption_leaves_running_and_pending_states_resumable(self):
        self.add_record("chb01", "chb01/a.edf")

        def terminate(*args, **kwargs):
            raise KeyboardInterrupt

        with patch.object(
            pipeline.edf_to_csv, "convert_edf_to_csv", side_effect=terminate
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline()

        stages = self.state()["recordings"][
            self.key_for("chb01", "chb01/a.edf")
        ]["stages"]
        self.assertEqual(stages["conversion"]["status"], "Running")
        self.assertEqual(stages["segmentation"]["status"], "Pending")
        self.assertEqual(stages["feature_extraction"]["status"], "Pending")

        resumed = self.run_pipeline()

        self.assertEqual(resumed["failed"], 0)
        self.assertEqual(
            {entry["status"] for entry in self.state()["recordings"][
                self.key_for("chb01", "chb01/a.edf")
            ]["stages"].values()},
            {"Completed"},
        )

    def test_interrupted_feature_extraction_resumes_only_failed_stage(self):
        self.add_record("chb01", "chb01/a.edf")
        original = self.fake_feature_extraction
        attempts = 0

        def interrupted(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("feature process interrupted")
            return original(**kwargs)

        with patch.object(
            pipeline.extract_features,
            "extract_feature_matrix",
            side_effect=interrupted,
        ):
            first = self.run_pipeline()
            second = self.run_pipeline()

        self.assertEqual(first["failed"], 1)
        self.assertEqual(second["failed"], 0)
        self.assertEqual(attempts, 2)
        self.assertEqual(len(self.calls["conversion"]), 1)
        self.assertEqual(len(self.calls["segmentation"]), 1)

    def test_changed_input_checksum_invalidates_downstream_stages(self):
        source = self.dataset_root / "chb01/a.edf"
        self.add_record("chb01", "chb01/a.edf")
        self.run_pipeline()
        source.write_bytes(b"changed input")

        self.run_pipeline()

        self.assertEqual(len(self.calls["conversion"]), 2)
        self.assertEqual(len(self.calls["segmentation"]), 2)
        self.assertEqual(len(self.calls["feature_extraction"]), 2)

    def test_changed_configuration_invalidates_stages(self):
        self.add_record("chb01", "chb01/a.edf")
        self.run_pipeline()

        self.run_pipeline(
            configuration=pipeline.PipelineConfiguration(window_seconds=4.0)
        )

        self.assertEqual(len(self.calls["conversion"]), 1)
        self.assertEqual(len(self.calls["segmentation"]), 2)
        self.assertEqual(len(self.calls["feature_extraction"]), 2)

    def test_corrupted_output_causes_affected_stage_to_rerun(self):
        self.add_record("chb01", "chb01/a.edf")
        self.run_pipeline()
        state = self.state()
        conversion = state["recordings"][
            self.key_for("chb01", "chb01/a.edf")
        ]["stages"]["conversion"]
        Path(conversion["output_path"]).write_text("corrupted", encoding="utf-8")

        self.run_pipeline()

        self.assertEqual(len(self.calls["conversion"]), 2)
        self.assertEqual(len(self.calls["segmentation"]), 1)
        self.assertEqual(len(self.calls["feature_extraction"]), 1)

    def test_missing_output_causes_stage_to_rerun(self):
        self.add_record("chb01", "chb01/a.edf")
        self.run_pipeline()
        conversion = self.state()["recordings"][
            self.key_for("chb01", "chb01/a.edf")
        ]["stages"]["conversion"]
        Path(conversion["output_path"]).unlink()

        self.run_pipeline()

        self.assertEqual(len(self.calls["conversion"]), 2)
        self.assertTrue(Path(self.state()["recordings"][
            self.key_for("chb01", "chb01/a.edf")
        ]["stages"]["conversion"]["output_path"]).is_file())

    def test_failed_recording_does_not_stop_later_recordings(self):
        self.add_record("chb01", "chb01/a.edf")
        self.add_record("chb02", "chb02/b.edf")
        original = self.fake_conversion

        def fail_first(source_path, output_dir, *, chunk_seconds):
            if Path(source_path).name == "a.edf":
                raise OSError("recording-specific failure")
            return original(source_path, output_dir, chunk_seconds=chunk_seconds)

        with patch.object(
            pipeline.edf_to_csv,
            "convert_edf_to_csv",
            side_effect=fail_first,
        ):
            result = self.run_pipeline()

        self.assertEqual(result["failed"], 1)
        self.assertEqual(len(self.calls["segmentation"]), 1)
        self.assertEqual(len(self.calls["feature_extraction"]), 1)
        errors = self.state()["errors"]
        self.assertEqual(errors[0]["patient_id"], "chb01")
        self.assertEqual(errors[0]["recording_id"], "a")
        self.assertEqual(errors[0]["stage"], "conversion")
        self.assertTrue(errors[0]["retryable"])
        self.assertIn("recording-specific failure", errors[0]["message"])

    def test_missing_edf_and_unavailable_annotations_are_explicitly_skipped(self):
        self.add_record(
            "chb01",
            "chb01/missing.edf",
            status="not_downloaded",
        )
        self.add_record(
            "chb02",
            "chb02/unknown.edf",
            annotation="annotation_unavailable",
        )
        (self.dataset_root / "chb01/missing.edf").unlink()

        self.run_pipeline()

        state = self.state()
        missing = state["recordings"][
            self.key_for("chb01", "chb01/missing.edf")
        ]["stages"]
        unknown = state["recordings"][
            self.key_for("chb02", "chb02/unknown.edf")
        ]["stages"]
        self.assertEqual({entry["status"] for entry in missing.values()}, {"Skipped"})
        self.assertEqual(
            missing["conversion"]["reason"], "manifest download_status=not_downloaded"
        )
        self.assertEqual(unknown["conversion"]["status"], "Completed")
        self.assertEqual(unknown["segmentation"]["status"], "Skipped")
        self.assertIn("annotation status", unknown["segmentation"]["reason"])

    def test_atomic_state_update_preserves_previous_file_on_replace_error(self):
        state_path = self.root / "state.json"
        state_path.write_text('{"previous": true}\n', encoding="utf-8")

        with patch.object(pipeline.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                pipeline._atomic_write_json(state_path, {"next": True})

        self.assertEqual(json.loads(state_path.read_text()), {"previous": True})
        self.assertEqual(list(self.root.glob(".state.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
