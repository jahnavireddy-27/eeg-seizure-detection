import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extract_features as features  # noqa: E402


class FeatureCalculationTests(unittest.TestCase):
    def test_time_features_use_population_standard_deviation(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        config = features.FeatureConfiguration(feature_families=("time",))
        result = features.extract_channel_features(signal, 4.0, config)
        self.assertEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["std_population"], np.std(signal, ddof=0))
        self.assertEqual(result["maximum"], 4.0)
        self.assertEqual(result["minimum"], 1.0)

    def test_constant_signal_has_zero_variability_and_spectral_power(self):
        signal = np.full(1280, 3.25)
        config = features.FeatureConfiguration()
        result = features.extract_channel_features(signal, 256, config)
        self.assertEqual(result["mean"], 3.25)
        self.assertEqual(result["std_population"], 0.0)
        self.assertEqual(result["maximum"], 3.25)
        self.assertEqual(result["minimum"], 3.25)
        self.assertEqual(result["dominant_frequency_hz"], 0.0)
        self.assertEqual(result["spectral_centroid_hz"], 0.0)
        self.assertAlmostEqual(result["total_spectral_power"], 0.0, places=25)
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            self.assertAlmostEqual(result[f"band_power_{band}"], 0.0, places=25)

    def test_sinusoid_known_frequency_and_band_power(self):
        sfreq = 256.0
        times = np.arange(1280) / sfreq
        signal = 2.0 * np.sin(2 * np.pi * 10.0 * times)
        config = features.FeatureConfiguration()
        result = features.extract_channel_features(signal, sfreq, config)
        self.assertAlmostEqual(result["dominant_frequency_hz"], 10.0)
        self.assertAlmostEqual(result["spectral_centroid_hz"], 10.0, places=10)
        self.assertAlmostEqual(result["total_spectral_power"], 2.0, places=10)
        self.assertAlmostEqual(result["band_power_alpha"], 2.0, places=10)
        for band in ("delta", "theta", "beta", "gamma"):
            self.assertAlmostEqual(result[f"band_power_{band}"], 0.0, places=10)

    def test_fft_bin_frequency_and_half_open_band_edges(self):
        sfreq = 256.0
        times = np.arange(1280) / sfreq
        signal = np.sin(2 * np.pi * 4.0 * times)
        band_config = features.FeatureConfiguration(
            feature_families=("frequency",),
            bands=(("below", 0.0, 4.0), ("at", 4.0, 8.0)),
        )
        result = features.extract_channel_features(signal, sfreq, band_config)
        self.assertEqual(result["dominant_frequency_hz"], 4.0)
        self.assertAlmostEqual(result["band_power_below"], 0.0, places=12)
        self.assertAlmostEqual(result["band_power_at"], 0.5, places=12)

    def test_wavelet_level_three_coefficient_order_and_energy(self):
        signal = np.arange(1280, dtype=np.float64)
        config = features.FeatureConfiguration(
            feature_families=("wavelet",),
            wavelet="db4",
            wavelet_mode="symmetric",
        )
        result = features.extract_channel_features(signal, 256, config)
        coefficients = pywt.wavedec(
            signal, "db4", mode="symmetric", level=3
        )
        for name, coefficient in zip(features.COEFFICIENT_SETS, coefficients):
            self.assertAlmostEqual(
                result[f"wavelet_{name}_mean"], float(np.mean(coefficient))
            )
            self.assertAlmostEqual(
                result[f"wavelet_{name}_std_population"],
                float(np.std(coefficient, ddof=0)),
            )
            self.assertAlmostEqual(
                result[f"wavelet_{name}_energy"],
                float(np.sum(coefficient**2)),
            )
        self.assertEqual(
            [key for key in result if key.endswith("_energy")],
            [
                "wavelet_A3_energy",
                "wavelet_D3_energy",
                "wavelet_D2_energy",
                "wavelet_D1_energy",
            ],
        )

    def test_non_finite_signals_are_rejected(self):
        config = features.FeatureConfiguration()
        for signal in ([1.0, np.nan], [1.0, np.inf]):
            with self.subTest(signal=signal):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    features.extract_channel_features(signal, 256, config)

    def test_custom_frequency_bands_must_be_valid_and_under_nyquist(self):
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            features.FeatureConfiguration(
                bands=(("one", 0, 5), ("two", 4, 8))
            ).validate()
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            features.extract_channel_features(
                np.ones(8),
                16,
                features.FeatureConfiguration(
                    feature_families=("frequency",),
                    bands=(("too_high", 0, 9),),
                ),
            )


class FeatureMatrixTests(unittest.TestCase):
    def _create_fixture(self, directory: Path) -> tuple[Path, Path, Path]:
        manifest_path = directory / "manifest.csv"
        csv_dir = directory / "csv"
        output_dir = directory / "features"
        csv_dir.mkdir()
        fields = [
            "patient_id",
            "recording_filename",
            "recording_id",
            "window_index",
            "start_sample",
            "end_sample_exclusive",
            "start_time_seconds",
            "end_time_seconds",
            "seizure_label",
            "label_policy",
            "window_samples",
            "stride_samples",
            "channel_count",
            "channel_labels",
            "sampling_frequency_hz",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for index in range(2):
                start = index * 640
                writer.writerow(
                    {
                        "patient_id": "chb01",
                        "recording_filename": "chb01_test.edf",
                        "recording_id": "chb01_test",
                        "window_index": index,
                        "start_sample": start,
                        "end_sample_exclusive": start + 1280,
                        "start_time_seconds": start / 256,
                        "end_time_seconds": (start + 1280) / 256,
                        "seizure_label": "seizure" if index else "non_seizure",
                        "label_policy": "any_overlap",
                        "window_samples": 1280,
                        "stride_samples": 640,
                        "channel_count": 3,
                        "channel_labels": json.dumps(
                            ["C3-P3", "T8-P8", "T8-P8"]
                        ),
                        "sampling_frequency_hz": 256,
                    }
                )
        csv_path = csv_dir / "chb01_test.edf.csv"
        times = np.arange(1920) / 256
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "C3-P3", "T8-P8", "T8-P8"])
            for sample, time in enumerate(times):
                writer.writerow(
                    [time, sample % 13, np.sin(sample / 10), np.cos(sample / 10)]
                )
        return manifest_path, csv_dir, output_dir

    def test_duplicate_label_feature_columns_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, csv_dir, output_dir = self._create_fixture(Path(directory))
            config = features.FeatureConfiguration(feature_families=("time",))
            output_path, rows, columns = features.extract_feature_matrix(
                manifest_path=manifest,
                csv_dir=csv_dir,
                output_dir=output_dir,
                configuration=config,
                chunk_rows=257,
            )
            self.assertEqual(rows, 2)
            self.assertEqual(columns, len(features.METADATA_COLUMNS) + 3 * 4)
            with output_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                data = list(reader)
                self.assertEqual(len(reader.fieldnames or []), columns)
                self.assertIn("channel_01__T8-P8__mean", reader.fieldnames or [])
                self.assertIn("channel_02__T8-P8__mean", reader.fieldnames or [])
                self.assertEqual(len(set(reader.fieldnames or [])), columns)
            self.assertEqual([row["window_index"] for row in data], ["0", "1"])
            self.assertEqual(data[0]["seizure_label"], "non_seizure")
            self.assertEqual(data[1]["label_policy"], "any_overlap")
            expected_mean = float(np.mean(np.arange(1280, dtype=np.float64) % 13))
            self.assertAlmostEqual(
                float(data[0]["channel_00__C3-P3__mean"]), expected_mean
            )

    def test_non_finite_csv_values_fail_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, csv_dir, output_dir = self._create_fixture(Path(directory))
            csv_path = csv_dir / "chb01_test.edf.csv"
            lines = csv_path.read_text(encoding="utf-8").splitlines()
            lines[1] = "0.0,0,0.0,nan"
            csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite value"):
                features.extract_feature_matrix(
                    manifest_path=manifest,
                    csv_dir=csv_dir,
                    output_dir=output_dir,
                    configuration=features.FeatureConfiguration(
                        feature_families=("time",)
                    ),
                    chunk_rows=257,
                )
            self.assertFalse((output_dir / "chb01_features.csv").exists())

    def test_repeated_runs_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, csv_dir, output_dir = self._create_fixture(Path(directory))
            config = features.FeatureConfiguration(
                feature_families=("time", "frequency")
            )
            path, rows, _ = features.extract_feature_matrix(
                manifest_path=manifest,
                csv_dir=csv_dir,
                output_dir=output_dir,
                configuration=config,
                chunk_rows=300,
            )
            first = path.read_bytes()
            first_schema = path.with_suffix(".schema.json").read_bytes()
            second_path, second_rows, _ = features.extract_feature_matrix(
                manifest_path=manifest,
                csv_dir=csv_dir,
                output_dir=output_dir,
                configuration=config,
                chunk_rows=511,
            )
            self.assertEqual(rows, second_rows)
            self.assertEqual(first, second_path.read_bytes())
            self.assertEqual(
                first_schema,
                second_path.with_suffix(".schema.json").read_bytes(),
            )

    def test_wrong_csv_channel_labels_are_not_silently_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, csv_dir, output_dir = self._create_fixture(Path(directory))
            csv_path = csv_dir / "chb01_test.edf.csv"
            text = csv_path.read_text(encoding="utf-8")
            csv_path.write_text(text.replace("T8-P8", "T7-P7", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                features.extract_feature_matrix(
                    manifest_path=manifest,
                    csv_dir=csv_dir,
                    output_dir=output_dir,
                    configuration=features.FeatureConfiguration(
                        feature_families=("time",)
                    ),
                )

    def test_multi_patient_extraction_aligns_channels_by_conservative_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "windows.csv"
            csv_dir = root / "signals"
            output_dir = root / "features"
            fields = [
                "patient_id",
                "relative_path",
                "recording_filename",
                "recording_id",
                "window_index",
                "start_sample",
                "end_sample_exclusive",
                "start_time_seconds",
                "end_time_seconds",
                "seizure_label",
                "annotation_status",
                "label_policy",
                "window_samples",
                "stride_samples",
                "channel_count",
                "channel_labels",
                "sampling_frequency_hz",
            ]
            recordings = [
                ("chb01", "chb01/same.edf", ["C3-P3", "P3-C3"], [[1, 2, 3, 4], [4, 3, 2, 1]]),
                ("chb02", "chb02/same.edf", [" p3-c3 ", "T8-P8"], [[10, 11, 12, 13], [7, 8, 9, 10]]),
            ]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for patient, relative, labels, signals in recordings:
                    writer.writerow(
                        {
                            "patient_id": patient,
                            "relative_path": relative,
                            "recording_filename": "same.edf",
                            "recording_id": "same",
                            "window_index": 0,
                            "start_sample": 0,
                            "end_sample_exclusive": 4,
                            "start_time_seconds": 0,
                            "end_time_seconds": 1,
                            "seizure_label": "non_seizure",
                            "annotation_status": "confirmed_no_seizure",
                            "label_policy": "any_overlap",
                            "window_samples": 4,
                            "stride_samples": 4,
                            "channel_count": len(labels),
                            "channel_labels": json.dumps(labels),
                            "sampling_frequency_hz": 4,
                        }
                    )
                    source_path = csv_dir / f"{relative}.csv"
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    with source_path.open("w", newline="", encoding="utf-8") as signal_file:
                        signal_writer = csv.writer(signal_file)
                        signal_writer.writerow(["Time", *labels])
                        for sample in range(4):
                            signal_writer.writerow(
                                [sample / 4, *(signal[sample] for signal in signals)]
                            )

            output, count, _ = features.extract_feature_matrix(
                manifest_path=manifest,
                csv_dir=csv_dir,
                output_dir=output_dir,
                configuration=features.FeatureConfiguration(
                    feature_families=("time",)
                ),
                chunk_rows=1,
            )
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            schema = json.loads(
                output.with_suffix(".schema.json").read_text(encoding="utf-8")
            )

        self.assertEqual(output.name, "chbmit_features.csv")
        self.assertEqual(count, 2)
        self.assertEqual([row["relative_path"] for row in rows], [
            "chb01/same.edf",
            "chb02/same.edf",
        ])
        self.assertIn("channel__C3-P3__instance_01__mean", rows[0])
        self.assertIn("channel__P3-C3__instance_01__mean", rows[0])
        self.assertIn("channel__T8-P8__instance_01__mean", rows[0])
        self.assertEqual(rows[1]["channel__C3-P3__instance_01__mean"], "nan")
        self.assertAlmostEqual(float(rows[0]["channel__C3-P3__instance_01__mean"]), 2.5)
        self.assertAlmostEqual(float(rows[1]["channel__P3-C3__instance_01__mean"]), 11.5)
        self.assertEqual(schema["schema_mode"], "channel_aligned")
        self.assertEqual(schema["target_column"], "seizure_label")
        unavailable = schema["unavailable_channels"][
            features._identity_json_key(("chb02", "same", "chb02/same.edf"))
        ]
        self.assertIn("channel__C3-P3__instance_01", unavailable)

    def test_existing_chb01_feature_values_are_preserved_when_artifacts_exist(self):
        project_root = Path(__file__).resolve().parents[1]
        window_manifest = (
            project_root / "data/processed/window_manifests/chb01_window_manifest.csv"
        )
        feature_matrix = project_root / "data/processed/features/chb01_features.csv"
        signal_dir = project_root / "data/processed/csv"
        if not window_manifest.is_file() or not feature_matrix.is_file():
            self.skipTest("Local CHB01 feature artifacts are not available.")

        with window_manifest.open(newline="", encoding="utf-8") as file:
            window_reader = csv.DictReader(file)
            manifest_fields = window_reader.fieldnames or []
            window_row = next(window_reader)
        relative_path = window_row.get("relative_path") or window_row["recording_filename"]
        source_signal = signal_dir / f"{window_row['recording_filename']}.csv"
        if not source_signal.is_file():
            self.skipTest("Local converted CHB01 signal CSV is not available.")

        with feature_matrix.open(newline="", encoding="utf-8") as file:
            feature_reader = csv.DictReader(file)
            baseline_row = next(
                row
                for row in feature_reader
                if row["patient_id"] == window_row["patient_id"]
                and row["recording_id"] == window_row["recording_id"]
                and row["window_index"] == window_row["window_index"]
            )
            baseline_feature_columns = [
                column
                for column in feature_reader.fieldnames or []
                if column.startswith("channel_")
            ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "one_window.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=manifest_fields)
                writer.writeheader()
                writer.writerow(window_row)

            signal_copy = root / f"{relative_path}.csv"
            signal_copy.parent.mkdir(parents=True, exist_ok=True)
            with source_signal.open(newline="", encoding="utf-8") as source, signal_copy.open(
                "w", newline="", encoding="utf-8"
            ) as destination:
                reader = csv.reader(source)
                writer = csv.writer(destination)
                writer.writerow(next(reader))
                for _ in range(int(window_row["window_samples"])):
                    writer.writerow(next(reader))

            output_path, rows, _ = features.extract_feature_matrix(
                manifest_path=manifest_path,
                csv_dir=root,
                output_dir=root / "output",
            )
            with output_path.open(newline="", encoding="utf-8") as file:
                regenerated = next(csv.DictReader(file))

        self.assertEqual(rows, 1)
        self.assertEqual(
            {column: regenerated[column] for column in baseline_feature_columns},
            {column: baseline_row[column] for column in baseline_feature_columns},
        )


if __name__ == "__main__":
    unittest.main()
