import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import edf_to_csv  # noqa: E402


class FakeEdfReader:
    def __init__(
        self,
        labels,
        signals,
        frequencies,
        duration,
        dimensions=None,
    ):
        self.labels = labels
        self.signals = [np.asarray(signal, dtype=np.float64) for signal in signals]
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.file_duration = duration
        self.dimensions = dimensions or ["uV"] * len(labels)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def getSignalHeaders(self):
        return [
            {"label": label, "dimension": dimension}
            for label, dimension in zip(self.labels, self.dimensions)
        ]

    def getSignalLabels(self):
        return self.labels

    def getSampleFrequencies(self):
        return self.frequencies

    def getNSamples(self):
        return np.asarray([len(signal) for signal in self.signals])

    def readSignal(self, signal_index, start=0, n=None, digital=False):
        stop = len(self.signals[signal_index]) if n is None else start + n
        return self.signals[signal_index][start:stop]


class EdfToCsvTests(unittest.TestCase):
    def test_channel_identification_is_explicit(self):
        self.assertTrue(edf_to_csv.classify_eeg_label("FP1-F7")[0])
        self.assertTrue(edf_to_csv.classify_eeg_label("EEG Cz-Ref")[0])
        self.assertFalse(edf_to_csv.classify_eeg_label("ECG1")[0])
        self.assertFalse(edf_to_csv.classify_eeg_label("Unknown Channel")[0])

    def test_conversion_preserves_duplicate_labels_values_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "subject_01.edf"
            source.write_bytes(b"fixture")
            output_dir = root / "processed"
            signals = [
                [1.25, 2.5, 3.75, 5.0],
                [10.0, 20.0, 30.0, 40.0],
                [100.0, np.nan, 300.0, 400.0],
                [7.0, 8.0, 9.0, 10.0],
            ]
            reader = FakeEdfReader(
                ["C3-P3", "T8-P8", "T8-P8", "ECG1"],
                signals,
                [4, 4, 4, 4],
                duration=1,
                dimensions=["uV", "uV", "uV", "mV"],
            )
            with patch.object(edf_to_csv.pyedflib, "EdfReader", return_value=reader):
                report = edf_to_csv.convert_edf_to_csv(
                    source, output_dir, chunk_seconds=0.5
                )

            self.assertEqual(report.output_path.name, "subject_01.edf.csv")
            self.assertEqual(report.sample_count, 4)
            self.assertEqual(report.exported_rows, 4)
            self.assertEqual(report.chunk_seconds, 0.5)
            self.assertEqual(report.sampling_frequency_hz, 4)
            self.assertEqual(report.first_time_seconds, 0)
            self.assertEqual(report.last_time_seconds, 0.75)
            self.assertEqual(report.eeg_channel_labels, ("C3-P3", "T8-P8", "T8-P8"))
            self.assertEqual(report.excluded_channels, (("ECG1", "label identifies an auxiliary/non-EEG signal"),))
            self.assertEqual(report.non_finite_values_by_channel[2], ("T8-P8", 1))

            with report.output_path.open(newline="", encoding="utf-8") as file:
                reader_csv = csv.reader(file)
                self.assertEqual(
                    next(reader_csv), ["Time", "C3-P3", "T8-P8", "T8-P8"]
                )
                rows = list(reader_csv)
            self.assertEqual(len(rows), 4)
            self.assertEqual([float(row[0]) for row in rows], [0.0, 0.25, 0.5, 0.75])
            self.assertEqual(float(rows[0][1]), 1.25)
            self.assertEqual(float(rows[3][2]), 40.0)
            self.assertTrue(np.isnan(float(rows[1][3])))

    def test_missing_source_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.edf"
            with self.assertRaises(FileNotFoundError):
                edf_to_csv.convert_edf_to_csv(missing, Path(directory) / "out")

    def test_inconsistent_channel_sampling_rates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "different_rates.edf"
            source.write_bytes(b"fixture")
            reader = FakeEdfReader(
                ["C3-P3", "C4-P4"],
                [[1, 2, 3, 4], [1, 2]],
                [4, 2],
                duration=1,
            )
            with patch.object(edf_to_csv.pyedflib, "EdfReader", return_value=reader):
                with self.assertRaisesRegex(ValueError, "different sampling frequencies"):
                    edf_to_csv.convert_edf_to_csv(source, root / "out")


if __name__ == "__main__":
    unittest.main()
