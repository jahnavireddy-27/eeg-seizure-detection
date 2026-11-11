import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import (  # noqa: E402
    _write_window_csv,
    FilterConfig,
    calculate_window_samples,
    inspect_signal_quality,
    iter_window_metadata,
    preprocess_raw,
    seconds_to_sample,
    segment_signal,
)


class WindowingTests(unittest.TestCase):
    def test_five_second_window_and_half_overlap_samples_at_256_hz(self):
        self.assertEqual(calculate_window_samples(256, 5, 0.5), (1280, 640, 640))

    def test_time_to_sample_conversion(self):
        self.assertEqual(seconds_to_sample(1015, 256), 259840)
        self.assertEqual(seconds_to_sample(1066, 256), 272896)

    def test_any_overlap_boundary_policy(self):
        signal = np.zeros((2, 15 * 10))
        windows = list(
            segment_signal(
                signal,
                10,
                recording_filename="record.edf",
                patient_id="chb01",
                seizure_intervals=[(5, 7)],
                window_seconds=5,
                overlap_fraction=0.5,
                boundary_policy="any_overlap",
            )
        )
        self.assertEqual(
            [window.label for window in windows],
            ["non-seizure", "seizure", "seizure", "non-seizure", "non-seizure"],
        )
        self.assertEqual(windows[0].end_time_seconds, 5)
        self.assertEqual(windows[0].end_sample, windows[1].start_sample + 25)
        self.assertTrue(all(window.signal.shape[1] == 50 for window in windows))
        self.assertTrue(all(window.patient_id == "chb01" for window in windows))
        self.assertTrue(all(window.recording_filename == "record.edf" for window in windows))

    def test_majority_policy_and_onset_policy_are_explicit(self):
        metadata = list(
            iter_window_metadata(
                150,
                10,
                recording_filename="record.edf",
                patient_id="chb01",
                seizure_intervals=[(5, 8)],
                window_seconds=5,
                overlap_fraction=0.5,
                boundary_policy="majority_overlap",
                majority_overlap_threshold=0.5,
            )
        )
        self.assertEqual(
            [window.label for window in metadata],
            ["non-seizure", "seizure", "seizure", "non-seizure", "non-seizure"],
        )
        onset_windows = list(
            iter_window_metadata(
                150,
                10,
                recording_filename="record.edf",
                patient_id="chb01",
                seizure_intervals=[(5, 8)],
                window_seconds=5,
                overlap_fraction=0.5,
                boundary_policy="onset",
            )
        )
        self.assertEqual(
            [window.label for window in onset_windows],
            ["non-seizure", "seizure", "seizure", "non-seizure", "non-seizure"],
        )

    def test_windows_do_not_extend_past_duration(self):
        windows = list(
            iter_window_metadata(
                103,
                10,
                recording_filename="record.edf",
                patient_id="chb01",
                window_seconds=5,
                overlap_fraction=0.5,
            )
        )
        self.assertTrue(windows)
        self.assertTrue(all(window.end_sample <= 103 for window in windows))
        self.assertEqual(windows[-1].end_sample, 100)

    def test_invalid_or_out_of_recording_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            list(
                iter_window_metadata(
                    100,
                    10,
                    recording_filename="record.edf",
                    patient_id="chb01",
                    seizure_intervals=[(8, 11)],
                    window_seconds=5,
                )
            )
        with self.assertRaises(ValueError):
            list(
                iter_window_metadata(
                    100,
                    10,
                    recording_filename="record.edf",
                    patient_id="chb01",
                    seizure_intervals=[(4, 3)],
                    window_seconds=5,
                )
            )


class FilteringAndQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mne

        cls.mne = mne

    def test_filtering_preserves_input_raw_and_channel_count(self):
        sfreq = 256
        times = np.arange(sfreq * 12) / sfreq
        data = np.vstack(
            [
                np.sin(2 * np.pi * 10 * times),
                np.sin(2 * np.pi * 60 * times),
            ]
        ) * 1e-6
        raw = self.mne.io.RawArray(
            data.copy(),
            self.mne.create_info(["C3", "C4"], sfreq, ch_types=["eeg", "eeg"]),
            verbose="ERROR",
        )
        before = raw.get_data().copy()
        filtered = preprocess_raw(raw, FilterConfig())
        np.testing.assert_array_equal(raw.get_data(), before)
        self.assertEqual(filtered.ch_names, raw.ch_names)
        self.assertEqual(filtered.info["sfreq"], sfreq)
        self.assertTrue(filtered.preload)
        self.assertFalse(np.array_equal(filtered.get_data(), before))

    def test_filter_limits_are_checked_against_nyquist(self):
        raw = self.mne.io.RawArray(
            np.zeros((1, 2560)),
            self.mne.create_info(["C3"], 128, ch_types=["eeg"]),
            verbose="ERROR",
        )
        with self.assertRaises(ValueError):
            preprocess_raw(raw, FilterConfig(h_freq=65))

    def test_filter_memory_cap_rejects_before_loading_recording(self):
        raw = self.mne.io.RawArray(
            np.zeros((1, 2560)),
            self.mne.create_info(["C3"], 256, ch_types=["eeg"]),
            verbose="ERROR",
        )
        with self.assertRaisesRegex(MemoryError, "exceeding the configured"):
            preprocess_raw(raw, max_memory_bytes=1)

    def test_window_csv_writer_accepts_a_streaming_iterable(self):
        windows = iter_window_metadata(
            2560,
            256,
            recording_filename="recording.edf",
            patient_id="chb01",
            seizure_intervals=[(2, 4)],
            window_seconds=5,
            overlap_fraction=0.5,
            boundary_policy="any_overlap",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.csv"
            _write_window_csv(path, windows)
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["recording_filename"], "recording.edf")

    def test_quality_report_does_not_remove_flat_channels(self):
        sfreq = 100
        data = np.vstack(
            [
                np.sin(2 * np.pi * 8 * np.arange(1000) / sfreq) * 1e-6,
                np.zeros(1000),
            ]
        )
        raw = self.mne.io.RawArray(
            data,
            self.mne.create_info(["C3", "C4"], sfreq, ch_types=["eeg", "eeg"]),
            verbose="ERROR",
        )
        report = inspect_signal_quality(raw, expected_eeg_channels=2)
        self.assertEqual(report["eeg_channel_count"], 2)
        self.assertIn("flat_channel", report["channels"][1]["flags"])
        self.assertEqual(report["channels"][0]["non_finite_sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
