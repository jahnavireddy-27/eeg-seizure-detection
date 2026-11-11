#!/usr/bin/env python3
"""Convert one EDF recording to a chunk-written EEG CSV without altering source."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyedflib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "csv"
DEFAULT_CHUNK_SECONDS = 30.0
TIME_COLUMN = "Time"
KNOWN_EEG_ELECTRODES = {
    "AF3", "AF4", "AF7", "AF8", "AFZ",
    "C1", "C2", "C3", "C4", "C5", "C6", "CZ",
    "CB1", "CB2",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FZ",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCZ",
    "FT9", "FT10",
    "FP1", "FP2",
    "M1", "M2", "O1", "O2",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "T7", "T8", "T9", "T10", "TP9", "TP10",
    "A1", "A2",
}
AUXILIARY_LABEL_PATTERN = re.compile(
    r"(?:^|[\s:_-])(ECG|EKG|EOG|EMG|VNS|RESP|"
    r"SAO2|SPO2|PLETH|PULSE|PHOTIC|TRIG(?:GER)?|MARKER|"
    r"STATUS|ANNOTATION)(?:\d+)?(?:$|[\s:_-])",
    re.IGNORECASE,
)
EEG_PREFIX_PATTERN = re.compile(r"^\s*EEG(?:[\s:_-]|$)", re.IGNORECASE)
ELECTRODE_PAIR_PATTERN = re.compile(
    r"^\s*([A-Z][A-Z0-9]*)\s*[-/]\s*([A-Z][A-Z0-9]*)\s*$",
    re.IGNORECASE,
)
TIME_TOLERANCE_SECONDS = 1e-9


@dataclass(frozen=True)
class ChannelSelection:
    signal_index: int
    label: str
    dimension: str
    sample_frequency_hz: float
    sample_count: int


@dataclass(frozen=True)
class ConversionReport:
    source_path: Path
    output_path: Path
    eeg_channel_labels: tuple[str, ...]
    excluded_channels: tuple[tuple[str, str], ...]
    sampling_frequency_hz: float
    duration_seconds: float
    chunk_seconds: float
    sample_count: int
    exported_rows: int
    non_finite_values_by_channel: tuple[tuple[str, int], ...]
    first_time_seconds: float
    last_time_seconds: float


def classify_eeg_label(label: str) -> tuple[bool, str]:
    """Classify a channel from its EDF label without silently dropping it.

    EDF signal headers do not contain a universally populated channel-type
    field. Explicit auxiliary labels are excluded; explicit EEG labels and
    recognized scalp-electrode labels/pairs are selected.
    """
    normalized = label.strip()
    if AUXILIARY_LABEL_PATTERN.search(normalized):
        return False, "label identifies an auxiliary/non-EEG signal"
    if EEG_PREFIX_PATTERN.match(normalized):
        return True, "explicit EEG label"

    upper_label = normalized.upper()
    if upper_label in KNOWN_EEG_ELECTRODES:
        return True, "recognized EEG electrode label"
    pair_match = ELECTRODE_PAIR_PATTERN.fullmatch(upper_label)
    if pair_match and all(
        electrode in KNOWN_EEG_ELECTRODES for electrode in pair_match.groups()
    ):
        return True, "recognized scalp EEG electrode pair"
    return False, "channel type not identifiable from EDF label"


def _validate_metadata(
    reader: Any,
    source_path: Path,
) -> tuple[list[ChannelSelection], list[tuple[str, str]], float, float, int]:
    headers = reader.getSignalHeaders()
    labels = reader.getSignalLabels()
    frequencies = np.asarray(reader.getSampleFrequencies(), dtype=float)
    sample_counts = np.asarray(reader.getNSamples(), dtype=np.int64)
    if len(headers) != len(labels) or len(headers) != len(frequencies):
        raise ValueError("EDF channel metadata arrays have inconsistent lengths.")
    if len(headers) != len(sample_counts):
        raise ValueError("EDF channel sample-count metadata has an inconsistent length.")

    selected: list[ChannelSelection] = []
    excluded: list[tuple[str, str]] = []
    for index, (header, label) in enumerate(zip(headers, labels)):
        is_eeg, reason = classify_eeg_label(label)
        if not is_eeg:
            excluded.append((label, reason))
            continue
        selected.append(
            ChannelSelection(
                signal_index=index,
                label=label,
                dimension=str(header.get("dimension", "")),
                sample_frequency_hz=float(frequencies[index]),
                sample_count=int(sample_counts[index]),
            )
        )

    if not selected:
        raise ValueError("No EEG channels could be identified from this EDF's labels.")
    if any(channel.label == TIME_COLUMN for channel in selected):
        raise ValueError(
            f"An EEG channel is named {TIME_COLUMN!r}, which conflicts with the required "
            "time column."
        )

    reference_frequency = selected[0].sample_frequency_hz
    reference_count = selected[0].sample_count
    if any(
        not math.isclose(
            channel.sample_frequency_hz,
            reference_frequency,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for channel in selected
    ):
        frequencies_found = sorted({channel.sample_frequency_hz for channel in selected})
        raise ValueError(
            "EEG channels have different sampling frequencies "
            f"({frequencies_found}); conversion to one time-column CSV would require "
            "resampling, which this baseline converter does not perform."
        )
    if any(channel.sample_count != reference_count for channel in selected):
        counts_found = sorted({channel.sample_count for channel in selected})
        raise ValueError(
            "EEG channels have different sample counts "
            f"({counts_found}); a rectangular CSV cannot be produced without "
            "an unspecified alignment or resampling decision."
        )

    duration = float(reader.file_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"EDF reports an invalid recording duration: {duration!r}.")
    expected_samples = duration * reference_frequency
    if abs(expected_samples - reference_count) > 1.0:
        raise ValueError(
            f"EDF duration ({duration:g} s) and sample metadata "
            f"({reference_count} samples at {reference_frequency:g} Hz) disagree."
        )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    return selected, excluded, reference_frequency, duration, reference_count


def _make_chunk_frame(
    reader: Any,
    channels: list[ChannelSelection],
    start_sample: int,
    stop_sample: int,
    sfreq: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    count = stop_sample - start_sample
    columns = [
        np.asarray(
            reader.readSignal(
                channel.signal_index,
                start=start_sample,
                n=count,
                digital=False,
            ),
            dtype=np.float64,
        )
        for channel in channels
    ]
    for channel, values in zip(channels, columns):
        if values.size != count:
            raise OSError(
                f"EDF reader returned {values.size} samples for {channel.label!r}; "
                f"expected {count}."
            )
    signal_matrix = np.column_stack(columns)
    time_values = np.arange(start_sample, stop_sample, dtype=np.float64) / sfreq
    frame = pd.DataFrame(signal_matrix, columns=[channel.label for channel in channels])
    frame.insert(0, TIME_COLUMN, time_values, allow_duplicates=True)
    return frame, signal_matrix


def _validate_exported_csv(
    path: Path,
    expected_columns: list[str],
    expected_rows: int,
    sfreq: float,
    chunk_rows: int,
) -> tuple[int, float, float, list[int]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        header = next(csv.reader(file), None)
    if header != expected_columns:
        raise ValueError(
            f"CSV columns do not match EDF metadata. Expected {expected_columns!r}, "
            f"found {header!r}."
        )

    row_count = 0
    first_time: float | None = None
    last_time: float | None = None
    non_finite_counts = np.zeros(len(expected_columns) - 1, dtype=np.int64)
    expected_interval = 1.0 / sfreq
    for chunk in pd.read_csv(
        path,
        chunksize=chunk_rows,
        header=None,
        skiprows=1,
        names=list(range(len(expected_columns))),
    ):
        if chunk.shape[1] != len(expected_columns):
            raise ValueError("CSV row width does not match the EDF channel metadata.")
        times = chunk.iloc[:, 0].to_numpy(dtype=np.float64)
        if times.size == 0 or not np.isfinite(times).all():
            raise ValueError("CSV time column is empty or contains non-finite values.")
        if first_time is None:
            first_time = float(times[0])
        if times.size > 1 and not np.allclose(
            np.diff(times),
            expected_interval,
            rtol=0.0,
            atol=TIME_TOLERANCE_SECONDS,
        ):
            raise ValueError("CSV time column does not have the expected sampling interval.")
        if last_time is not None and not math.isclose(
            float(times[0]) - last_time,
            expected_interval,
            rel_tol=0.0,
            abs_tol=TIME_TOLERANCE_SECONDS,
        ):
            raise ValueError("CSV time interval is discontinuous between output chunks.")
        last_time = float(times[-1])
        signal_values = chunk.iloc[:, 1:].to_numpy(dtype=np.float64)
        non_finite_counts += np.count_nonzero(~np.isfinite(signal_values), axis=0)
        row_count += len(chunk)

    if row_count != expected_rows:
        raise ValueError(
            f"Exported CSV has {row_count} rows; EDF metadata reports {expected_rows} samples."
        )
    expected_last_time = (expected_rows - 1) / sfreq
    if first_time is None or not math.isclose(
        first_time, 0.0, rel_tol=0.0, abs_tol=TIME_TOLERANCE_SECONDS
    ):
        raise ValueError(f"CSV first timestamp is {first_time}, expected 0 seconds.")
    if not math.isclose(
        last_time if last_time is not None else float("nan"),
        expected_last_time,
        rel_tol=0.0,
        abs_tol=TIME_TOLERANCE_SECONDS,
    ):
        raise ValueError(
            f"CSV last timestamp is {last_time}; expected {expected_last_time} seconds."
        )
    return row_count, first_time, float(last_time), non_finite_counts.tolist()


def convert_edf_to_csv(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> ConversionReport:
    """Convert one EDF to its deterministic `.edf.csv` output incrementally."""
    source_path = Path(input_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"EDF file does not exist: {source_path}")
    if source_path.suffix.lower() != ".edf":
        raise ValueError(f"Input must be an EDF file: {source_path}")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be a positive finite number.")

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{source_path.name}.csv"
    temporary_path: Path | None = None

    try:
        with pyedflib.EdfReader(str(source_path)) as reader:
            channels, excluded, sfreq, duration, sample_count = _validate_metadata(
                reader, source_path
            )
            samples_per_chunk = max(1, int(round(chunk_seconds * sfreq)))
            expected_columns = [TIME_COLUMN, *(channel.label for channel in channels)]
            non_finite_source_counts = np.zeros(len(channels), dtype=np.int64)
            rows_written = 0

            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                suffix=".tmp",
                prefix=f".{output_path.name}.",
                dir=destination_dir,
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                for start_sample in range(0, sample_count, samples_per_chunk):
                    stop_sample = min(start_sample + samples_per_chunk, sample_count)
                    frame, signal_matrix = _make_chunk_frame(
                        reader, channels, start_sample, stop_sample, sfreq
                    )
                    non_finite_source_counts += np.count_nonzero(
                        ~np.isfinite(signal_matrix), axis=0
                    )
                    frame.to_csv(
                        output_file,
                        index=False,
                        header=(start_sample == 0),
                        float_format="%.17g",
                        na_rep="NaN",
                    )
                    rows_written += len(frame)
                    del frame, signal_matrix
                output_file.flush()
                os.fsync(output_file.fileno())

        if rows_written != sample_count:
            raise ValueError(
                f"Converter wrote {rows_written} rows; EDF reports {sample_count} samples."
            )
        validated_rows, first_time, last_time, exported_non_finite = _validate_exported_csv(
            temporary_path,
            expected_columns,
            sample_count,
            sfreq,
            samples_per_chunk,
        )
        if exported_non_finite != non_finite_source_counts.tolist():
            raise ValueError(
                "Non-finite signal value counts changed during CSV serialization: "
                f"source={non_finite_source_counts.tolist()}, "
                f"CSV={exported_non_finite}."
            )
        os.replace(temporary_path, output_path)
        temporary_path = None

        report = ConversionReport(
            source_path=source_path,
            output_path=output_path,
            eeg_channel_labels=tuple(channel.label for channel in channels),
            excluded_channels=tuple(excluded),
            sampling_frequency_hz=sfreq,
            duration_seconds=duration,
            chunk_seconds=chunk_seconds,
            sample_count=sample_count,
            exported_rows=validated_rows,
            non_finite_values_by_channel=tuple(
                (channel.label, int(count))
                for channel, count in zip(channels, exported_non_finite)
            ),
            first_time_seconds=first_time,
            last_time_seconds=last_time,
        )
        log_conversion(report, channels)
        return report
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def log_conversion(report: ConversionReport, channels: list[ChannelSelection]) -> None:
    print(f"Source EDF: {report.source_path}")
    print(f"Output CSV: {report.output_path}")
    print(f"Original recording filename: {report.source_path.name}")
    print(f"Recording duration: {report.duration_seconds:.6f} seconds")
    print(f"Sampling frequency: {report.sampling_frequency_hz:g} Hz")
    print(f"Read/write chunk size: {report.chunk_seconds:g} seconds")
    print(f"Samples/CSV rows: {report.sample_count}/{report.exported_rows}")
    print(f"pyEDFlib version: {pyedflib.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(
        f"Time range: {report.first_time_seconds:.9f} to "
        f"{report.last_time_seconds:.9f} seconds "
        f"(interval {1 / report.sampling_frequency_hz:.12g} seconds)"
    )
    print(f"EEG channel count: {len(report.eeg_channel_labels)}")
    print("EEG channel labels and EDF units:")
    for channel in channels:
        print(f"  {channel.label} (EDF unit: {channel.dimension or 'unspecified'})")
    print(f"Non-EEG/unclassified channels excluded: {len(report.excluded_channels)}")
    for label, reason in report.excluded_channels:
        print(f"  {label!r}: {reason}")
    channels_with_non_finite = [
        (label, count)
        for label, count in report.non_finite_values_by_channel
        if count
    ]
    if channels_with_non_finite:
        print("Non-finite signal values preserved as NaN:")
        for label, count in channels_with_non_finite:
            print(f"  {label}: {count}")
    else:
        print("Non-finite signal values: none")
    print("CSV validation: labels, sample count, first/last time, and time interval passed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one EDF recording to chunk-written EEG CSV."
    )
    parser.add_argument("edf_path", type=Path, help="Path to one source EDF recording.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help="Number of seconds read per chunk (default: 30).",
    )
    args = parser.parse_args()
    try:
        convert_edf_to_csv(
            args.edf_path,
            args.output_dir,
            chunk_seconds=args.chunk_seconds,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"EDF conversion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
