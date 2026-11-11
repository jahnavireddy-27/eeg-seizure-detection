#!/usr/bin/env python3
"""Create a window metadata manifest using validated CHB-MIT annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
import pyedflib

from edf_to_csv import classify_eeg_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_ANNOTATION_SOURCE = PROJECT_ROOT / "results" / "chb01_annotation_audit.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_manifests"
DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_OVERLAP_FRACTION = 0.5
DEFAULT_POLICY = "any_overlap"
POLICIES = ("any_overlap", "majority_overlap", "center_in_seizure")
SAMPLE_GRID_TOLERANCE_SECONDS = 1e-9


@dataclass(frozen=True)
class SeizureInterval:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class RecordingAnnotation:
    status: Literal["confirmed_seizure", "confirmed_no_seizure", "unknown"]
    intervals: tuple[SeizureInterval, ...]
    reason: str = ""


@dataclass(frozen=True)
class WindowConfiguration:
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION
    label_policy: str = DEFAULT_POLICY

    def validate(self) -> None:
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be a positive finite number.")
        if (
            not math.isfinite(self.overlap_fraction)
            or not 0 <= self.overlap_fraction < 1
        ):
            raise ValueError("overlap_fraction must be in [0, 1).")
        if self.label_policy not in POLICIES:
            raise ValueError(f"label_policy must be one of {POLICIES}.")

    def sample_counts(self, sampling_frequency_hz: float) -> tuple[int, int, int]:
        self.validate()
        if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
            raise ValueError("Sampling frequency must be positive and finite.")
        exact_window_samples = self.window_seconds * sampling_frequency_hz
        window_samples = int(round(exact_window_samples))
        if not math.isclose(
            exact_window_samples,
            window_samples,
            rel_tol=0.0,
            abs_tol=SAMPLE_GRID_TOLERANCE_SECONDS * sampling_frequency_hz,
        ):
            raise ValueError(
                f"Window duration {self.window_seconds:g} s does not map to a whole "
                f"sample count at {sampling_frequency_hz:g} Hz."
            )
        exact_stride_samples = (
            window_samples * (1.0 - self.overlap_fraction)
        )
        stride_samples = int(round(exact_stride_samples))
        if not math.isclose(
            exact_stride_samples,
            stride_samples,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "The configured overlap does not produce an integer-sample stride."
            )
        if window_samples < 1 or stride_samples < 1:
            raise ValueError("Window and stride must each span at least one sample.")
        overlap_samples = window_samples - stride_samples
        return window_samples, overlap_samples, stride_samples


@dataclass(frozen=True)
class WindowRecord:
    patient_id: str
    recording_filename: str
    recording_id: str
    window_index: int
    start_sample: int
    end_sample_exclusive: int
    start_time_seconds: float
    end_time_seconds: float
    seizure_overlap_seconds: float | None
    seizure_label: str
    annotation_status: str
    label_policy: str
    window_samples: int
    overlap_samples: int
    stride_samples: int
    overlap_fraction: float
    channel_count: int
    channel_labels: tuple[str, ...]
    sampling_frequency_hz: float
    relative_path: str = ""


def _validate_intervals(
    intervals: Sequence[SeizureInterval],
) -> tuple[SeizureInterval, ...]:
    ordered = tuple(sorted(intervals, key=lambda item: (item.start_seconds, item.end_seconds)))
    previous_end: float | None = None
    for interval in ordered:
        if not math.isfinite(interval.start_seconds) or not math.isfinite(interval.end_seconds):
            raise ValueError("Seizure interval values must be finite.")
        if interval.start_seconds < 0:
            raise ValueError("Seizure starts must be non-negative.")
        if interval.end_seconds <= interval.start_seconds:
            raise ValueError("Seizure end must be later than its start.")
        if previous_end is not None and interval.start_seconds < previous_end:
            raise ValueError("Seizure intervals overlap or conflict.")
        previous_end = interval.end_seconds
    return ordered


def seizure_overlap_seconds(
    window_start_seconds: float,
    window_end_seconds: float,
    intervals: Sequence[SeizureInterval],
) -> float:
    """Measure overlap using half-open intervals [start, end)."""
    overlap = 0.0
    for interval in intervals:
        overlap += max(
            0.0,
            min(window_end_seconds, interval.end_seconds)
            - max(window_start_seconds, interval.start_seconds),
        )
    return overlap


def label_window(
    window_start_seconds: float,
    window_end_seconds: float,
    annotation: RecordingAnnotation,
    policy: str,
) -> tuple[str, float | None]:
    if window_end_seconds <= window_start_seconds:
        raise ValueError("Window end must be later than window start.")
    if policy not in POLICIES:
        raise ValueError(f"label_policy must be one of {POLICIES}.")
    if annotation.status == "unknown":
        return "unknown", None
    if annotation.status == "confirmed_no_seizure":
        if annotation.intervals:
            raise ValueError("A confirmed no-seizure annotation cannot contain intervals.")
        return "non_seizure", 0.0
    if annotation.status != "confirmed_seizure":
        raise ValueError(f"Unsupported annotation status: {annotation.status}")
    if not annotation.intervals:
        raise ValueError("Confirmed seizure status requires at least one interval.")

    intervals = _validate_intervals(annotation.intervals)
    window_duration = window_end_seconds - window_start_seconds
    overlap = seizure_overlap_seconds(
        window_start_seconds, window_end_seconds, intervals
    )
    if policy == "any_overlap":
        positive = overlap > 0.0
    elif policy == "majority_overlap":
        positive = overlap > window_duration / 2.0
    else:
        center = (window_start_seconds + window_end_seconds) / 2.0
        positive = any(
            interval.start_seconds <= center < interval.end_seconds
            for interval in intervals
        )
    return ("seizure" if positive else "non_seizure"), overlap


def generate_window_records(
    *,
    patient_id: str,
    recording_filename: str,
    sample_count: int,
    sampling_frequency_hz: float,
    channel_labels: Sequence[str],
    annotation: RecordingAnnotation,
    configuration: WindowConfiguration = WindowConfiguration(),
    relative_path: str | None = None,
) -> Iterator[WindowRecord]:
    """Yield full windows anchored at sample zero; omit any incomplete tail."""
    if sample_count < 0:
        raise ValueError("sample_count cannot be negative.")
    if not channel_labels:
        raise ValueError("At least one EEG channel label is required.")
    labels = tuple(channel_labels)
    if len(labels) != len(channel_labels):
        raise ValueError("Channel labels must be a finite sequence.")
    intervals = _validate_intervals(annotation.intervals)
    if annotation.status == "confirmed_no_seizure" and intervals:
        raise ValueError("A confirmed no-seizure annotation cannot contain intervals.")
    if annotation.status == "confirmed_seizure" and not intervals:
        raise ValueError("Confirmed seizure status requires seizure intervals.")

    window_samples, overlap_samples, stride_samples = configuration.sample_counts(
        sampling_frequency_hz
    )
    duration_seconds = sample_count / sampling_frequency_hz
    if any(item.end_seconds > duration_seconds for item in intervals):
        raise ValueError("A seizure interval extends beyond the EDF recording duration.")

    window_index = 0
    start_sample = 0
    while start_sample + window_samples <= sample_count:
        end_sample = start_sample + window_samples
        start_time = start_sample / sampling_frequency_hz
        end_time = end_sample / sampling_frequency_hz
        label, overlap = label_window(
            start_time,
            end_time,
            RecordingAnnotation(annotation.status, intervals, annotation.reason),
            configuration.label_policy,
        )
        yield WindowRecord(
            patient_id=patient_id,
            recording_filename=recording_filename,
            recording_id=Path(recording_filename).stem,
            window_index=window_index,
            start_sample=start_sample,
            end_sample_exclusive=end_sample,
            start_time_seconds=start_time,
            end_time_seconds=end_time,
            seizure_overlap_seconds=overlap,
            seizure_label=label,
            annotation_status=annotation.status,
            label_policy=configuration.label_policy,
            window_samples=window_samples,
            overlap_samples=overlap_samples,
            stride_samples=stride_samples,
            overlap_fraction=configuration.overlap_fraction,
            channel_count=len(labels),
            channel_labels=labels,
            sampling_frequency_hz=sampling_frequency_hz,
            relative_path=relative_path or recording_filename,
        )
        window_index += 1
        start_sample += stride_samples


def load_validated_annotations(
    audit_path: Path,
    patient_id: str | None = None,
) -> dict[str | tuple[str, str], RecordingAnnotation]:
    """Load validated annotations from a legacy audit or corpus-wide manifest."""
    annotations: dict[str | tuple[str, str], RecordingAnnotation] = {}
    with audit_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if {"patient_id", "relative_path", "seizure_status", "seizure_intervals"} <= fields:
            for row in reader:
                row_patient = row["patient_id"].strip()
                if patient_id is not None and row_patient.casefold() != patient_id.casefold():
                    continue
                relative_path = Path(row["relative_path"]).as_posix()
                key = (row_patient.casefold(), relative_path)
                if key in annotations:
                    raise ValueError(f"Duplicate corpus annotation identity: {key}.")
                if row.get("annotation_agreement", "").strip() == "disagreement":
                    annotations[key] = RecordingAnnotation(
                        "unknown", (), "summary and sidecar annotations conflict"
                    )
                    continue
                status = row["seizure_status"].strip()
                if status == "no_seizures_annotated":
                    annotations[key] = RecordingAnnotation("confirmed_no_seizure", ())
                    continue
                if status != "seizures_annotated":
                    annotations[key] = RecordingAnnotation(
                        "unknown",
                        (),
                        "corpus manifest does not confirm usable seizure annotations",
                    )
                    continue
                try:
                    interval_values = json.loads(row["seizure_intervals"])
                    intervals = _validate_intervals(
                        tuple(
                            SeizureInterval(
                                float(item["start_seconds"]),
                                float(item["end_seconds"]),
                            )
                            for item in interval_values
                        )
                    )
                    duration = float(row["recording_duration_seconds"])
                    if any(interval.end_seconds > duration for interval in intervals):
                        raise ValueError("seizure interval exceeds EDF duration")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    annotations[key] = RecordingAnnotation(
                        "unknown", (), f"invalid corpus manifest intervals: {exc}"
                    )
                    continue
                if not intervals:
                    annotations[key] = RecordingAnnotation(
                        "unknown", (), "corpus manifest annotations are incomplete or invalid"
                    )
                    continue
                annotations[key] = RecordingAnnotation("confirmed_seizure", intervals)
            return annotations

        required_fields = {
            "recording",
            "downloaded",
            "contains_seizure",
            "summary_seizure_count",
            "seizure_intervals",
            "validation_status",
            "validation_issues",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                "Annotation audit is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
        for row in reader:
            filename = row["recording"].strip()
            if not filename:
                continue
            annotation_key: str | tuple[str, str] = filename
            if {"patient_id", "relative_path"} <= fields:
                row_patient = row["patient_id"].strip()
                if patient_id is not None and row_patient.casefold() != patient_id.casefold():
                    continue
                annotation_key = (
                    row_patient.casefold(),
                    Path(row["relative_path"]).as_posix(),
                )
            if annotation_key in annotations:
                raise ValueError(f"Duplicate annotation audit row for {annotation_key}.")
            if row.get("annotation_agreement", "").strip() == "disagreement":
                annotations[annotation_key] = RecordingAnnotation(
                    "unknown", (), "summary and sidecar annotations conflict"
                )
                continue
            if row["validation_status"].strip().upper() != "PASS":
                annotations[annotation_key] = RecordingAnnotation(
                    "unknown",
                    (),
                    row["validation_issues"].strip() or "annotation audit did not pass",
                )
                continue
            if row["downloaded"].strip().lower() != "true":
                annotations[annotation_key] = RecordingAnnotation(
                    "unknown", (), "annotation audit does not confirm a local EDF"
                )
                continue
            try:
                declared_count = int(row["summary_seizure_count"])
                interval_values = json.loads(row["seizure_intervals"])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                annotations[annotation_key] = RecordingAnnotation(
                    "unknown", (), f"invalid audit annotation fields: {exc}"
                )
                continue
            if not isinstance(interval_values, list):
                annotations[annotation_key] = RecordingAnnotation(
                    "unknown", (), "seizure_intervals is not a JSON list"
                )
                continue
            try:
                intervals = _validate_intervals(
                    tuple(
                        SeizureInterval(
                            float(item["start_seconds"]),
                            float(item["end_seconds"]),
                        )
                        for item in interval_values
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                annotations[annotation_key] = RecordingAnnotation(
                    "unknown", (), f"invalid seizure intervals: {exc}"
                )
                continue
            contains_seizure = row["contains_seizure"].strip().lower()
            if contains_seizure == "true" and declared_count > 0 and len(intervals) == declared_count:
                annotations[annotation_key] = RecordingAnnotation(
                    "confirmed_seizure", intervals
                )
            elif (
                contains_seizure == "false"
                and declared_count == 0
                and not intervals
            ):
                annotations[annotation_key] = RecordingAnnotation(
                    "confirmed_no_seizure", ()
                )
            else:
                annotations[annotation_key] = RecordingAnnotation(
                    "unknown",
                    (),
                    "audit seizure flag, count, and interval list are inconsistent",
                )
    return annotations


def read_edf_metadata(
    path: Path,
) -> tuple[float, int, tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Read channel labels/count/rate from an EDF without loading EEG samples."""
    with pyedflib.EdfReader(str(path)) as reader:
        labels = reader.getSignalLabels()
        frequencies = np.asarray(reader.getSampleFrequencies(), dtype=float)
        sample_counts = np.asarray(reader.getNSamples(), dtype=np.int64)
        if len(labels) != len(frequencies) or len(labels) != len(sample_counts):
            raise ValueError(f"{path.name}: inconsistent EDF channel metadata lengths.")

        selection = [classify_eeg_label(label) for label in labels]
        selected_indices = [
            index for index, (is_eeg, _) in enumerate(selection) if is_eeg
        ]
        excluded = tuple(
            (label, selection[index][1])
            for index, label in enumerate(labels)
            if not selection[index][0]
        )
        if not selected_indices:
            raise ValueError(f"{path.name}: no EEG channels identified from EDF labels.")
        selected_rates = frequencies[selected_indices]
        selected_counts = sample_counts[selected_indices]
        if not np.isfinite(selected_rates).all() or np.any(selected_rates <= 0):
            raise ValueError(f"{path.name}: invalid EEG channel sampling frequency.")
        if not np.allclose(selected_rates, selected_rates[0], rtol=0, atol=1e-9):
            details = ", ".join(
                f"{labels[index]}={frequencies[index]:g} Hz"
                for index in selected_indices
            )
            raise ValueError(
                f"{path.name}: inconsistent EEG channel sampling frequencies: {details}"
            )
        if not np.all(selected_counts == selected_counts[0]):
            details = ", ".join(
                f"{labels[index]}={sample_counts[index]} samples"
                for index in selected_indices
            )
            raise ValueError(
                f"{path.name}: inconsistent EEG channel sample counts: {details}"
            )
        return (
            float(selected_rates[0]),
            int(selected_counts[0]),
            tuple(labels[index] for index in selected_indices),
            excluded,
        )


def _serialize_window(record: WindowRecord) -> dict[str, str | int]:
    row: dict[str, str | int] = asdict(record)
    row["channel_labels"] = json.dumps(record.channel_labels, ensure_ascii=False)
    row["start_time_seconds"] = f"{record.start_time_seconds:.9f}"
    row["end_time_seconds"] = f"{record.end_time_seconds:.9f}"
    row["seizure_overlap_seconds"] = (
        ""
        if record.seizure_overlap_seconds is None
        else f"{record.seizure_overlap_seconds:.9f}"
    )
    row["sampling_frequency_hz"] = f"{record.sampling_frequency_hz:g}"
    row["overlap_fraction"] = f"{record.overlap_fraction:g}"
    return row


def write_manifest(path: Path, records: Iterable[WindowRecord]) -> int:
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
        "seizure_overlap_seconds",
        "seizure_label",
        "annotation_status",
        "label_policy",
        "window_samples",
        "overlap_samples",
        "stride_samples",
        "overlap_fraction",
        "channel_count",
        "channel_labels",
        "sampling_frequency_hz",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            row_count = 0
            for record in records:
                writer.writerow(_serialize_window(record))
                row_count += 1
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return row_count


def create_recording_manifest(
    *,
    edf_path: Path,
    data_dir: Path,
    annotation_source: Path,
    output_path: Path,
    patient_id: str,
    configuration: WindowConfiguration,
) -> tuple[int, str]:
    """Write one recording's windows incrementally, without retaining them."""
    edf_path = edf_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    relative_path = edf_path.relative_to(data_dir).as_posix()
    annotations = load_validated_annotations(annotation_source, patient_id=patient_id)
    annotation = annotations.get(
        (patient_id.casefold(), relative_path),
        RecordingAnnotation(
            "unknown", (), "recording has no validated corpus annotation"
        ),
    )
    sfreq, sample_count, channel_labels, excluded_channels = read_edf_metadata(edf_path)
    window_count = 0
    labels: dict[str, int] = {}

    def records() -> Iterator[WindowRecord]:
        nonlocal window_count
        for record in generate_window_records(
            patient_id=patient_id,
            recording_filename=edf_path.name,
            sample_count=sample_count,
            sampling_frequency_hz=sfreq,
            channel_labels=channel_labels,
            annotation=annotation,
            configuration=configuration,
            relative_path=relative_path,
        ):
            window_count += 1
            labels[record.seizure_label] = labels.get(record.seizure_label, 0) + 1
            yield record

    write_manifest(output_path, records())
    message = (
        f"{relative_path}: {window_count} windows; "
        + ", ".join(f"{key}={value}" for key, value in sorted(labels.items()))
        + f"; annotation={annotation.status}"
    )
    if annotation.reason:
        message += f" ({annotation.reason})"
    if excluded_channels:
        message += "; excluded non-EEG/unclassified signals: " + ", ".join(
            f"{label!r}: {reason}" for label, reason in excluded_channels
        )
    return window_count, message


def create_manifest(
    *,
    data_dir: Path,
    annotation_source: Path,
    output_path: Path,
    patient_id: str,
    configuration: WindowConfiguration,
) -> tuple[int, list[str]]:
    annotations = load_validated_annotations(annotation_source)
    edf_paths = sorted(
        path
        for path in data_dir.rglob(f"{patient_id}_*.edf")
        if path.is_file()
    )
    if not edf_paths:
        raise FileNotFoundError(f"No {patient_id}_*.edf files found under {data_dir}.")

    messages = []
    total_windows = 0

    def records() -> Iterator[WindowRecord]:
        nonlocal total_windows
        legacy_basename_counts: dict[str, int] = {}
        for filename in annotations:
            if isinstance(filename, str):
                legacy_basename_counts[filename.casefold()] = (
                    legacy_basename_counts.get(filename.casefold(), 0) + 1
                )
        for edf_path in edf_paths:
            sfreq, sample_count, channel_labels, excluded_channels = read_edf_metadata(
                edf_path
            )
            relative_path = edf_path.relative_to(data_dir).as_posix()
            composite_key = (patient_id.casefold(), relative_path)
            annotation = annotations.get(composite_key)
            if annotation is None and legacy_basename_counts.get(edf_path.name.casefold()) == 1:
                annotation = annotations.get(edf_path.name)
            if annotation is None:
                annotation = RecordingAnnotation(
                    "unknown", (), "recording is absent or ambiguous in annotation source"
                )
            labels: dict[str, int] = {}
            window_count = 0
            for record in generate_window_records(
                patient_id=patient_id,
                recording_filename=edf_path.name,
                sample_count=sample_count,
                sampling_frequency_hz=sfreq,
                channel_labels=channel_labels,
                annotation=annotation,
                configuration=configuration,
                relative_path=relative_path,
            ):
                total_windows += 1
                window_count += 1
                labels[record.seizure_label] = labels.get(record.seizure_label, 0) + 1
                yield record
            message = (
                f"{relative_path}: {window_count} windows; "
                + ", ".join(f"{key}={value}" for key, value in sorted(labels.items()))
                + f"; annotation={annotation.status}"
            )
            if annotation.reason:
                message += f" ({annotation.reason})"
            if excluded_channels:
                message += "; excluded non-EEG/unclassified signals: " + ", ".join(
                    f"{label!r}: {reason}" for label, reason in excluded_channels
                )
            messages.append(message)

    write_manifest(output_path, records())
    return total_windows, messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a window segmentation and seizure-label manifest."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--annotation-source",
        type=Path,
        default=DEFAULT_ANNOTATION_SOURCE,
        help="Validated annotation audit CSV (default: results/chb01_annotation_audit.csv).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient", default="chb01")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRACTION)
    parser.add_argument("--label-policy", choices=POLICIES, default=DEFAULT_POLICY)
    args = parser.parse_args()

    configuration = WindowConfiguration(
        window_seconds=args.window_seconds,
        overlap_fraction=args.overlap,
        label_policy=args.label_policy,
    )
    patients = [args.patient]
    if args.patient.casefold() == "all":
        discovered = set()
        for path in args.data_dir.expanduser().resolve().rglob("*-summary.txt"):
            patient = path.name[: -len("-summary.txt")]
            if re.fullmatch(r"chb\d+", patient, re.IGNORECASE):
                discovered.add(patient)
        if not discovered:
            for path in args.data_dir.expanduser().resolve().rglob("*.edf"):
                match = re.match(r"^(chb\d+)_", path.name, re.IGNORECASE)
                if match:
                    discovered.add(match.group(1))
                elif re.fullmatch(r"chb\d+", path.parent.name, re.IGNORECASE):
                    discovered.add(path.parent.name)
        patients = sorted(discovered, key=str.casefold)
        if not patients:
            print("Segmentation failed: no CHB-MIT patient recordings found.", file=sys.stderr)
            return 1
    try:
        total_windows = 0
        for patient in patients:
            output_path = (
                args.output_dir.expanduser().resolve()
                / f"{patient}_window_manifest.csv"
            )
            window_count, messages = create_manifest(
                data_dir=args.data_dir.expanduser().resolve(),
                annotation_source=args.annotation_source.expanduser().resolve(),
                output_path=output_path,
                patient_id=patient,
                configuration=configuration,
            )
            for message in messages:
                print(message)
            print(f"{patient} total windows: {window_count}")
            print(f"Manifest: {output_path}")
            total_windows += window_count
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Segmentation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Total windows: {total_windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
