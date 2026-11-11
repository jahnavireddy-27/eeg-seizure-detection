#!/usr/bin/env python3
"""Extract reproducible per-channel features from segmented EEG CSV windows."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import pywt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "window_manifests"
    / "chb01_window_manifest.csv"
)
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "processed" / "csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features"
DEFAULT_CHUNK_ROWS = 16_384
FEATURE_FAMILIES = ("time", "frequency", "wavelet")
DEFAULT_BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 80.0),
)
COEFFICIENT_SETS = ("A3", "D3", "D2", "D1")
METADATA_COLUMNS = (
    "patient_id",
    "recording_id",
    "relative_path",
    "recording_filename",
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "seizure_label",
    "annotation_status",
    "label_policy",
    "sampling_frequency_hz",
    "window_samples",
    "feature_config_json",
)
TIME_FEATURES = ("mean", "std_population", "maximum", "minimum")
SPECTRAL_FEATURES = (
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "total_spectral_power",
)


@dataclass(frozen=True)
class FeatureConfiguration:
    wavelet: str = "db4"
    wavelet_mode: str = "symmetric"
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS
    feature_families: tuple[str, ...] = FEATURE_FAMILIES

    def validate(self) -> None:
        if not self.wavelet.strip():
            raise ValueError("wavelet name cannot be empty.")
        if not self.wavelet_mode.strip():
            raise ValueError("wavelet boundary mode cannot be empty.")
        try:
            pywt.Wavelet(self.wavelet)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown wavelet family {self.wavelet!r}.") from exc
        if self.wavelet_mode not in pywt.Modes.modes:
            raise ValueError(
                f"Unknown wavelet boundary mode {self.wavelet_mode!r}; "
                f"choose from {pywt.Modes.modes}."
            )
        if not self.feature_families or any(
            family not in FEATURE_FAMILIES for family in self.feature_families
        ):
            raise ValueError(
                f"feature_families must contain values from {FEATURE_FAMILIES}."
            )
        if len(set(self.feature_families)) != len(self.feature_families):
            raise ValueError("feature_families cannot contain duplicates.")
        names: set[str] = set()
        previous_high = -math.inf
        for name, low, high in self.bands:
            if not name or not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError(f"Invalid frequency band name: {name!r}.")
            if name in names:
                raise ValueError(f"Duplicate frequency band name: {name!r}.")
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or low < 0
                or high <= low
            ):
                raise ValueError(
                    f"Invalid frequency band {name!r}: expected finite 0 <= low < high."
                )
            if low < previous_high:
                raise ValueError("Frequency bands must be ordered and non-overlapping.")
            names.add(name)
            previous_high = high

    def to_json(self) -> str:
        return json.dumps(
            {
                "bands_hz_half_open": [
                    {"name": name, "low": low, "high_exclusive": high}
                    for name, low, high in self.bands
                ],
                "feature_families": list(self.feature_families),
                "fft": {
                    "transform": "rfft",
                    "detrend": "subtract_window_mean",
                    "window": "rectangular",
                    "power": "one_sided_periodogram_density_integrated_over_bins",
                    "spectral_centroid_weighting": "magnitude",
                    "dominant_frequency": "largest_nonzero_frequency_bin_magnitude",
                },
                "standard_deviation": "population_ddof_0",
                "wavelet": self.wavelet,
                "wavelet_mode": self.wavelet_mode,
                "wavelet_level": 3,
                "wavelet_coefficients": list(COEFFICIENT_SETS),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class ManifestWindow:
    patient_id: str
    recording_filename: str
    recording_id: str
    relative_path: str
    window_index: int
    start_sample: int
    end_sample_exclusive: int
    start_time_seconds: float
    end_time_seconds: float
    seizure_label: str
    annotation_status: str
    label_policy: str
    sampling_frequency_hz: float
    channel_labels: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return self.end_sample_exclusive - self.start_sample


@dataclass(frozen=True)
class RecordingManifest:
    patient_id: str
    recording_filename: str
    recording_id: str
    relative_path: str
    channel_labels: tuple[str, ...]
    window_count: int
    sample_count: int
    stride_samples: int
    sampling_frequency_hz: float
    label_policy: str

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.patient_id, self.recording_id, self.relative_path)


def _manifest_windows(path: Path) -> Iterator[ManifestWindow]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_fields = {
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
            "sampling_frequency_hz",
            "window_samples",
            "stride_samples",
            "channel_count",
            "channel_labels",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"Window manifest is missing fields: {', '.join(sorted(missing_fields))}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                labels_value = json.loads(row["channel_labels"])
                if (
                    not isinstance(labels_value, list)
                    or not labels_value
                    or not all(isinstance(label, str) and label for label in labels_value)
                ):
                    raise ValueError("channel_labels must be a non-empty JSON string list")
                labels = tuple(labels_value)
                if int(row["channel_count"]) != len(labels):
                    raise ValueError("channel_count does not match channel_labels")
                filename = row["recording_filename"]
                relative_path = row.get("relative_path") or filename
                relative = Path(relative_path)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("relative_path must remain inside the dataset root")
                if relative.name != filename:
                    raise ValueError("relative_path basename does not match recording_filename")
                window = ManifestWindow(
                    patient_id=row["patient_id"],
                    recording_filename=filename,
                    recording_id=row["recording_id"],
                    relative_path=relative.as_posix(),
                    window_index=int(row["window_index"]),
                    start_sample=int(row["start_sample"]),
                    end_sample_exclusive=int(row["end_sample_exclusive"]),
                    start_time_seconds=float(row["start_time_seconds"]),
                    end_time_seconds=float(row["end_time_seconds"]),
                    seizure_label=row["seizure_label"],
                    annotation_status=row.get("annotation_status") or "unknown",
                    label_policy=row["label_policy"],
                    sampling_frequency_hz=float(row["sampling_frequency_hz"]),
                    channel_labels=labels,
                )
                window_samples = int(row["window_samples"])
                stride_samples = int(row["stride_samples"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid manifest row {line_number}: {exc}") from exc
            if (
                not window.patient_id
                or not window.recording_filename
                or window.recording_id != Path(window.recording_filename).stem
                or not window.recording_id
            ):
                raise ValueError(f"Invalid recording identity in manifest row {line_number}.")
            if (
                window.start_sample < 0
                or window.end_sample_exclusive <= window.start_sample
                or window_samples != window.sample_count
                or stride_samples <= 0
            ):
                raise ValueError(f"Invalid sample bounds in manifest row {line_number}.")
            if (
                not math.isfinite(window.sampling_frequency_hz)
                or window.sampling_frequency_hz <= 0
                or not math.isclose(
                    window.start_time_seconds,
                    window.start_sample / window.sampling_frequency_hz,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                or not math.isclose(
                    window.end_time_seconds,
                    window.end_sample_exclusive / window.sampling_frequency_hz,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError(f"Invalid sample/time mapping in manifest row {line_number}.")
            if window.seizure_label not in {"seizure", "non_seizure", "unknown"}:
                raise ValueError(f"Invalid seizure label in manifest row {line_number}.")
            if not window.label_policy:
                raise ValueError(f"Missing label policy in manifest row {line_number}.")
            yield window


def _validate_manifest(path: Path) -> dict[tuple[str, str, str], RecordingManifest]:
    recordings: dict[tuple[str, str, str], RecordingManifest] = {}
    completed: set[tuple[str, str, str]] = set()
    current_identity: tuple[str, str, str] | None = None
    current_rows = 0
    previous: ManifestWindow | None = None
    info: RecordingManifest | None = None
    for window in _manifest_windows(path):
        identity = (window.patient_id, window.recording_id, window.relative_path)
        if identity != current_identity:
            if current_identity is not None:
                assert info is not None
                recordings[current_identity] = info
                completed.add(current_identity)
            if identity in completed:
                raise ValueError(
                    f"Manifest windows for recording identity {identity} are not contiguous."
                )
            current_identity = identity
            current_rows = 0
            previous = None
            info = RecordingManifest(
                patient_id=window.patient_id,
                recording_filename=window.recording_filename,
                recording_id=window.recording_id,
                relative_path=window.relative_path,
                channel_labels=window.channel_labels,
                window_count=0,
                sample_count=window.sample_count,
                stride_samples=0,
                sampling_frequency_hz=window.sampling_frequency_hz,
                label_policy=window.label_policy,
            )
        if info is None or identity != current_identity:
            raise RuntimeError("Internal manifest grouping error.")
        expected_stride = (
            window.start_sample - previous.start_sample if previous is not None else 0
        )
        if (
            window.window_index != current_rows
            or window.recording_filename != info.recording_filename
            or window.channel_labels != info.channel_labels
            or window.sampling_frequency_hz != info.sampling_frequency_hz
            or window.sample_count != info.sample_count
            or window.label_policy != info.label_policy
            or (previous is not None and expected_stride <= 0)
        ):
            raise ValueError(
                f"{window.relative_path}: inconsistent channel schema or window sequence."
            )
        if previous is not None and info.stride_samples not in {0, expected_stride}:
            raise ValueError(f"{window.relative_path}: window stride changes within recording.")
        info = RecordingManifest(
            patient_id=info.patient_id,
            recording_filename=info.recording_filename,
            recording_id=info.recording_id,
            relative_path=info.relative_path,
            channel_labels=info.channel_labels,
            window_count=current_rows + 1,
            sample_count=info.sample_count,
            stride_samples=info.stride_samples or expected_stride,
            sampling_frequency_hz=info.sampling_frequency_hz,
            label_policy=info.label_policy,
        )
        current_rows += 1
        previous = window
    if current_identity is not None and info is not None:
        recordings[current_identity] = info
    if not recordings:
        raise ValueError(f"Window manifest has no data rows: {path}")
    return recordings


def _feature_value_names(configuration: FeatureConfiguration) -> tuple[str, ...]:
    result: list[str] = []
    if "time" in configuration.feature_families:
        result.extend(TIME_FEATURES)
    if "frequency" in configuration.feature_families:
        result.extend(SPECTRAL_FEATURES)
        result.extend(f"band_power_{name}" for name, _, _ in configuration.bands)
    if "wavelet" in configuration.feature_families:
        result.extend(
            f"wavelet_{coefficient}_{statistic}"
            for coefficient in COEFFICIENT_SETS
            for statistic in ("mean", "std_population", "energy")
        )
    return tuple(result)


def feature_column_names(
    channel_labels: Sequence[str],
    configuration: FeatureConfiguration = FeatureConfiguration(),
) -> tuple[str, ...]:
    configuration.validate()
    columns = []
    for channel_index, label in enumerate(channel_labels):
        encoded_label = quote(label, safe="")
        columns.extend(
            f"channel_{channel_index:02d}__{encoded_label}__{feature}"
            for feature in _feature_value_names(configuration)
        )
    if len(columns) != len(set(columns)):
        raise ValueError("Generated feature column names are not unique.")
    return tuple(columns)


def _normalize_channel_label(label: str) -> str:
    """Normalize only case and surrounding/repeated whitespace."""
    return " ".join(label.strip().split()).upper()


def _channel_instances(
    labels: Sequence[str],
) -> list[tuple[str, int, str]]:
    counts: dict[str, int] = {}
    instances = []
    for label in labels:
        normalized = _normalize_channel_label(label)
        if not normalized:
            raise ValueError("Channel labels cannot be empty after normalization.")
        counts[normalized] = counts.get(normalized, 0) + 1
        instances.append((normalized, counts[normalized], label))
    return instances


def _feature_schema(
    recordings: Sequence[RecordingManifest],
    configuration: FeatureConfiguration,
) -> tuple[tuple[str, ...], dict[tuple[str, str, str], dict[str, str]], dict[str, object]]:
    if not recordings:
        raise ValueError("At least one recording is required to build a feature schema.")
    value_names = _feature_value_names(configuration)
    legacy_mode = len({item.channel_labels for item in recordings}) == 1
    mappings: dict[tuple[str, str, str], dict[str, str]] = {}
    recording_channels: dict[str, list[dict[str, object]]] = {}
    if legacy_mode:
        labels = recordings[0].channel_labels
        columns = feature_column_names(labels, configuration)
        for recording in recordings:
            mappings[recording.identity] = {column: column for column in columns}
    else:
        all_instances = {
            (normalized, occurrence)
            for recording in recordings
            for normalized, occurrence, _ in _channel_instances(recording.channel_labels)
        }
        ordered_instances = sorted(all_instances, key=lambda item: (item[0], item[1]))
        columns = tuple(
            f"channel__{quote(normalized, safe='')}__instance_{occurrence:02d}__{name}"
            for normalized, occurrence in ordered_instances
            for name in value_names
        )
        for recording in recordings:
            source_to_target: dict[str, str] = {}
            for index, (normalized, occurrence, original) in enumerate(
                _channel_instances(recording.channel_labels)
            ):
                canonical_id = (
                    f"channel__{quote(normalized, safe='')}__instance_{occurrence:02d}"
                )
                encoded_original = quote(original, safe="")
                for name in value_names:
                    source_column = (
                        f"channel_{index:02d}__{encoded_original}__{name}"
                    )
                    source_to_target[source_column] = f"{canonical_id}__{name}"
            mappings[recording.identity] = source_to_target

    channel_labels: dict[str, set[str]] = {}
    present_channels: dict[str, set[str]] = {}
    for recording in recordings:
        entries = []
        for index, (normalized, occurrence, original) in enumerate(
            _channel_instances(recording.channel_labels)
        ):
            key = f"channel__{quote(normalized, safe='')}__instance_{occurrence:02d}"
            channel_labels.setdefault(key, set()).add(original)
            present_channels.setdefault(
                _identity_json_key(recording.identity), set()
            ).add(key)
            entries.append(
                {
                    "channel_index": index,
                    "original_label": original,
                    "normalized_label": normalized,
                    "instance": occurrence,
                    "canonical_channel_id": key,
                    "feature_columns": (
                        [
                            column
                            for column in columns
                            if column.startswith(
                                f"channel_{index:02d}__{quote(original, safe='')}__"
                            )
                        ]
                        if legacy_mode
                        else [
                            f"{key}__{name}" for name in value_names
                        ]
                    ),
                }
            )
        recording_channels[_identity_json_key(recording.identity)] = entries
    schema: dict[str, object] = {
        "schema_version": 1,
        "channel_matching": {
            "normalization": "uppercase; trim and collapse whitespace only",
            "bipolar_derivations": "direction and electrode order are preserved",
            "duplicate_labels": (
                "repeated normalized labels are distinguished by occurrence order "
                "within each recording"
            ),
        },
        "feature_configuration": json.loads(configuration.to_json()),
        "schema_mode": "legacy_ordered" if legacy_mode else "channel_aligned",
        "metadata_columns": list(METADATA_COLUMNS),
        "target_column": "seizure_label",
        "annotation_status_column": "annotation_status",
        "predictor_columns": list(columns),
        "channels": [
            {
                "channel_id": key,
                "original_labels": sorted(channel_labels[key]),
                "predictor_columns": [
                    column for column in columns if column.startswith(f"{key}__")
                ]
                if not legacy_mode
                else [
                    column
                    for column in columns
                    if any(
                        entry["canonical_channel_id"] == key
                        and column in entry["feature_columns"]
                        for entries in recording_channels.values()
                        for entry in entries
                    )
                ],
            }
            for key in sorted(channel_labels)
        ],
        "recordings": recording_channels,
        "unavailable_channels": {
            identity: sorted(set(channel_labels) - available)
            for identity, available in sorted(present_channels.items())
        },
    }
    return columns, mappings, schema


def _identity_json_key(identity: tuple[str, str, str]) -> str:
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _manifest_recording_groups(
    path: Path,
) -> Iterator[tuple[tuple[str, str, str], list[ManifestWindow]]]:
    windows = _manifest_windows(path)
    for identity, rows in itertools.groupby(
        windows,
        key=lambda window: (
            window.patient_id,
            window.recording_id,
            window.relative_path,
        ),
    ):
        yield identity, list(rows)


def _resolve_signal_csv(
    csv_dir: Path,
    recording: RecordingManifest,
    basename_counts: dict[str, int],
) -> Path:
    relative = Path(recording.relative_path)
    candidates = [
        csv_dir / f"{relative.as_posix()}.csv",
    ]
    patient_relative = relative
    if patient_relative.parts and patient_relative.parts[0].casefold() == recording.patient_id.casefold():
        patient_relative = Path(*patient_relative.parts[1:])
    if patient_relative.parts:
        candidates.append(csv_dir / recording.patient_id / f"{patient_relative.as_posix()}.csv")
    if basename_counts[recording.recording_filename.casefold()] == 1:
        candidates.append(csv_dir / f"{recording.recording_filename}.csv")
    existing = list(dict.fromkeys(candidate for candidate in candidates if candidate.is_file()))
    if not existing:
        raise FileNotFoundError(
            f"No converted CSV found for {recording.patient_id}/"
            f"{recording.relative_path} under {csv_dir}."
        )
    if len(existing) > 1:
        raise ValueError(
            f"Multiple converted CSV paths match {recording.patient_id}/"
            f"{recording.relative_path}: {existing}"
        )
    return existing[0]


def _frequency_features(
    signal: np.ndarray,
    sampling_frequency_hz: float,
    bands: Sequence[tuple[str, float, float]],
) -> dict[str, float]:
    sample_count = signal.size
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency_hz)
    magnitudes = np.abs(spectrum)
    power_density = np.square(magnitudes) / (sampling_frequency_hz * sample_count)
    if sample_count % 2 == 0:
        if power_density.size > 2:
            power_density[1:-1] *= 2.0
    elif power_density.size > 1:
        power_density[1:] *= 2.0

    bin_width = sampling_frequency_hz / sample_count
    total_power = float(np.sum(power_density) * bin_width)
    positive = frequencies > 0
    if not np.any(positive) or not np.any(magnitudes[positive] > 0):
        dominant_frequency = 0.0
        centroid = 0.0
    else:
        positive_indices = np.flatnonzero(positive)
        dominant_frequency = float(
            frequencies[positive_indices[np.argmax(magnitudes[positive])]]
        )
        positive_magnitudes = magnitudes[positive]
        centroid = float(
            np.sum(frequencies[positive] * positive_magnitudes)
            / np.sum(positive_magnitudes)
        )

    result = {
        "dominant_frequency_hz": dominant_frequency,
        "spectral_centroid_hz": centroid,
        "total_spectral_power": total_power,
    }
    for name, low, high in bands:
        if high > sampling_frequency_hz / 2.0:
            raise ValueError(
                f"Band {name!r} ends at {high:g} Hz, above the Nyquist frequency "
                f"{sampling_frequency_hz / 2:g} Hz."
            )
        in_band = (frequencies >= low) & (frequencies < high)
        result[f"band_power_{name}"] = float(
            np.sum(power_density[in_band]) * bin_width
        )
    return result


def extract_channel_features(
    signal: Sequence[float] | np.ndarray,
    sampling_frequency_hz: float,
    configuration: FeatureConfiguration = FeatureConfiguration(),
) -> dict[str, float]:
    """Extract one channel's feature vector; non-finite data is rejected."""
    configuration.validate()
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Each channel window must be a non-empty 1D signal.")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive and finite.")
    if not np.isfinite(values).all():
        raise ValueError("EEG window contains missing or non-finite signal values.")

    features: dict[str, float] = {}
    if "time" in configuration.feature_families:
        features.update(
            {
                "mean": float(np.mean(values)),
                "std_population": float(np.std(values, ddof=0)),
                "maximum": float(np.max(values)),
                "minimum": float(np.min(values)),
            }
        )
    if "frequency" in configuration.feature_families:
        features.update(
            _frequency_features(
                values, sampling_frequency_hz, configuration.bands
            )
        )
    if "wavelet" in configuration.feature_families:
        coefficients = pywt.wavedec(
            values,
            wavelet=configuration.wavelet,
            mode=configuration.wavelet_mode,
            level=3,
        )
        for name, coefficient in zip(COEFFICIENT_SETS, coefficients):
            features[f"wavelet_{name}_mean"] = float(np.mean(coefficient))
            features[f"wavelet_{name}_std_population"] = float(
                np.std(coefficient, ddof=0)
            )
            features[f"wavelet_{name}_energy"] = float(
                np.sum(np.square(coefficient, dtype=np.float64))
            )
    return features


def _feature_row(
    window: ManifestWindow,
    channel_segments: np.ndarray,
    configuration: FeatureConfiguration,
    config_json: str,
    feature_column_map: dict[str, str] | None = None,
    feature_columns: Sequence[str] = (),
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "patient_id": window.patient_id,
        "recording_id": window.recording_id,
        "relative_path": window.relative_path,
        "recording_filename": window.recording_filename,
        "window_index": window.window_index,
        "start_time_seconds": f"{window.start_time_seconds:.9f}",
        "end_time_seconds": f"{window.end_time_seconds:.9f}",
        "seizure_label": window.seizure_label,
        "annotation_status": window.annotation_status,
        "label_policy": window.label_policy,
        "sampling_frequency_hz": f"{window.sampling_frequency_hz:g}",
        "window_samples": window.sample_count,
        "feature_config_json": config_json,
    }
    if feature_columns:
        row.update({name: float("nan") for name in feature_columns})
    expected_names = _feature_value_names(configuration)
    for channel_index, channel_values in enumerate(channel_segments.T):
        feature_values = extract_channel_features(
            channel_values,
            window.sampling_frequency_hz,
            configuration,
        )
        if tuple(feature_values) != expected_names:
            raise RuntimeError("Feature extraction did not produce the configured schema.")
        encoded_label = quote(window.channel_labels[channel_index], safe="")
        for name, value in feature_values.items():
            source_column = f"channel_{channel_index:02d}__{encoded_label}__{name}"
            output_column = (
                feature_column_map.get(source_column, source_column)
                if feature_column_map is not None
                else source_column
            )
            row[output_column] = value
    return row


def _iter_csv_chunks(
    csv_path: Path, column_count: int, chunk_rows: int
) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(
        csv_path,
        header=None,
        skiprows=1,
        names=list(range(column_count)),
        dtype=np.float64,
        chunksize=chunk_rows,
        encoding="utf-8",
    )
    try:
        yield from reader
    finally:
        reader.close()


def _read_recording_windows(
    csv_path: Path,
    windows: Sequence[ManifestWindow],
    configuration: FeatureConfiguration,
    chunk_rows: int,
    feature_column_map: dict[str, str] | None = None,
    feature_columns: Sequence[str] = (),
) -> Iterator[dict[str, str | int | float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Converted recording CSV does not exist: {csv_path}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")
    labels = windows[0].channel_labels
    expected_header = ["Time", *labels]
    with csv_path.open(newline="", encoding="utf-8") as file:
        header = next(csv.reader(file), None)
    if header != expected_header:
        raise ValueError(
            f"{csv_path.name}: converted CSV channel labels do not exactly match "
            "the manifest, including their order and duplicates."
        )

    window_size = windows[0].sample_count
    stride = (
        windows[1].start_sample - windows[0].start_sample
        if len(windows) > 1
        else windows[0].sample_count
    )
    if stride <= 0:
        raise ValueError(f"{csv_path.name}: invalid window stride in manifest.")
    if any(
        window.sample_count != window_size
        or window.start_sample != index * stride
        or window.end_sample_exclusive != window.start_sample + window_size
        for index, window in enumerate(windows)
    ):
        raise ValueError(
            f"{csv_path.name}: windows do not form the expected anchored sequence."
        )

    buffer = np.empty((0, len(labels)), dtype=np.float64)
    buffer_start = 0
    total_seen = 0
    window_cursor = 0
    column_count = len(labels) + 1
    for chunk in _iter_csv_chunks(csv_path, column_count, chunk_rows):
        values = chunk.to_numpy(dtype=np.float64, copy=False)
        if values.shape[1] != column_count:
            raise ValueError(
                f"{csv_path.name}: CSV row width {values.shape[1]} does not match "
                f"{column_count} expected columns."
            )
        if not np.isfinite(values).all():
            position = np.argwhere(~np.isfinite(values))[0]
            sample_index = total_seen + int(position[0])
            column_index = int(position[1])
            channel = "Time" if column_index == 0 else labels[column_index - 1]
            raise ValueError(
                f"{csv_path.name}: non-finite value at sample {sample_index}, "
                f"column {channel!r}; no imputation was applied."
            )
        signal_start = total_seen
        expected_times = (
            np.arange(signal_start, signal_start + len(values), dtype=np.float64)
            / windows[0].sampling_frequency_hz
        )
        if not np.allclose(
            values[:, 0], expected_times, rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                f"{csv_path.name}: time column does not match the manifest sampling "
                "frequency and sample indices."
            )
        incoming = values[:, 1:]
        if buffer.size == 0:
            buffer_start = total_seen
            buffer = incoming.copy()
        else:
            if buffer_start + len(buffer) != total_seen:
                raise RuntimeError("Internal sample buffer lost continuity.")
            buffer = np.concatenate((buffer, incoming), axis=0)
        total_seen += len(values)
        available_end = total_seen

        while window_cursor < len(windows):
            window = windows[window_cursor]
            if window.end_sample_exclusive > available_end:
                break
            if window.start_sample < buffer_start:
                raise RuntimeError("Window start was discarded before extraction.")
            start_offset = window.start_sample - buffer_start
            stop_offset = start_offset + window_size
            segment = buffer[start_offset:stop_offset]
            if len(segment) != window_size:
                raise RuntimeError("Window sample count differs from manifest.")
            yield _feature_row(
                window,
                segment,
                configuration,
                configuration.to_json(),
                feature_column_map,
                feature_columns,
            )
            window_cursor += 1

        if window_cursor < len(windows):
            keep_from = min(windows[window_cursor].start_sample, available_end)
        else:
            keep_from = available_end
        discard_count = keep_from - buffer_start
        if discard_count > 0:
            buffer = buffer[discard_count:]
            buffer_start = keep_from

    if window_cursor != len(windows):
        raise ValueError(
            f"{csv_path.name}: exported CSV ended before all manifest windows "
            f"were available ({window_cursor}/{len(windows)})."
        )


def _parse_bands(value: str) -> tuple[tuple[str, float, float], ...]:
    """Parse comma-separated name:low:high bands; each high edge is exclusive."""
    bands = []
    for definition in value.split(","):
        parts = definition.strip().split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Each band must use name:low:high format."
            )
        try:
            bands.append((parts[0], float(parts[1]), float(parts[2])))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid numeric band boundary in {definition!r}."
            ) from exc
    return tuple(bands)


def extract_feature_matrix(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    csv_dir: Path = DEFAULT_CSV_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    configuration: FeatureConfiguration = FeatureConfiguration(),
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[Path, int, int]:
    configuration.validate()
    recordings_by_identity = _validate_manifest(manifest_path)
    recordings = sorted(
        recordings_by_identity.values(),
        key=lambda item: (
            item.patient_id.casefold(),
            item.relative_path.casefold(),
            item.recording_id.casefold(),
        ),
    )
    patient_ids = sorted({item.patient_id for item in recordings}, key=str.casefold)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = (
        f"{patient_ids[0]}_features.csv"
        if len(patient_ids) == 1
        else "chbmit_features.csv"
    )
    output_path = output_dir / output_name
    feature_columns, feature_maps, schema = _feature_schema(recordings, configuration)
    schema["dataset"] = str(output_path)
    fieldnames = [*METADATA_COLUMNS, *feature_columns]
    config_json = configuration.to_json()
    basename_counts: dict[str, int] = {}
    for recording in recordings:
        filename_key = recording.recording_filename.casefold()
        basename_counts[filename_key] = basename_counts.get(filename_key, 0) + 1
    temporary_path: Path | None = None
    output_rows = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for identity, windows in _manifest_recording_groups(manifest_path):
                recording = recordings_by_identity.get(identity)
                if recording is None:
                    raise ValueError(f"Unvalidated recording identity in manifest: {identity}")
                if len(windows) != recording.window_count:
                    raise ValueError(
                        f"{recording.relative_path}: manifest changed after validation."
                    )
                csv_path = _resolve_signal_csv(csv_dir, recording, basename_counts)
                for row in _read_recording_windows(
                    csv_path,
                    windows,
                    configuration,
                    chunk_rows,
                    feature_maps[identity],
                    feature_columns,
                ):
                    writer.writerow(row)
                    output_rows += 1
            output_file.flush()
            os.fsync(output_file.fileno())

        expected_rows = sum(item.window_count for item in recordings)
        if output_rows != expected_rows:
            raise ValueError(
                f"Feature matrix has {output_rows} rows; manifest has "
                f"{expected_rows} windows."
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    schema_path = output_path.with_suffix(".schema.json")
    schema_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{schema_path.name}.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as schema_file:
            schema_temp = Path(schema_file.name)
            json.dump(schema, schema_file, sort_keys=True, indent=2, ensure_ascii=False)
            schema_file.write("\n")
            schema_file.flush()
            os.fsync(schema_file.fileno())
        os.replace(schema_temp, schema_path)
        schema_temp = None
    finally:
        if schema_temp is not None:
            schema_temp.unlink(missing_ok=True)
    return output_path, output_rows, len(fieldnames)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-channel features from existing EEG window manifests."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-mode", default="symmetric")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_FAMILIES,
        default=list(FEATURE_FAMILIES),
        help="Feature families to include (default: time frequency wavelet).",
    )
    parser.add_argument(
        "--bands",
        type=_parse_bands,
        default=DEFAULT_BANDS,
        help=(
            "Comma-separated name:low:high bands in Hz, with high edges exclusive "
            "(default: delta:0.5:4,theta:4:8,alpha:8:13,beta:13:30,gamma:30:80)."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = parser.parse_args()
    configuration = FeatureConfiguration(
        wavelet=args.wavelet,
        wavelet_mode=args.wavelet_mode,
        bands=args.bands,
        feature_families=tuple(args.features),
    )
    try:
        output_path, row_count, column_count = extract_feature_matrix(
            manifest_path=args.manifest.expanduser().resolve(),
            csv_dir=args.csv_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            configuration=configuration,
            chunk_rows=args.chunk_rows,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Feature extraction failed: {exc}", file=sys.stderr)
        return 1
    print(f"Feature matrix: {output_path}")
    print(f"Rows/windows: {row_count}")
    print(f"Columns: {column_count}")
    print(f"Feature families: {', '.join(configuration.feature_families)}")
    print(f"Wavelet: {configuration.wavelet}, mode={configuration.wavelet_mode}, level=3")
    print(f"Frequency bands: {configuration.to_json()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
