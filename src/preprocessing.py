#!/usr/bin/env python3
"""Reproducible continuous-recording EEG filtering, diagnostics, and windows."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
BoundaryPolicy = Literal["any_overlap", "majority_overlap", "onset"]
DEFAULT_MAX_FILTER_MEMORY_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class FilterConfig:
    """Band-pass and optional line-noise notch settings for continuous EEG."""

    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freq: float | None = None
    method: str = "fir"
    phase: str = "zero"
    fir_design: str = "firwin"

    def validate(self, sfreq: float) -> None:
        nyquist = sfreq / 2.0
        if sfreq <= 0:
            raise ValueError("Sampling frequency must be positive.")
        if not (0 < self.l_freq < self.h_freq < nyquist):
            raise ValueError(
                f"Require 0 < l_freq < h_freq < Nyquist ({nyquist:g} Hz); "
                f"received {self.l_freq:g}-{self.h_freq:g} Hz."
            )
        if self.notch_freq is not None and not (0 < self.notch_freq < nyquist):
            raise ValueError(
                f"Notch frequency must be below Nyquist ({nyquist:g} Hz)."
            )

    def description(self) -> str:
        notch = f"{self.notch_freq:g} Hz" if self.notch_freq is not None else "disabled"
        return (
            f"band-pass={self.l_freq:g}-{self.h_freq:g} Hz; notch={notch}; "
            f"method={self.method}; phase={self.phase}; fir_design={self.fir_design}"
        )


@dataclass(frozen=True)
class SeizureInterval:
    """Recording-relative expert annotation interval in seconds."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class WindowMetadata:
    patient_id: str
    recording_filename: str
    start_sample: int
    end_sample: int
    start_time_seconds: float
    end_time_seconds: float
    label: str
    seizure_overlap_fraction: float
    boundary_policy: BoundaryPolicy


@dataclass(frozen=True)
class EEGWindow(WindowMetadata):
    """A window backed by a view into the continuous input array."""

    signal: np.ndarray


def seconds_to_sample(time_seconds: float, sfreq: float) -> int:
    """Map a time to the nearest sample index."""
    if sfreq <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if time_seconds < 0:
        raise ValueError("Time must be non-negative.")
    return int(round(time_seconds * sfreq))


def calculate_window_samples(
    sfreq: float, window_seconds: float, overlap_fraction: float
) -> tuple[int, int, int]:
    """Return (window, overlap, step) sample counts using nearest-sample rounding."""
    if sfreq <= 0 or window_seconds <= 0:
        raise ValueError("Sampling frequency and window duration must be positive.")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be at least 0 and less than 1.")
    window_samples = int(round(sfreq * window_seconds))
    if window_samples < 1:
        raise ValueError("Window duration rounds to fewer than one sample.")
    overlap_samples = int(round(window_samples * overlap_fraction))
    step_samples = window_samples - overlap_samples
    if step_samples < 1:
        raise ValueError("Window overlap leaves no forward progress.")
    return window_samples, overlap_samples, step_samples


def preprocess_raw(
    raw: Any,
    config: FilterConfig = FilterConfig(),
    *,
    max_memory_bytes: int = DEFAULT_MAX_FILTER_MEMORY_BYTES,
) -> Any:
    """Filter EEG channels on a copy of the complete continuous MNE Raw object.

    The input is never modified. The returned object is loaded and filtered
    continuously; do not call this independently on short EEG windows.
    """
    sfreq = float(raw.info["sfreq"])
    config.validate(sfreq)
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive.")
    estimated_peak_bytes = int(raw.info["nchan"]) * int(raw.n_times) * 8 * 3
    if estimated_peak_bytes > max_memory_bytes:
        raise MemoryError(
            "Full-recording filtering is estimated to require up to "
            f"{estimated_peak_bytes / 1024**3:.2f} GiB "
            "(three float64 signal copies), exceeding the configured "
            f"{max_memory_bytes / 1024**3:.2f} GiB cap. Use a smaller recording "
            "or explicitly raise --max-filter-memory-gib."
        )
    eeg_picks = [
        index
        for index, channel_type in enumerate(raw.get_channel_types())
        if channel_type == "eeg"
    ]
    if not eeg_picks:
        raise ValueError("The recording contains no channels identified as EEG.")

    filtered = raw.copy().load_data()
    filtered.filter(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        picks=eeg_picks,
        method=config.method,
        phase=config.phase,
        fir_design=config.fir_design,
        verbose="ERROR",
    )
    if config.notch_freq is not None:
        filtered.notch_filter(
            freqs=[config.notch_freq],
            picks=eeg_picks,
            method=config.method,
            phase=config.phase,
            fir_design=config.fir_design,
            verbose="ERROR",
        )
    return filtered


def inspect_signal_quality(
    raw: Any,
    *,
    chunk_seconds: float = 30.0,
    flat_peak_to_peak_threshold_volts: float = 1e-12,
    relative_noise_robust_z_threshold: float = 6.0,
    expected_eeg_channels: int | None = 23,
) -> dict[str, Any]:
    """Summarize EEG channel quality in chunks without dropping any channels."""
    sfreq = float(raw.info["sfreq"])
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")
    channel_names = list(raw.ch_names)
    channel_types = list(raw.get_channel_types())
    eeg_picks = [
        index
        for index, channel_type in enumerate(channel_types)
        if channel_type == "eeg"
    ]
    eeg_names = [channel_names[index] for index in eeg_picks]
    n_samples = int(raw.n_times)
    chunk_samples = max(1, int(round(chunk_seconds * sfreq)))

    counts = np.zeros(len(eeg_picks), dtype=np.int64)
    nonfinite_counts = np.zeros(len(eeg_picks), dtype=np.int64)
    sums = np.zeros(len(eeg_picks), dtype=np.float64)
    sums_of_squares = np.zeros(len(eeg_picks), dtype=np.float64)
    minima = np.full(len(eeg_picks), np.inf)
    maxima = np.full(len(eeg_picks), -np.inf)

    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)
        chunk = raw.get_data(
            picks=eeg_picks,
            start=start,
            stop=stop,
            reject_by_annotation=None,
        )
        finite_mask = np.isfinite(chunk)
        nonfinite_counts += np.count_nonzero(~finite_mask, axis=1)
        finite_values = np.where(finite_mask, chunk, 0.0)
        counts += np.count_nonzero(finite_mask, axis=1)
        sums += np.sum(finite_values, axis=1)
        sums_of_squares += np.sum(finite_values * finite_values, axis=1)
        chunk_minima = np.min(np.where(finite_mask, chunk, np.inf), axis=1)
        chunk_maxima = np.max(np.where(finite_mask, chunk, -np.inf), axis=1)
        minima = np.minimum(minima, chunk_minima)
        maxima = np.maximum(maxima, chunk_maxima)

    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    variances = np.divide(
        sums_of_squares,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    ) - means**2
    standard_deviations = np.sqrt(np.maximum(variances, 0.0))
    rms_values = np.sqrt(
        np.divide(
            sums_of_squares,
            counts,
            out=np.full_like(sums, np.nan),
            where=counts > 0,
        )
    )

    valid_std = standard_deviations[np.isfinite(standard_deviations) & (standard_deviations > 0)]
    relative_z_by_name: dict[str, float] = {}
    if valid_std.size:
        log_std = np.log(valid_std)
        median = float(np.median(log_std))
        mad = float(np.median(np.abs(log_std - median)))
        if mad > 0:
            all_log_std = np.full_like(standard_deviations, np.nan)
            valid = np.isfinite(standard_deviations) & (standard_deviations > 0)
            all_log_std[valid] = np.log(standard_deviations[valid])
            for index in np.flatnonzero(valid):
                relative_z_by_name[eeg_names[index]] = (
                    0.6745 * (all_log_std[index] - median) / mad
                )

    channel_results = []
    for index, name in enumerate(eeg_names):
        has_finite = counts[index] > 0
        peak_to_peak = (
            float(maxima[index] - minima[index]) if has_finite else float("nan")
        )
        relative_z = relative_z_by_name.get(name)
        flags = []
        if not has_finite:
            flags.append("no_finite_samples")
        if nonfinite_counts[index] > 0:
            flags.append("non_finite_samples")
        if has_finite and peak_to_peak <= flat_peak_to_peak_threshold_volts:
            flags.append("flat_channel")
        if (
            relative_z is not None
            and relative_z > relative_noise_robust_z_threshold
        ):
            flags.append("relative_high_variability")
        channel_results.append(
            {
                "name": name,
                "type": channel_types[eeg_picks[index]],
                "finite_sample_count": int(counts[index]),
                "non_finite_sample_count": int(nonfinite_counts[index]),
                "minimum_volts": float(minima[index]) if has_finite else None,
                "maximum_volts": float(maxima[index]) if has_finite else None,
                "peak_to_peak_volts": peak_to_peak if has_finite else None,
                "mean_volts": float(means[index]) if has_finite else None,
                "standard_deviation_volts": (
                    float(standard_deviations[index]) if has_finite else None
                ),
                "rms_volts": float(rms_values[index]) if has_finite else None,
                "relative_variability_robust_z": relative_z,
                "flags": flags,
            }
        )

    eeg_name_bases: dict[str, list[str]] = {}
    for name in eeg_names:
        base_name = re.sub(r"-\d+$", "", name)
        eeg_name_bases.setdefault(base_name, []).append(name)
    duplicate_label_groups = [
        names for names in eeg_name_bases.values() if len(names) > 1
    ]
    issues = []
    if expected_eeg_channels is not None and len(eeg_picks) != expected_eeg_channels:
        issues.append(
            f"Expected {expected_eeg_channels} EEG channels; found {len(eeg_picks)}."
        )
    if duplicate_label_groups:
        issues.append(
            "MNE-disambiguated duplicate labels retained: "
            + ", ".join("/".join(group) for group in duplicate_label_groups)
        )
    if any(item["flags"] for item in channel_results):
        issues.append("One or more channel quality flags require review; no channels were removed.")

    return {
        "sampling_frequency_hz": sfreq,
        "duration_seconds": n_samples / sfreq,
        "total_channels": len(channel_names),
        "eeg_channel_count": len(eeg_picks),
        "channel_names": channel_names,
        "channel_types": channel_types,
        "duplicate_label_groups": duplicate_label_groups,
        "issues": issues,
        "channels": channel_results,
    }


def _normalize_intervals(
    intervals: Sequence[SeizureInterval | tuple[float, float]],
) -> list[SeizureInterval]:
    normalized = []
    for interval in intervals:
        if isinstance(interval, SeizureInterval):
            item = interval
        else:
            if len(interval) != 2:
                raise ValueError("Each seizure interval must contain a start and end time.")
            item = SeizureInterval(float(interval[0]), float(interval[1]))
        if item.start_seconds < 0:
            raise ValueError("Seizure start times must be non-negative.")
        if item.end_seconds < item.start_seconds:
            raise ValueError("Seizure end time cannot be earlier than its start time.")
        normalized.append(item)
    return sorted(normalized, key=lambda item: (item.start_seconds, item.end_seconds))


def _iter_window_metadata(
    n_samples: int,
    sfreq: float,
    *,
    recording_filename: str,
    patient_id: str,
    seizure_intervals: Sequence[SeizureInterval | tuple[float, float]] = (),
    window_seconds: float = 5.0,
    overlap_fraction: float = 0.5,
    boundary_policy: BoundaryPolicy = "any_overlap",
    majority_overlap_threshold: float = 0.5,
) -> Iterator[WindowMetadata]:
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if boundary_policy not in ("any_overlap", "majority_overlap", "onset"):
        raise ValueError(f"Unknown boundary policy: {boundary_policy}")
    if not 0 < majority_overlap_threshold <= 1:
        raise ValueError("majority_overlap_threshold must be in (0, 1].")
    window_samples, _, step_samples = calculate_window_samples(
        sfreq, window_seconds, overlap_fraction
    )
    intervals = _normalize_intervals(seizure_intervals)
    sample_intervals = [
        (
            seconds_to_sample(interval.start_seconds, sfreq),
            seconds_to_sample(interval.end_seconds, sfreq),
        )
        for interval in intervals
    ]
    if any(end > n_samples for _, end in sample_intervals):
        raise ValueError("A seizure interval extends beyond the provided recording.")

    start_sample = 0
    while start_sample + window_samples <= n_samples:
        end_sample = start_sample + window_samples
        overlap_samples = sum(
            max(0, min(end_sample, seizure_end) - max(start_sample, seizure_start))
            for seizure_start, seizure_end in sample_intervals
        )
        overlap_fraction = min(1.0, overlap_samples / window_samples)
        if boundary_policy == "any_overlap":
            is_seizure = overlap_samples > 0
        elif boundary_policy == "majority_overlap":
            is_seizure = overlap_fraction >= majority_overlap_threshold
        else:
            is_seizure = any(
                start_sample <= seizure_start < end_sample
                for seizure_start, _ in sample_intervals
            )
        yield WindowMetadata(
            patient_id=patient_id,
            recording_filename=recording_filename,
            start_sample=start_sample,
            end_sample=end_sample,
            start_time_seconds=start_sample / sfreq,
            end_time_seconds=end_sample / sfreq,
            label="seizure" if is_seizure else "non-seizure",
            seizure_overlap_fraction=overlap_fraction,
            boundary_policy=boundary_policy,
        )
        start_sample += step_samples


def iter_window_metadata(
    n_samples: int,
    sfreq: float,
    *,
    recording_filename: str,
    patient_id: str,
    seizure_intervals: Sequence[SeizureInterval | tuple[float, float]] = (),
    window_seconds: float = 5.0,
    overlap_fraction: float = 0.5,
    boundary_policy: BoundaryPolicy = "any_overlap",
    majority_overlap_threshold: float = 0.5,
) -> Iterator[WindowMetadata]:
    """Yield metadata for full windows only; a trailing partial window is omitted."""
    yield from _iter_window_metadata(
        n_samples,
        sfreq,
        recording_filename=recording_filename,
        patient_id=patient_id,
        seizure_intervals=seizure_intervals,
        window_seconds=window_seconds,
        overlap_fraction=overlap_fraction,
        boundary_policy=boundary_policy,
        majority_overlap_threshold=majority_overlap_threshold,
    )


def segment_signal(
    signal: np.ndarray,
    sfreq: float,
    *,
    recording_filename: str,
    patient_id: str,
    seizure_intervals: Sequence[SeizureInterval | tuple[float, float]] = (),
    window_seconds: float = 5.0,
    overlap_fraction: float = 0.5,
    boundary_policy: BoundaryPolicy = "any_overlap",
    majority_overlap_threshold: float = 0.5,
) -> Iterator[EEGWindow]:
    """Yield channel-by-sample window views with recording-relative labels."""
    array = np.asarray(signal)
    if array.ndim != 2:
        raise ValueError("signal must have shape (channels, samples).")
    for metadata in _iter_window_metadata(
        array.shape[1],
        sfreq,
        recording_filename=recording_filename,
        patient_id=patient_id,
        seizure_intervals=seizure_intervals,
        window_seconds=window_seconds,
        overlap_fraction=overlap_fraction,
        boundary_policy=boundary_policy,
        majority_overlap_threshold=majority_overlap_threshold,
    ):
        yield EEGWindow(
            **asdict(metadata),
            signal=array[:, metadata.start_sample : metadata.end_sample],
        )


def _parse_summary(summary_path: Path) -> tuple[dict[str, list[SeizureInterval]], float | None]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    rate_match = re.search(
        r"Data Sampling Rate:\s*([0-9]+(?:\.[0-9]+)?)\s*Hz",
        text,
        re.IGNORECASE,
    )
    sfreq = float(rate_match.group(1)) if rate_match else None
    block_pattern = re.compile(
        r"File Name:\s*(?P<filename>[^\r\n]+)(?P<body>.*?)(?=File Name:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    intervals_by_recording: dict[str, list[SeizureInterval]] = {}
    for match in block_pattern.finditer(text):
        body = match.group("body")
        starts = re.findall(
            r"Seizure Start Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
            body,
            re.IGNORECASE,
        )
        ends = re.findall(
            r"Seizure End Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
            body,
            re.IGNORECASE,
        )
        if len(starts) != len(ends):
            raise ValueError(
                f"Annotation start/end count differs for {match.group('filename').strip()}."
            )
        if starts:
            intervals_by_recording[match.group("filename").strip()] = [
                SeizureInterval(float(start), float(end))
                for start, end in zip(starts, ends)
            ]
    return intervals_by_recording, sfreq


def _find_summary(data_dir: Path, patient_id: str) -> Path:
    candidates = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.name.lower() == f"{patient_id}-summary.txt".lower()
    )
    if not candidates:
        raise FileNotFoundError(f"No {patient_id}-summary.txt found under {data_dir}.")
    return candidates[0]


def _format_volts(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 1e6:.3f}"


def _write_quality_report(
    path: Path,
    patient_id: str,
    config: FilterConfig,
    reports: list[tuple[str, dict[str, Any]]],
    *,
    window_seconds: float,
    overlap_fraction: float,
    boundary_policy: BoundaryPolicy,
    majority_overlap_threshold: float,
) -> None:
    lines = [
        f"{patient_id} EEG signal characteristics and preprocessing",
        "=" * (len(patient_id) + len(" EEG signal characteristics and preprocessing")),
        "",
        "Filtering is applied to each complete continuous recording before plots/windows.",
        "The original EDF files are read-only; filtered data are not saved as EDF.",
        "Filtering is exploratory and is not assumed to improve seizure detection.",
        f"Filter parameters: {config.description()}",
        "The default 60 Hz notch is disabled because 60 Hz lies above the 40 Hz",
        "band-pass upper cutoff; enable it explicitly when investigating a different",
        "passband or raw line-noise content.",
        "Band-pass filtering can attenuate slow components below 0.5 Hz and faster",
        "activity above 40 Hz, and zero-phase filtering can cause edge/ringing effects.",
        "No channels are silently removed; flat/noisy/non-finite flags require review.",
        "",
        f"Window parameters: {window_seconds:g} seconds; {overlap_fraction:.0%} overlap.",
        f"Label policy: {boundary_policy}.",
        (
            f"majority_overlap threshold: {majority_overlap_threshold:.0%} of window samples."
            if boundary_policy == "majority_overlap"
            else ""
        ),
        (
            "Windows are half-open sample intervals [start, end). Under any_overlap, "
            "any seizure sample overlapping a window marks it seizure-positive. "
            "A window ending exactly at onset does not overlap; a window starting "
            "exactly at offset does not overlap. majority_overlap requires the "
            "configured fraction. onset labels a window only when onset falls within "
            "it. Only complete windows are emitted; trailing partial samples are omitted."
        ),
        "Times in the window CSV are recording-relative.",
        "",
    ]
    for filename, report in reports:
        lines.extend(
            [
                f"{filename}",
                "-" * len(filename),
                f"Sampling frequency: {report['sampling_frequency_hz']:g} Hz",
                f"Duration: {report['duration_seconds']:.6f} seconds",
                f"Channels: {report['total_channels']} total; {report['eeg_channel_count']} EEG",
                "Channel names and types: "
                + ", ".join(
                    f"{name} ({channel_type})"
                    for name, channel_type in zip(
                        report["channel_names"], report["channel_types"]
                    )
                ),
            ]
        )
        for issue in report["issues"]:
            lines.append(f"Review: {issue}")
        lines.extend(
            [
                "",
                "Per-channel statistics (amplitude values in microvolts):",
                "channel | type | min | max | peak-to-peak | mean | std | RMS | non-finite samples | flags",
            ]
        )
        by_name = {item["name"]: item for item in report["channels"]}
        for name, channel_type in zip(report["channel_names"], report["channel_types"]):
            item = by_name.get(name)
            if item is None:
                lines.append(f"{name} | {channel_type} | not summarized (non-EEG channel)")
                continue
            lines.append(
                f"{name} | {channel_type} | "
                f"{_format_volts(item['minimum_volts'])} | "
                f"{_format_volts(item['maximum_volts'])} | "
                f"{_format_volts(item['peak_to_peak_volts'])} | "
                f"{_format_volts(item['mean_volts'])} | "
                f"{_format_volts(item['standard_deviation_volts'])} | "
                f"{_format_volts(item['rms_volts'])} | "
                f"{item['non_finite_sample_count']} | "
                f"{', '.join(item['flags']) if item['flags'] else 'none'}"
            )
        lines.append("")
    path.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


def _plot_recording_comparison(
    raw: Any,
    filtered: Any,
    *,
    filename: str,
    intervals: Sequence[SeizureInterval],
    config: FilterConfig,
    output_dir: Path,
    n_channels: int = 5,
    context_seconds: float = 30.0,
) -> list[Path]:
    import matplotlib.pyplot as plt

    sfreq = float(raw.info["sfreq"])
    duration = int(raw.n_times) / sfreq
    eeg_names = [
        name
        for name, channel_type in zip(raw.ch_names, raw.get_channel_types())
        if channel_type == "eeg"
    ][:n_channels]
    if not eeg_names:
        raise ValueError(f"No EEG channels available for plotting {filename}.")
    phases: list[tuple[str, float, float, SeizureInterval]] = []
    for interval in intervals:
        if interval.start_seconds < 0 or interval.end_seconds <= interval.start_seconds:
            raise ValueError(f"Invalid seizure interval for {filename}: {interval}")
        if interval.end_seconds > duration:
            raise ValueError(
                f"Seizure interval {interval.start_seconds:g}-{interval.end_seconds:g} s "
                f"exceeds {filename} duration ({duration:g} s)."
            )
        phases.extend(
            [
                (
                    "Before seizure",
                    max(0.0, interval.start_seconds - context_seconds),
                    interval.start_seconds,
                    interval,
                ),
                (
                    "Seizure interval",
                    interval.start_seconds,
                    interval.end_seconds,
                    interval,
                ),
                (
                    "After seizure",
                    interval.end_seconds,
                    min(duration, interval.end_seconds + context_seconds),
                    interval,
                ),
            ]
        )

    plots = []
    for kind, source in (("raw", raw), ("filtered", filtered)):
        fig, axes = plt.subplots(
            len(eeg_names),
            len(phases),
            figsize=(6.0 * len(phases), max(8.0, 1.8 * len(eeg_names))),
            squeeze=False,
            sharex="col",
        )
        for column, (phase_name, start_seconds, end_seconds, interval) in enumerate(phases):
            start_sample = seconds_to_sample(start_seconds, sfreq)
            end_sample = min(int(source.n_times), seconds_to_sample(end_seconds, sfreq))
            if end_sample <= start_sample:
                raise ValueError(f"Empty {phase_name.lower()} segment for {filename}.")
            values = source.get_data(
                picks=eeg_names,
                start=start_sample,
                stop=end_sample,
                reject_by_annotation=None,
            ) * 1e6
            times = start_sample / sfreq + np.arange(values.shape[1]) / sfreq
            for row, name in enumerate(eeg_names):
                axis = axes[row, column]
                axis.plot(times, values[row], color="black", linewidth=0.55)
                axis.axvline(
                    interval.start_seconds,
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    label="Expert onset",
                )
                axis.axvline(
                    interval.end_seconds,
                    color="blue",
                    linestyle="--",
                    linewidth=1.0,
                    label="Expert offset",
                )
                seizure_left = max(start_seconds, interval.start_seconds)
                seizure_right = min(end_seconds, interval.end_seconds)
                if seizure_right > seizure_left:
                    axis.axvspan(seizure_left, seizure_right, color="orange", alpha=0.16)
                if column == 0:
                    axis.set_ylabel(f"{name}\n(µV)", rotation=0, ha="right", va="center")
                if row == 0:
                    axis.set_title(
                        f"{phase_name}\n{start_seconds:.2f}–{end_seconds:.2f} s"
                    )
                axis.grid(True, alpha=0.2)
                if row == 0 and column == 0:
                    axis.legend(loc="upper right", fontsize="small")
                axis.set_xlim(start_seconds, end_seconds)
        for axis in axes[-1, :]:
            axis.set_xlabel("Recording-relative time (seconds)")
        figure_title = (
            f"{kind.capitalize()} EEG — {filename}\n"
            "Expert-provided seizure annotations (not model-generated)"
        )
        if kind == "filtered":
            figure_title += f"\n{config.description()}"
        fig.suptitle(figure_title)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        output_path = output_dir / f"{Path(filename).stem}_{kind}_before_during_after.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(output_path)
    return plots


def _write_window_csv(path: Path, windows: Iterable[WindowMetadata]) -> None:
    fields = [
        "patient_id",
        "recording_filename",
        "start_sample",
        "end_sample",
        "start_time_seconds",
        "end_time_seconds",
        "label",
        "seizure_overlap_fraction",
        "boundary_policy",
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
            for window in windows:
                row = asdict(window)
                row["start_time_seconds"] = f"{window.start_time_seconds:.9f}"
                row["end_time_seconds"] = f"{window.end_time_seconds:.9f}"
                row["seizure_overlap_fraction"] = f"{window.seizure_overlap_fraction:.6f}"
                writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_preprocessing(
    *,
    data_dir: Path,
    output_dir: Path,
    patient_id: str,
    config: FilterConfig,
    window_seconds: float,
    overlap_fraction: float,
    boundary_policy: BoundaryPolicy,
    majority_overlap_threshold: float,
    max_filter_memory_bytes: int = DEFAULT_MAX_FILTER_MEMORY_BYTES,
) -> tuple[list[Path], Path, Path, Path]:
    import mne

    summary_path = _find_summary(data_dir, patient_id)
    seizure_intervals, summary_sfreq = _parse_summary(summary_path)
    if not seizure_intervals:
        raise ValueError(f"No seizure intervals found in {summary_path}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_reports: list[tuple[str, dict[str, Any]]] = []
    all_plots: list[Path] = []
    missing_recordings = []
    available_recordings: list[tuple[str, list[SeizureInterval], Path]] = []
    for filename, intervals in sorted(seizure_intervals.items()):
        edf_matches = sorted(data_dir.rglob(filename))
        if not edf_matches:
            missing_recordings.append(filename)
            continue
        available_recordings.append((filename, intervals, edf_matches[0]))

    if not available_recordings:
        raise FileNotFoundError("No seizure-containing EDF recordings are available locally.")

    def window_records() -> Iterator[WindowMetadata]:
        for filename, intervals, edf_path in available_recordings:
            print(f"Processing {filename} (one recording at a time)")
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
            filtered = None
            try:
                if summary_sfreq is not None and float(raw.info["sfreq"]) != summary_sfreq:
                    raise ValueError(
                        f"{filename} sampling rate {raw.info['sfreq']:g} Hz does not match "
                        f"summary rate {summary_sfreq:g} Hz."
                    )
                quality = inspect_signal_quality(raw)
                for interval in intervals:
                    if interval.end_seconds > quality["duration_seconds"]:
                        raise ValueError(
                            f"{filename} seizure interval {interval} exceeds recording duration."
                        )
                quality_reports.append((filename, quality))

                filtered = preprocess_raw(
                    raw, config, max_memory_bytes=max_filter_memory_bytes
                )
                all_plots.extend(
                    _plot_recording_comparison(
                        raw,
                        filtered,
                        filename=filename,
                        intervals=intervals,
                        config=config,
                        output_dir=output_dir,
                    )
                )
                yield from iter_window_metadata(
                    int(raw.n_times),
                    float(raw.info["sfreq"]),
                    recording_filename=filename,
                    patient_id=patient_id,
                    seizure_intervals=intervals,
                    window_seconds=window_seconds,
                    overlap_fraction=overlap_fraction,
                    boundary_policy=boundary_policy,
                    majority_overlap_threshold=majority_overlap_threshold,
                )
            finally:
                del filtered, raw
                gc.collect()

    if missing_recordings:
        print(
            "Seizure recordings not downloaded; skipped without fabricating data: "
            + ", ".join(missing_recordings)
        )

    windows_path = output_dir / f"{patient_id}_window_metadata.csv"
    _write_window_csv(windows_path, window_records())
    if not quality_reports:
        raise FileNotFoundError("No seizure-containing EDF recordings are available locally.")

    window_samples, overlap_samples, step_samples = calculate_window_samples(
        summary_sfreq or quality_reports[0][1]["sampling_frequency_hz"],
        window_seconds,
        overlap_fraction,
    )
    metadata = {
        "patient_id": patient_id,
        "filter": asdict(config),
        "window_seconds": window_seconds,
        "overlap_fraction": overlap_fraction,
        "window_samples_at_summary_rate": window_samples,
        "overlap_samples_at_summary_rate": overlap_samples,
        "step_samples_at_summary_rate": step_samples,
        "boundary_policy": boundary_policy,
        "majority_overlap_threshold": majority_overlap_threshold,
        "source_summary": str(summary_path),
        "filtered_data_saved": False,
        "plots": [str(path) for path in all_plots],
    }
    metadata_path = output_dir / f"{patient_id}_preprocessing_parameters.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    quality_path = output_dir / f"{patient_id}_signal_quality.txt"
    _write_quality_report(
        quality_path,
        patient_id,
        config,
        quality_reports,
        window_seconds=window_seconds,
        overlap_fraction=overlap_fraction,
        boundary_policy=boundary_policy,
        majority_overlap_threshold=majority_overlap_threshold,
    )
    return all_plots, quality_path, windows_path, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect, filter, visualize, and segment seizure-containing CHB-MIT EEG."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient", default="chb01")
    parser.add_argument("--l-freq", type=float, default=0.5)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument(
        "--notch-freq",
        type=float,
        default=None,
        help="Optional notch frequency in Hz; defaults to disabled.",
    )
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--boundary-policy",
        choices=("any_overlap", "majority_overlap", "onset"),
        default="any_overlap",
    )
    parser.add_argument("--majority-overlap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-filter-memory-gib",
        type=float,
        default=DEFAULT_MAX_FILTER_MEMORY_BYTES / 1024**3,
        help="Reject full-recording filtering if its conservative peak estimate exceeds this cap.",
    )
    args = parser.parse_args()
    config = FilterConfig(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freq=args.notch_freq,
    )
    try:
        plots, quality_path, windows_path, metadata_path = run_preprocessing(
            data_dir=args.data_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            patient_id=args.patient,
            config=config,
            window_seconds=args.window_seconds,
            overlap_fraction=args.overlap,
            boundary_policy=args.boundary_policy,
            majority_overlap_threshold=args.majority_overlap_threshold,
            max_filter_memory_bytes=int(args.max_filter_memory_gib * 1024**3),
        )
    except (FileNotFoundError, ImportError, MemoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Signal quality report: {quality_path}")
    print(f"Window metadata: {windows_path}")
    print(f"Experiment parameters: {metadata_path}")
    print("Plots:")
    for path in plots:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
