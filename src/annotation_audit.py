#!/usr/bin/env python3
"""Audit CHB-MIT patient-summary seizure annotations against local EDF files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from annotation_utils import (
    compare_summary_sidecar,
    parse_patient_summary as _parse_patient_summary,
    read_wfdb_sidecar as _read_wfdb_sidecar,
    validate_annotations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
def parse_patient_summary(
    summary_path: Path,
) -> tuple[list[dict[str, Any]], float | None]:
    """Parse recording-relative seizure times from a CHB-MIT summary."""
    return _parse_patient_summary(summary_path)


def _interval_text(intervals: list[dict[str, Any]]) -> str:
    if not intervals:
        return "none"
    parts = []
    for interval in intervals:
        start = interval["start_seconds"]
        end = interval["end_seconds"]
        if start is None or end is None:
            parts.append(f"incomplete ({start if start is not None else '?'} to {end if end is not None else '?'} s)")
        else:
            parts.append(
                f"{start:.15g}-{end:.15g} s "
                f"(duration {interval['duration_seconds']:.15g} s)"
            )
    return "; ".join(parts)


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def read_wfdb_sidecar(path: Path, sample_rate: float | None) -> dict[str, Any]:
    """Read an annotation sidecar using WFDB's format-aware parser."""
    return _read_wfdb_sidecar(path, sample_rate)


def _sidecar_matches_summary(
    sidecar_intervals: list[dict[str, Any]],
    summary_intervals: list[dict[str, Any]],
    sample_rate: float | None,
) -> bool:
    return (
        compare_summary_sidecar(summary_intervals, sidecar_intervals, sample_rate)
        == "agreement"
    )


def _group_summary_records(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["filename"], []).append(record)
    return grouped


def _build_validation_issues(
    filename: str,
    source_records: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    declared_counts: list[int],
    duration_seconds: float | None,
    sidecar: dict[str, Any] | None,
    sample_rate: float | None,
) -> list[str]:
    return validate_annotations(
        filename,
        source_records,
        intervals,
        declared_counts,
        duration_seconds,
        sidecar,
        sample_rate,
    )


def _find_summaries(data_dir: Path, patient: str) -> list[Path]:
    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.name.lower() == f"{patient}-summary.txt".lower()
    )
    if not paths:
        raise FileNotFoundError(f"No {patient}-summary.txt found under {data_dir}")
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "patient_id",
        "recording_id",
        "relative_path",
        "recording",
        "downloaded",
        "contains_seizure",
        "summary_seizure_count",
        "seizure_intervals",
        "edf_duration_seconds",
        "sidecar_available",
        "sidecar_annotation_samples",
        "sidecar_seizure_intervals",
        "annotation_source",
        "annotation_agreement",
        "validation_status",
        "validation_issues",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    patient: str,
    summary_paths: list[Path],
    rows: list[dict[str, Any]],
    sidecar_paths: list[Path],
    sample_rate: float | None,
    sidecar_errors: dict[str, str],
    plot_path: Path | None,
) -> None:
    seizure_rows = [row for row in rows if row["_contains_seizure"]]
    downloaded_seizures = [row["recording"] for row in seizure_rows if row["_downloaded"]]
    missing_seizures = [row["recording"] for row in seizure_rows if not row["_downloaded"]]
    failed_rows = [row for row in rows if row["_issues"]]
    missing_information = [
        row for row in rows
        if any("MISSING_ANNOTATION_INFORMATION" in issue for issue in row["_issues"])
    ]
    invalid_rows = [
        row for row in rows
        if any(
            issue.startswith(
                (
                    "NEGATIVE_START_TIME",
                    "END_BEFORE_START",
                    "INTERVAL_EXCEEDS_EDF_DURATION",
                    "DUPLICATE_ANNOTATION",
                    "CONFLICTING_ANNOTATIONS",
                    "CONFLICTING_SUMMARY_RECORDS",
                    "SIDECAR_SUMMARY_CONFLICT",
                )
            )
            for issue in row["_issues"]
        )
    ]
    matching_sidecars = sum(
        "SIDECAR_SUMMARY_CONFLICT" not in row["_issues"]
        and any(
            path.name == row["recording"] + ".seizures"
            for path in sidecar_paths
        )
        and not any("SIDECAR_READ_OR_FORMAT_ERROR" in issue for issue in row["_issues"])
        for row in rows
    )

    lines = [
        f"{patient} seizure annotation audit",
        "=" * (len(patient) + len(" seizure annotation audit")),
        "",
        "Patient summaries: " + "; ".join(str(item) for item in summary_paths),
        f"Summary sampling rate: {sample_rate:g} Hz" if sample_rate else "Summary sampling rate: not stated",
        "Summary times are recording-relative seconds, not patient-session timestamps.",
        f"Recordings listed: {len(rows)}",
        f"EDF recordings downloaded: {sum(row['_downloaded'] for row in rows)}",
        f"Recordings marked as containing seizures: {len(seizure_rows)}",
        f"Downloaded seizure recordings: {', '.join(downloaded_seizures) if downloaded_seizures else 'none'}",
        f"Missing seizure recordings: {', '.join(missing_seizures) if missing_seizures else 'none'}",
        f"Separate .edf.seizures files found: {len(sidecar_paths)}",
        f"Sidecars parsed and matching summary intervals: {matching_sidecars}",
        f"Rows with validation issues: {len(failed_rows)}",
        f"Rows with invalid/conflicting annotations: {len(invalid_rows)}",
        f"Rows with missing annotation information: {len(missing_information)}",
        "",
        "Sidecar format investigation",
        "----------------------------",
        "The sidecars are binary WFDB annotation files, not plain text. Their format",
        "was read through WFDB's annotation reader (wfdb.rdann), rather than decoding",
        "bytes at guessed offsets. PhysioNet's WFDB annotation specification describes",
        "the binary MIT and AHA formats and automatic format recognition:",
        "https://physionet.org/physiotools/wag/annot-5.htm",
        "The authoritative rdann reader documentation is:",
        "https://physionet.org/physiotools/wag/rdann-1.htm",
        "For this dataset, each available sidecar yielded paired '[' and ']' sample",
        "markers. Their sample positions were converted using the summary/EDF sampling",
        "rate and compared against the summary intervals. Summary and sidecar",
        "intervals are recorded independently; disagreements are marked for review,",
        "and neither source is silently selected as the sole annotation truth.",
        "",
        "Per-recording audit",
        "-------------------",
        "recording | local EDF | seizure | EDF duration (s) | summary interval(s)",
    ]
    for row in rows:
        lines.append(
            f"{row['recording']} | {'yes' if row['_downloaded'] else 'no'} | "
            f"{'yes' if row['_contains_seizure'] else 'no'} | "
            f"{row['edf_duration_seconds'] or 'not available'} | "
            f"{_interval_text(row['_intervals'])}"
        )
    lines.extend(["", "Validation findings", "-------------------"])
    if failed_rows:
        for row in failed_rows:
            lines.append(f"{row['recording']}: " + "; ".join(row["_issues"]))
    else:
        lines.append("No validation issues found.")
    if sidecar_errors:
        lines.extend(["", "Sidecar reader errors", "---------------------"])
        for name, message in sorted(sidecar_errors.items()):
            lines.append(f"{name}: {message}")
    lines.extend(
        [
            "",
            "Visualisation",
            "-------------",
            (
                f"Saved seizure-onset plot: {plot_path}"
                if plot_path is not None
                else "No downloaded seizure-containing EDF was available; no plot was generated."
            ),
            "The plot labels the event as an EEG segment with expert-provided seizure annotations.",
            "",
            "Validation scope",
            "----------------",
            "EDF duration checks are performed only for recordings available locally and",
            "readable by MNE. Missing EDFs are included in the CSV and are not assigned",
            "fabricated durations or labels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(data_dir: Path, patient: str, output_dir: Path) -> tuple[Path, Path, Path | None]:
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    summary_paths = _find_summaries(data_dir, patient)
    summary_path = summary_paths[0]
    source_records: list[dict[str, Any]] = []
    summary_rates: list[float] = []
    for path in summary_paths:
        parsed, rate = parse_patient_summary(path)
        source_records.extend(parsed)
        if rate is not None:
            summary_rates.append(rate)
    summary_rate = (
        summary_rates[0]
        if summary_rates and len(set(summary_rates)) == 1
        else None
    )
    grouped = _group_summary_records(source_records)
    edf_paths = sorted(data_dir.rglob(f"{patient}_*.edf"))
    edf_paths_by_absolute_path = {path.resolve(): path for path in edf_paths}
    sidecar_paths = sorted(data_dir.rglob(f"{patient}_*.edf.seizures"))
    sidecars_by_edf_path = {
        path.with_name(path.name.removesuffix(".seizures")).resolve(): path
        for path in sidecar_paths
    }

    try:
        import mne
    except ImportError as exc:
        raise RuntimeError(
            "MNE is required to validate local EDF durations. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    if importlib.util.find_spec("wfdb") is None:
        raise RuntimeError(
            "WFDB is required to inspect .edf.seizures files. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        )

    rows: list[dict[str, Any]] = []
    durations: dict[Path, float] = {}
    sampling_rates: dict[Path, float] = {}
    load_errors: dict[Path, str] = {}
    for edf_path in edf_paths:
        identity_path = edf_path.resolve()
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
            sampling_rates[identity_path] = float(raw.info["sfreq"])
            durations[identity_path] = int(raw.n_times) / sampling_rates[identity_path]
        except Exception as exc:
            load_errors[identity_path] = f"{type(exc).__name__}: {exc}"

    sidecar_results: dict[Path, dict[str, Any]] = {}
    sidecar_errors: dict[str, str] = {}
    for edf_identity_path, sidecar_path in sidecars_by_edf_path.items():
        rate = sampling_rates.get(edf_identity_path, summary_rate)
        result = read_wfdb_sidecar(sidecar_path, rate)
        sidecar_results[edf_identity_path] = result
        if result["error"]:
            sidecar_errors[str(sidecar_path)] = result["error"]

    for filename, source_records_for_file in grouped.items():
        intervals = [
            interval
            for record in source_records_for_file
            for interval in record["intervals"]
        ]
        declared_counts = [
            record["declared_count"]
            for record in source_records_for_file
            if record["declared_count"] is not None
        ]
        if len(set(declared_counts)) > 1:
            count_value: str | int = _json(sorted(set(declared_counts)))
        elif declared_counts:
            count_value = declared_counts[0]
        else:
            count_value = ""
        expected_path = (summary_path.parent / filename).resolve()
        matching_paths = [
            path.resolve() for path in edf_paths if path.name.casefold() == filename.casefold()
        ]
        if expected_path in edf_paths_by_absolute_path:
            edf_identity_path: Path | None = expected_path
            path_issue = None
        elif len(matching_paths) == 1:
            edf_identity_path = matching_paths[0]
            path_issue = None
        elif len(matching_paths) > 1:
            edf_identity_path = None
            path_issue = "AMBIGUOUS_LOCAL_EDF_MATCH"
        else:
            edf_identity_path = None
            path_issue = None
        downloaded = edf_identity_path is not None
        duration = durations.get(edf_identity_path) if edf_identity_path else None
        sidecar_identity_path = edf_identity_path or expected_path
        sidecar = sidecar_results.get(sidecar_identity_path)
        rate = (
            sampling_rates.get(edf_identity_path, summary_rate)
            if edf_identity_path
            else summary_rate
        )
        comparison = (
            compare_summary_sidecar(
                intervals,
                sidecar["intervals"],
                rate,
                summary_declared_count=(
                    declared_counts[0]
                    if declared_counts and len(set(declared_counts)) == 1
                    else None
                ),
            )
            if sidecar is not None and not sidecar["error"]
            else "not_available"
        )
        issues = _build_validation_issues(
            filename,
            source_records_for_file,
            intervals,
            declared_counts,
            duration,
            sidecar,
            rate,
        )
        if len(summary_paths) > 1:
            issues.append(
                f"AMBIGUOUS_PATIENT_SUMMARIES:{len(summary_paths)} summary files found"
            )
        if edf_identity_path is not None and edf_identity_path in load_errors:
            issues.append(f"EDF_READ_ERROR: {load_errors[edf_identity_path]}")
        if path_issue:
            issues.append(path_issue)
        if sidecar is not None and sidecar["error"]:
            issues.append(f"SIDECAR_READ_OR_FORMAT_ERROR: {sidecar['error']}")
        contains_seizure = any(count > 0 for count in declared_counts) or any(
            item["start_seconds"] is not None or item["end_seconds"] is not None
            for item in intervals
        )
        relative_identity_path = (
            edf_identity_path.relative_to(data_dir).as_posix()
            if edf_identity_path and edf_identity_path.is_relative_to(data_dir)
            else (
                expected_path.relative_to(data_dir).as_posix()
                if expected_path.is_relative_to(data_dir)
                else filename
            )
        )
        row = {
            "patient_id": patient,
            "recording_id": Path(filename).stem,
            "relative_path": relative_identity_path,
            "recording": filename,
            "downloaded": str(downloaded).lower(),
            "contains_seizure": str(contains_seizure).lower(),
            "summary_seizure_count": count_value,
            "seizure_intervals": _json(intervals),
            "edf_duration_seconds": f"{duration:.6f}" if duration is not None else "",
            "sidecar_available": str(sidecar is not None).lower(),
            "sidecar_annotation_samples": _json(sidecar["samples"]) if sidecar else "[]",
            "sidecar_seizure_intervals": _json(sidecar["intervals"]) if sidecar else "[]",
            "annotation_source": (
                "ambiguous_patient_summaries"
                if len(summary_paths) > 1
                else (
                    "patient_summary_and_wfdb_sidecar"
                    if sidecar is not None
                    else "patient_summary"
                )
            ),
            "annotation_agreement": comparison,
            "validation_status": "PASS" if not issues else "REVIEW",
            "validation_issues": "; ".join(dict.fromkeys(issues)),
            "_contains_seizure": contains_seizure,
            "_downloaded": downloaded,
            "_intervals": intervals,
            "_issues": list(dict.fromkeys(issues)),
            "_edf_path": (
                edf_paths_by_absolute_path.get(edf_identity_path)
                if edf_identity_path
                else None
            ),
            "_duration": duration,
        }
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{patient}_annotation_audit.csv"
    report_path = output_dir / f"{patient}_annotation_audit.txt"
    _write_csv(csv_path, rows)

    plot_path = _plot_seizure_recording(rows, output_dir, patient, mne)
    _write_report(
        report_path,
        patient,
        summary_paths,
        rows,
        sidecar_paths,
        summary_rate,
        sidecar_errors,
        plot_path,
    )
    return csv_path, report_path, plot_path


def _plot_seizure_recording(
    rows: list[dict[str, Any]],
    output_dir: Path,
    patient: str,
    mne: Any,
) -> Path | None:
    target = next(
        (
            row
            for row in rows
            if row["_contains_seizure"]
            and row["_edf_path"] is not None
            and row["_duration"] is not None
            and any(
                interval["start_seconds"] is not None
                and interval["end_seconds"] is not None
                for interval in row["_intervals"]
            )
        ),
        None,
    )
    if target is None:
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    interval = next(
        item
        for item in target["_intervals"]
        if item["start_seconds"] is not None and item["end_seconds"] is not None
    )
    raw = mne.io.read_raw_edf(target["_edf_path"], preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    onset = float(interval["start_seconds"])
    offset = float(interval["end_seconds"])
    segment_start = max(0.0, onset - 8.0)
    segment_end = min(target["_duration"], offset + 8.0)
    start_sample = int(round(segment_start * sfreq))
    stop_sample = int(round(segment_end * sfreq))
    eeg_picks = [
        name
        for name, channel_type in zip(raw.ch_names, raw.get_channel_types())
        if channel_type == "eeg"
    ][:5]
    if not eeg_picks:
        return None
    data = raw.get_data(picks=eeg_picks, start=start_sample, stop=stop_sample) * 1e6
    times = segment_start + np.arange(data.shape[1]) / sfreq
    fig, axes = plt.subplots(
        len(eeg_picks), 1, figsize=(12, max(6, len(eeg_picks) * 1.7)), sharex=True
    )
    axes_list = np.atleast_1d(axes)
    for index, (axis, channel_name) in enumerate(zip(axes_list, eeg_picks)):
        axis.plot(times, data[index], linewidth=0.65, color="black")
        axis.axvspan(onset, offset, color="orange", alpha=0.18)
        axis.axvline(onset, color="red", linestyle="--", linewidth=1.2, label="Seizure onset")
        axis.axvline(offset, color="blue", linestyle="--", linewidth=1.2, label="Seizure offset")
        axis.set_ylabel(f"{channel_name}\n(µV)", rotation=0, ha="right", va="center")
        axis.grid(True, alpha=0.2)
        if index == 0:
            axis.legend(loc="upper right", ncol=2, fontsize="small")
    axes_list[-1].set_xlabel("Recording-relative time (seconds)")
    fig.suptitle(
        f"EEG segment with expert-provided seizure annotations\n"
        f"{target['recording']} — onset {onset:g} s, offset {offset:g} s"
    )
    fig.tight_layout()
    plot_path = output_dir / f"{patient}_seizure_annotation_example.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CHB-MIT seizure summary intervals, EDF durations, and WFDB sidecars."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--patient", default="chb01")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        print(f"Dataset directory does not exist: {data_dir}", file=sys.stderr)
        return 2
    try:
        csv_path, report_path, plot_path = run_audit(data_dir, args.patient, output_dir)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 1

    print(f"CSV report: {csv_path}")
    print(f"Text report: {report_path}")
    if plot_path:
        print(f"Seizure plot: {plot_path}")
    else:
        print("No seizure plot generated: no downloadable, readable seizure EDF was available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
