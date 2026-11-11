"""Shared CHB-MIT summary and WFDB seizure-annotation parsing/validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def parse_patient_summary(
    summary_path: Path,
) -> tuple[list[dict[str, Any]], float | None]:
    """Parse recording-relative seizure intervals without rounding their values."""
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    rate_match = re.search(
        rf"Data Sampling Rate:\s*({NUMBER_PATTERN})\s*Hz", text, re.IGNORECASE
    )
    sample_rate = float(rate_match.group(1)) if rate_match else None
    block_pattern = re.compile(
        r"File Name:\s*(?P<filename>[^\r\n]+)(?P<body>.*?)(?=File Name:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    records: list[dict[str, Any]] = []
    for match in block_pattern.finditer(text):
        body = match.group("body")
        count_match = re.search(
            r"Number of Seizures in File:\s*(\d+)", body, re.IGNORECASE
        )
        starts = re.findall(
            rf"Seizure Start Time:\s*({NUMBER_PATTERN})\s*seconds",
            body,
            re.IGNORECASE,
        )
        ends = re.findall(
            rf"Seizure End Time:\s*({NUMBER_PATTERN})\s*seconds",
            body,
            re.IGNORECASE,
        )
        declared_count = int(count_match.group(1)) if count_match else None
        interval_count = max(len(starts), len(ends), declared_count or 0)
        intervals = []
        for index in range(interval_count):
            start = float(starts[index]) if index < len(starts) else None
            end = float(ends[index]) if index < len(ends) else None
            intervals.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": (
                        end - start
                        if start is not None and end is not None
                        else None
                    ),
                }
            )
        records.append(
            {
                "filename": match.group("filename").strip(),
                "declared_count": declared_count,
                "intervals": intervals,
            }
        )
    return records, sample_rate


def read_wfdb_sidecar(path: Path, sample_rate: float | None) -> dict[str, Any]:
    """Parse a WFDB annotation sidecar using WFDB, never guessed byte offsets."""
    result: dict[str, Any] = {
        "available": True,
        "intervals": [],
        "samples": [],
        "error": None,
    }
    try:
        import wfdb

        annotations = wfdb.rdann(str(path.with_suffix("")), "seizures")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["samples"] = [
        {"sample": int(sample), "symbol": symbol}
        for sample, symbol in zip(annotations.sample, annotations.symbol)
    ]
    start_sample: int | None = None
    for sample_value, symbol in zip(annotations.sample, annotations.symbol):
        sample = int(sample_value)
        if symbol == "[":
            if start_sample is not None:
                result["error"] = "A second onset marker occurred before an offset marker."
            start_sample = sample
        elif symbol == "]":
            if start_sample is None:
                result["error"] = "An offset marker occurred without a preceding onset marker."
                continue
            start_seconds = start_sample / sample_rate if sample_rate else None
            end_seconds = sample / sample_rate if sample_rate else None
            result["intervals"].append(
                {
                    "start_sample": start_sample,
                    "end_sample": sample,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": (
                        end_seconds - start_seconds
                        if start_seconds is not None and end_seconds is not None
                        else None
                    ),
                }
            )
            start_sample = None
    if start_sample is not None:
        result["error"] = "An onset marker has no matching offset marker."
    return result


def compare_summary_sidecar(
    summary_intervals: list[dict[str, Any]],
    sidecar_intervals: list[dict[str, Any]],
    sample_rate: float | None,
    *,
    summary_declared_count: int | None = None,
) -> str:
    """Compare two annotation sources, allowing at most one sample of timing error."""
    complete_summary_intervals = [
        item
        for item in summary_intervals
        if item["start_seconds"] is not None and item["end_seconds"] is not None
    ]
    if complete_summary_intervals and len(complete_summary_intervals) == len(summary_intervals):
        if len(complete_summary_intervals) != len(sidecar_intervals) or sample_rate is None:
            return "disagreement"
        tolerance = 1.0 / sample_rate
        for summary, sidecar in zip(complete_summary_intervals, sidecar_intervals):
            start, end = summary["start_seconds"], summary["end_seconds"]
            sidecar_start = sidecar["start_seconds"]
            sidecar_end = sidecar["end_seconds"]
            if None in (start, end, sidecar_start, sidecar_end):
                return "unavailable"
            if (
                abs(sidecar_start - start) > tolerance
                or abs(sidecar_end - end) > tolerance
            ):
                return "disagreement"
        return "agreement"
    if summary_declared_count == 0:
        return "agreement" if not sidecar_intervals else "disagreement"
    if summary_declared_count is not None:
        if len(sidecar_intervals) != summary_declared_count:
            return "disagreement"
        return "count_agreement_times_unavailable"
    return "unavailable"


def validate_annotations(
    filename: str,
    source_records: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    declared_counts: list[int],
    duration_seconds: float | None,
    sidecar: dict[str, Any] | None,
    sample_rate: float | None,
) -> list[str]:
    """Return explicit consistency and interval-validation issues."""
    issues: list[str] = []
    if len(source_records) > 1:
        issues.append(f"DUPLICATE_SUMMARY_RECORD:{len(source_records)} entries")
        signatures = {
            (
                record["declared_count"],
                tuple(
                    (item["start_seconds"], item["end_seconds"])
                    for item in record["intervals"]
                ),
            )
            for record in source_records
        }
        if len(signatures) > 1:
            issues.append("CONFLICTING_SUMMARY_RECORDS: repeated filename has different annotations")

    interval_signatures = [
        (interval["start_seconds"], interval["end_seconds"])
        for interval in intervals
    ]
    if len(interval_signatures) != len(set(interval_signatures)):
        issues.append("DUPLICATE_ANNOTATION: repeated seizure interval")

    complete_intervals = [
        item
        for item in intervals
        if item["start_seconds"] is not None and item["end_seconds"] is not None
    ]
    for index, interval in enumerate(intervals, start=1):
        start, end = interval["start_seconds"], interval["end_seconds"]
        if start is None or end is None:
            issues.append(
                f"MISSING_ANNOTATION_INFORMATION: interval {index} lacks a start or end"
            )
            continue
        if start < 0:
            issues.append(f"NEGATIVE_START_TIME: interval {index} starts at {start:g} seconds")
        if end < start:
            issues.append(f"END_BEFORE_START: interval {index} ends at {end:g} before {start:g} seconds")
        if duration_seconds is not None and (start > duration_seconds or end > duration_seconds):
            issues.append(
                f"INTERVAL_EXCEEDS_EDF_DURATION: interval {index} ends at {end:g} seconds; "
                f"EDF duration is {duration_seconds:g} seconds"
            )

    ordered = sorted(
        complete_intervals,
        key=lambda item: (item["start_seconds"], item["end_seconds"]),
    )
    for previous, current in zip(ordered, ordered[1:]):
        if current["start_seconds"] < previous["end_seconds"]:
            issues.append("CONFLICTING_ANNOTATIONS: seizure intervals overlap")
            break

    if declared_counts:
        if len(set(declared_counts)) > 1:
            issues.append("CONFLICTING_SUMMARY_RECORDS: declared seizure counts differ")
        declared_count = declared_counts[0]
        if declared_count != len(intervals):
            issues.append(
                f"COUNT_INTERVAL_MISMATCH: summary declares {declared_count}, "
                f"but {len(intervals)} interval slot(s) were parsed"
            )
        if declared_count > 0 and not complete_intervals:
            issues.append("MISSING_ANNOTATION_INFORMATION: seizures declared but no complete interval")
    elif source_records:
        issues.append("MISSING_ANNOTATION_INFORMATION: seizure count is absent from summary")

    if sidecar is not None:
        if sidecar["error"]:
            issues.append(f"SIDECAR_READ_OR_FORMAT_ERROR: {sidecar['error']}")
        else:
            sidecar_signatures = [
                (item.get("start_sample"), item.get("end_sample"))
                for item in sidecar["intervals"]
            ]
            if len(sidecar_signatures) != len(set(sidecar_signatures)):
                issues.append("DUPLICATE_ANNOTATION: repeated WFDB sidecar interval")
            for index, interval in enumerate(sidecar["intervals"], start=1):
                start, end = interval["start_seconds"], interval["end_seconds"]
                sample_start = interval.get("start_sample")
                sample_end = interval.get("end_sample")
                if sample_start is not None and sample_start < 0:
                    issues.append(f"NEGATIVE_START_TIME: WFDB sidecar interval {index}")
                if sample_start is not None and sample_end is not None and sample_end <= sample_start:
                    issues.append(f"END_BEFORE_START: WFDB sidecar interval {index}")
                if start is not None and end is not None:
                    if start < 0:
                        issues.append(f"NEGATIVE_START_TIME: WFDB sidecar interval {index}")
                    if end < start:
                        issues.append(f"END_BEFORE_START: WFDB sidecar interval {index}")
                    if duration_seconds is not None and end > duration_seconds:
                        issues.append(
                            f"INTERVAL_EXCEEDS_EDF_DURATION: WFDB sidecar interval {index}"
                        )

            summary_count = declared_counts[0] if declared_counts else None
            comparison = compare_summary_sidecar(
                intervals, sidecar["intervals"], sample_rate,
                summary_declared_count=summary_count,
            )
            if comparison == "disagreement":
                issues.append("SIDECAR_SUMMARY_CONFLICT: WFDB sidecar and summary disagree")

    return list(dict.fromkeys(issues))
