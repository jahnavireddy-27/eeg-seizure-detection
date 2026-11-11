#!/usr/bin/env python3
"""Build a corpus-wide, metadata-only inventory of local CHB-MIT recordings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyedflib

from annotation_utils import (
    compare_summary_sidecar,
    parse_patient_summary,
    read_wfdb_sidecar,
    validate_annotations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "chbmit_dataset_manifest.csv"
PATIENT_PATTERN = re.compile(r"^chb\d+$", re.IGNORECASE)
SUMMARY_SUFFIX = "-summary.txt"
HASH_CHUNK_BYTES = 1024 * 1024
MANIFEST_FIELDS = (
    "patient_id",
    "recording_id",
    "relative_path",
    "edf_filename",
    "expected_status",
    "download_status",
    "recording_duration_seconds",
    "seizure_status",
    "seizure_intervals",
    "summary_seizure_intervals",
    "sidecar_seizure_intervals",
    "annotation_source",
    "annotation_agreement",
    "annotation_validation_status",
    "annotation_validation_issues",
    "annotation_sidecar_present",
    "annotation_sidecar_relative_path",
    "edf_size_bytes",
    "edf_sha256",
    "processing_conversion_status",
    "processing_segmentation_status",
    "processing_feature_extraction_status",
    "inventory_issues",
)


def _patient_id_from_summary(path: Path) -> str | None:
    name = path.name
    if not name.lower().endswith(SUMMARY_SUFFIX):
        return None
    patient_id = name[: -len(SUMMARY_SUFFIX)]
    return patient_id if PATIENT_PATTERN.fullmatch(patient_id) else None


def _patient_id_for_recording(path: Path, dataset_root: Path) -> str | None:
    for parent in path.parents:
        if parent == dataset_root.parent:
            break
        if PATIENT_PATTERN.fullmatch(parent.name):
            return parent.name
        if parent == dataset_root:
            break
    match = re.match(r"^(chb\d+)_", path.name, re.IGNORECASE)
    return match.group(1) if match else None


def _relative_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Dataset input escaped root: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_edf_metadata(path: Path) -> tuple[float | None, float | None]:
    """Return recording duration and common signal rate without reading samples."""
    with pyedflib.EdfReader(str(path)) as reader:
        duration_value = float(reader.file_duration)
        if not math.isfinite(duration_value) or duration_value <= 0:
            raise ValueError(f"EDF reports invalid duration {duration_value!r}.")
        frequencies = [float(value) for value in reader.getSampleFrequencies()]
        if not frequencies:
            raise ValueError("EDF contains no signal sampling frequencies.")
        if any(not math.isfinite(value) or value <= 0 for value in frequencies):
            raise ValueError("EDF contains invalid signal sampling frequencies.")
        rate = frequencies[0]
        if any(not math.isclose(value, rate, rel_tol=0.0, abs_tol=1e-9) for value in frequencies):
            rate = None
        return duration_value, rate


def _summary_annotation_state(
    source_records: list[dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]], list[str], list[int]]:
    """Return seizure status, source, preserved intervals, issues, and counts."""
    issues: list[str] = []
    intervals = [
        interval
        for record in source_records
        for interval in record["intervals"]
    ]
    declared_counts = [
        int(record["declared_count"])
        for record in source_records
        if record["declared_count"] is not None
    ]
    if not source_records:
        return "annotation_unavailable", "unavailable", [], issues, declared_counts
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
            issues.append("CONFLICTING_SUMMARY_RECORDS")
    if not declared_counts:
        return "annotation_unavailable", "patient_summary", intervals, issues, declared_counts
    if len(set(declared_counts)) > 1:
        issues.append("CONFLICTING_SUMMARY_RECORDS: declared seizure counts differ")
        return "annotation_unavailable", "patient_summary", intervals, issues, declared_counts
    if len(source_records) != 1:
        return "annotation_unavailable", "patient_summary", intervals, issues, declared_counts

    declared_count = declared_counts[0]
    if declared_count == 0:
        if intervals:
            issues.append("COUNT_INTERVAL_MISMATCH")
            return "annotation_unavailable", "patient_summary", intervals, issues, declared_counts
        return "no_seizures_annotated", "patient_summary", [], issues, declared_counts

    if declared_count != len(intervals):
        issues.append(
            f"COUNT_INTERVAL_MISMATCH: summary declares {declared_count}, "
            f"but {len(intervals)} interval slot(s) were parsed"
        )
    complete_intervals = [
        item
        for item in intervals
        if item["start_seconds"] is not None and item["end_seconds"] is not None
    ]
    if len(complete_intervals) != declared_count:
        issues.append("MISSING_ANNOTATION_INFORMATION")
        return "annotation_unavailable", "patient_summary", intervals, issues, declared_counts
    return "seizures_annotated", "patient_summary", intervals, issues, declared_counts


def _sidecar_status(
    *,
    sidecar_path: Path | None,
    summary_records: list[dict[str, Any]],
    summary_intervals: list[dict[str, Any]],
    sample_rate: float | None,
    duration_seconds: float | None,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    if sidecar_path is None:
        return "absent", None, []
    sidecar = read_wfdb_sidecar(sidecar_path, sample_rate)
    if sidecar["error"]:
        return "unreadable_or_invalid", sidecar, [sidecar["error"]]

    intervals = sidecar["intervals"]
    for index, interval in enumerate(intervals, start=1):
        sample_start = interval.get("start_sample")
        sample_end = interval.get("end_sample")
        if sample_start is not None and sample_end is not None:
            if sample_start < 0 or sample_end <= sample_start:
                return "unreadable_or_invalid", sidecar, [
                    f"Invalid sidecar sample interval {index}: "
                    f"{sample_start} to {sample_end}."
                ]
        start = interval["start_seconds"]
        end = interval["end_seconds"]
        if start is None or end is None:
            return "unvalidated_missing_sample_rate", sidecar, [
                "Sidecar sample indices could not be converted without a sampling rate."
            ]
        if start < 0 or end <= start:
            return "unreadable_or_invalid", sidecar, [
                f"Invalid sidecar interval {index}: {start:g} to {end:g} seconds."
            ]
        if duration_seconds is not None and end > duration_seconds:
            return "unreadable_or_invalid", sidecar, [
                f"Sidecar interval {index} exceeds EDF duration."
            ]

    if summary_records:
        declared_counts = [
            record["declared_count"]
            for record in summary_records
            if record["declared_count"] is not None
        ]
        declared_count = declared_counts[0] if len(set(declared_counts)) == 1 and declared_counts else None
        comparison = compare_summary_sidecar(
            summary_intervals,
            intervals,
            sample_rate,
            summary_declared_count=declared_count,
        )
        if comparison == "agreement":
            return "validated_match", sidecar, []
        if comparison == "disagreement":
            return "mismatch", sidecar, [
                "WFDB sidecar intervals do not agree with the patient summary."
            ]
        if comparison == "count_agreement_times_unavailable":
            return "validated_count_match", sidecar, [
                "Sidecar interval count matches; summary timing is incomplete."
            ]
        return "parsed_unconfirmed", sidecar, [
            "The patient summary does not contain enough information to compare intervals."
        ]
    return "parsed_unconfirmed", sidecar, [
        "No patient summary was available to cross-validate the sidecar."
    ]


def _record_row(
    *,
    patient_id: str,
    recording_filename: str,
    relative_path: str,
    expected_status: str,
    edf_path: Path | None,
    summary_records: list[dict[str, Any]],
    sidecar_path: Path | None,
    dataset_root: Path,
    summary_sample_rate: float | None = None,
) -> dict[str, Any]:
    recording_id = Path(recording_filename).stem
    seizure_status, annotation_source, intervals, annotation_issues, declared_counts = (
        _summary_annotation_state(summary_records)
    )
    duration_seconds: float | None = None
    sample_rate: float | None = None
    edf_size: int | None = None
    checksum: str | None = None
    download_status = "not_downloaded"
    inventory_issues: list[str] = []

    if edf_path is not None:
        try:
            duration_seconds, sample_rate = _read_edf_metadata(edf_path)
            edf_size = edf_path.stat().st_size
            checksum = _sha256(edf_path)
            download_status = "downloaded_validated"
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            download_status = "missing_or_unreadable"
            inventory_issues.append(f"EDF_READ_OR_HASH_ERROR: {type(exc).__name__}: {exc}")

    if sample_rate is None:
        sample_rate = summary_sample_rate

    sidecar_validation, sidecar_result, sidecar_issues = _sidecar_status(
        sidecar_path=sidecar_path,
        summary_records=summary_records,
        summary_intervals=intervals,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )
    if sidecar_issues:
        annotation_issues.extend(sidecar_issues)
    if summary_records:
        annotation_issues.extend(
            validate_annotations(
                recording_filename,
                summary_records,
                intervals,
                declared_counts,
                duration_seconds,
                sidecar_result,
                sample_rate,
            )
        )
    if sidecar_path is not None and summary_records:
        if sidecar_validation in {"validated_match", "validated_count_match"}:
            annotation_source = "patient_summary_and_wfdb_sidecar"
        elif sidecar_validation == "mismatch":
            annotation_source = "patient_summary_and_wfdb_sidecar_conflict"
        elif sidecar_validation == "unreadable_or_invalid":
            annotation_source = "patient_summary_and_unreadable_wfdb_sidecar"
        else:
            annotation_source = "patient_summary_and_wfdb_sidecar_unconfirmed"
    if sidecar_path is not None and sidecar_result and not sidecar_result["error"]:
        sidecar_intervals = sidecar_result["intervals"]
        summary_confirms_none = any(
            record["declared_count"] == 0 for record in summary_records
        )
        if (
            seizure_status == "annotation_unavailable"
            and not summary_confirms_none
            and sidecar_intervals
            and sidecar_validation in {"parsed_unconfirmed", "validated_count_match", "unvalidated_missing_sample_rate"}
        ):
            seizure_status = "seizures_annotated"
            annotation_source = (
                "patient_summary_and_wfdb_sidecar"
                if summary_records
                else "wfdb_sidecar"
            )
            intervals = [
                {
                    "start_sample": item["start_sample"],
                    "end_sample": item["end_sample"],
                    "start_seconds": item["start_seconds"],
                    "end_seconds": item["end_seconds"],
                    "duration_seconds": item["duration_seconds"],
                }
                for item in sidecar_intervals
            ]

    if sidecar_validation == "mismatch":
        seizure_status = "annotation_unavailable"
        intervals = []

    if sidecar_path is not None:
        try:
            sidecar_relative_path = _relative_path(sidecar_path, dataset_root)
        except ValueError:
            sidecar_relative_path = ""
            inventory_issues.append("SIDECAR_PATH_OUTSIDE_DATASET_ROOT")
    else:
        sidecar_relative_path = ""

    return {
        "patient_id": patient_id,
        "recording_id": recording_id,
        "relative_path": relative_path,
        "edf_filename": recording_filename,
        "expected_status": expected_status,
        "download_status": download_status,
        "recording_duration_seconds": (
            f"{duration_seconds:.9f}" if duration_seconds is not None else ""
        ),
        "seizure_status": seizure_status,
        "seizure_intervals": json.dumps(intervals, sort_keys=True, separators=(",", ":")),
        "summary_seizure_intervals": json.dumps(
            _summary_annotation_state(summary_records)[2],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "sidecar_seizure_intervals": json.dumps(
            sidecar_result["intervals"] if sidecar_result else [],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "annotation_source": annotation_source,
        "annotation_agreement": (
            "not_available"
            if sidecar_path is None
            else {
                "validated_match": "agreement",
                "validated_count_match": "count_agreement_times_unavailable",
                "mismatch": "disagreement",
            }.get(sidecar_validation, "unavailable")
        ),
        "annotation_validation_status": sidecar_validation,
        "annotation_validation_issues": "; ".join(dict.fromkeys(annotation_issues)),
        "annotation_sidecar_present": str(sidecar_path is not None).lower(),
        "annotation_sidecar_relative_path": sidecar_relative_path,
        "edf_size_bytes": str(edf_size) if edf_size is not None else "",
        "edf_sha256": checksum or "",
        "processing_conversion_status": "not_assessed",
        "processing_segmentation_status": "not_assessed",
        "processing_feature_extraction_status": "not_assessed",
        "inventory_issues": "; ".join(dict.fromkeys(inventory_issues)),
    }


def build_dataset_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    """Inventory expected and locally found EDFs using patient-scoped identities."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CHB-MIT dataset root does not exist: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    summaries_by_patient: dict[str, list[Path]] = {}
    edfs_by_patient: dict[str, list[Path]] = {}
    sidecars_by_path: dict[Path, Path] = {}
    patient_ids: set[str] = set()

    for path in files:
        summary_patient = _patient_id_from_summary(path)
        if summary_patient:
            patient_ids.add(summary_patient)
            summaries_by_patient.setdefault(summary_patient, []).append(path)
        if path.name.lower().endswith(".edf.seizures"):
            edf_path = path.with_name(path.name[: -len(".seizures")])
            sidecars_by_path[edf_path] = path
            patient_id = _patient_id_for_recording(edf_path, root)
            if patient_id:
                patient_ids.add(patient_id)
        elif path.suffix.lower() == ".edf":
            patient_id = _patient_id_for_recording(path, root)
            if patient_id:
                patient_ids.add(patient_id)
                edfs_by_patient.setdefault(patient_id, []).append(path)

    summaries_by_patient = {
        key: sorted(value) for key, value in summaries_by_patient.items()
    }
    edfs_by_patient = {key: sorted(value) for key, value in edfs_by_patient.items()}
    rows: list[dict[str, Any]] = []

    for patient_id in sorted(patient_ids, key=str.casefold):
        summary_paths = summaries_by_patient.get(patient_id, [])
        summary_records_by_filename: dict[str, list[dict[str, Any]]] = {}
        summary_file_for_record: dict[str, Path] = {}
        summary_sample_rates: list[float] = []
        summary_parse_issues: list[str] = []
        if len(summary_paths) > 1:
            summary_parse_issues.append(
                f"MULTIPLE_PATIENT_SUMMARIES:{len(summary_paths)}"
            )
        for summary_path in summary_paths:
            try:
                parsed, sample_rate = parse_patient_summary(summary_path)
            except OSError as exc:
                summary_parse_issues.append(
                    f"SUMMARY_READ_ERROR:{summary_path.name}:{type(exc).__name__}:{exc}"
                )
                continue
            if sample_rate is not None:
                summary_sample_rates.append(sample_rate)
            for record in parsed:
                filename = record["filename"]
                summary_records_by_filename.setdefault(filename, []).append(record)
                summary_file_for_record.setdefault(filename, summary_path)

        patient_edfs = edfs_by_patient.get(patient_id, [])
        rows_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
        consumed_edfs: set[Path] = set()

        for recording_filename in sorted(summary_records_by_filename, key=str.casefold):
            summary_path = summary_file_for_record[recording_filename]
            expected_path = (summary_path.parent / recording_filename).resolve()
            if not expected_path.is_relative_to(root):
                raise ValueError(
                    f"Summary recording path escapes dataset root: {recording_filename!r}"
                )
            candidates = [
                path
                for path in patient_edfs
                if path.name.casefold() == Path(recording_filename).name.casefold()
            ]
            edf_path: Path | None = None
            path_issue: str | None = None
            if expected_path.is_file():
                edf_path = expected_path
            elif len(candidates) == 1:
                edf_path = candidates[0]
            elif len(candidates) > 1:
                path_issue = "AMBIGUOUS_LOCAL_EDF_MATCH"

            actual_path = edf_path or expected_path
            relative = _relative_path(actual_path, root)
            sidecar = sidecars_by_path.get(actual_path)
            row = _record_row(
                patient_id=patient_id,
                recording_filename=Path(recording_filename).name,
                relative_path=relative,
                expected_status="expected_in_patient_summary",
                edf_path=edf_path,
                summary_records=summary_records_by_filename[recording_filename],
                sidecar_path=sidecar,
                dataset_root=root,
                summary_sample_rate=(
                    summary_sample_rates[0]
                    if len(summary_paths) == 1 and summary_sample_rates
                    else None
                ),
            )
            if len(summary_paths) > 1:
                row["seizure_status"] = "annotation_unavailable"
                row["annotation_source"] = "ambiguous_patient_summaries"
                row["annotation_validation_issues"] = "; ".join(
                    filter(
                        None,
                        [row["annotation_validation_issues"], *summary_parse_issues],
                    )
                )
            elif summary_parse_issues:
                row["seizure_status"] = "annotation_unavailable"
                row["annotation_source"] = "patient_summary"
                row["annotation_validation_issues"] = "; ".join(
                    filter(
                        None,
                        [row["annotation_validation_issues"], *summary_parse_issues],
                    )
                )
            if path_issue:
                row["download_status"] = "missing_or_unreadable"
                row["inventory_issues"] = path_issue
            identity = (patient_id, row["recording_id"], relative)
            if identity in rows_by_identity:
                raise ValueError(f"Duplicate corpus identity discovered: {identity}")
            rows_by_identity[identity] = row
            if edf_path is not None:
                consumed_edfs.add(edf_path)

        for edf_path in patient_edfs:
            if edf_path in consumed_edfs:
                continue
            relative = _relative_path(edf_path, root)
            sidecar = sidecars_by_path.get(edf_path)
            row = _record_row(
                patient_id=patient_id,
                recording_filename=edf_path.name,
                relative_path=relative,
                expected_status="not_listed_in_patient_summary",
                edf_path=edf_path,
                summary_records=[],
                sidecar_path=sidecar,
                dataset_root=root,
                summary_sample_rate=(
                    summary_sample_rates[0]
                    if len(summary_paths) == 1 and summary_sample_rates
                    else None
                ),
            )
            if not summary_paths:
                row["annotation_validation_status"] = (
                    "parsed_unconfirmed" if sidecar else "absent"
                )
            elif len(summary_paths) > 1:
                row["annotation_source"] = "ambiguous_patient_summaries"
                row["annotation_validation_issues"] = "; ".join(
                    filter(
                        None,
                        [row["annotation_validation_issues"], *summary_parse_issues],
                    )
                )
            elif summary_parse_issues:
                row["annotation_source"] = "patient_summary"
                row["annotation_validation_issues"] = "; ".join(
                    filter(
                        None,
                        [row["annotation_validation_issues"], *summary_parse_issues],
                    )
                )
            identity = (patient_id, row["recording_id"], relative)
            if identity in rows_by_identity:
                raise ValueError(f"Duplicate corpus identity discovered: {identity}")
            rows_by_identity[identity] = row

        represented_paths = {identity[2] for identity in rows_by_identity}
        for expected_edf_path, sidecar_path in sorted(sidecars_by_path.items()):
            sidecar_patient = _patient_id_for_recording(expected_edf_path, root)
            if (
                sidecar_patient != patient_id
                or expected_edf_path.is_file()
            ):
                continue
            relative = _relative_path(expected_edf_path, root)
            if relative in represented_paths:
                continue
            row = _record_row(
                patient_id=patient_id,
                recording_filename=expected_edf_path.name,
                relative_path=relative,
                expected_status="not_listed_in_patient_summary",
                edf_path=None,
                summary_records=[],
                sidecar_path=sidecar_path,
                dataset_root=root,
            )
            identity = (patient_id, row["recording_id"], relative)
            if identity in rows_by_identity:
                raise ValueError(f"Duplicate corpus identity discovered: {identity}")
            rows_by_identity[identity] = row

        rows.extend(rows_by_identity[key] for key in sorted(rows_by_identity))

    return rows


def _existing_manifest_is_ours(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream), [])
    except (OSError, UnicodeError, csv.Error):
        return False
    legacy_fields = (
        "patient_id",
        "recording_id",
        "relative_path",
        "edf_filename",
        "expected_status",
        "download_status",
        "recording_duration_seconds",
        "seizure_status",
        "seizure_intervals",
        "annotation_source",
        "annotation_validation_status",
        "annotation_validation_issues",
        "annotation_sidecar_present",
        "annotation_sidecar_relative_path",
        "edf_size_bytes",
        "edf_sha256",
        "processing_conversion_status",
        "processing_segmentation_status",
        "processing_feature_extraction_status",
        "inventory_issues",
    )
    return tuple(header) in {MANIFEST_FIELDS, legacy_fields}


def write_dataset_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Atomically publish the manifest; refuse to replace unrelated files."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not _existing_manifest_is_ours(output_path):
        raise FileExistsError(
            f"Refusing to overwrite an existing non-manifest file: {output_path}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(
                stream,
                fieldnames=MANIFEST_FIELDS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    try:
        rows = build_dataset_manifest(args.dataset_root)
        output_path = write_dataset_manifest(args.output, rows)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Dataset manifest failed: {exc}", file=sys.stderr)
        return 1
    print(f"Recordings inventoried: {len(rows)}")
    print(f"Manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
