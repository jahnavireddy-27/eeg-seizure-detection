#!/usr/bin/env python3
"""Resumable, single-recording-at-a-time CHB-MIT processing orchestrator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import edf_to_csv
import extract_features
import segment_windows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_CORPUS_MANIFEST = PROJECT_ROOT / "results" / "chbmit_dataset_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "corpus_pipeline"
STATE_FILENAME = ".process_chbmit_state.json"
STAGES = ("conversion", "segmentation", "feature_extraction")
SHA256_CHUNK_BYTES = 1024 * 1024
MANIFEST_FIELDS = {
    "patient_id",
    "recording_id",
    "relative_path",
    "edf_filename",
    "download_status",
    "recording_duration_seconds",
    "seizure_status",
    "seizure_intervals",
    "annotation_agreement",
}


@dataclass(frozen=True)
class PipelineConfiguration:
    conversion_chunk_seconds: float = edf_to_csv.DEFAULT_CHUNK_SECONDS
    window_seconds: float = segment_windows.DEFAULT_WINDOW_SECONDS
    overlap_fraction: float = segment_windows.DEFAULT_OVERLAP_FRACTION
    label_policy: str = segment_windows.DEFAULT_POLICY
    feature_families: tuple[str, ...] = extract_features.FEATURE_FAMILIES
    wavelet: str = "db4"
    wavelet_mode: str = "symmetric"
    feature_chunk_rows: int = extract_features.DEFAULT_CHUNK_ROWS

    def validate(self) -> None:
        if self.conversion_chunk_seconds <= 0:
            raise ValueError("conversion_chunk_seconds must be positive.")
        if self.feature_chunk_rows <= 0:
            raise ValueError("feature_chunk_rows must be positive.")
        segment_windows.WindowConfiguration(
            self.window_seconds, self.overlap_fraction, self.label_policy
        ).validate()
        extract_features.FeatureConfiguration(
            feature_families=self.feature_families,
            wavelet=self.wavelet,
            wavelet_mode=self.wavelet_mode,
        ).validate()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": _now(), "recordings": {}, "errors": []}
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("schema_version") != 1 or not isinstance(state.get("recordings"), dict):
        raise ValueError(f"Unsupported or malformed processing state: {path}")
    state.setdefault("errors", [])
    return state


def _record_key(patient_id: str, relative_path: str) -> str:
    return f"{patient_id.casefold()}::{Path(relative_path).as_posix()}"


def _identity_slug(relative_path: str) -> str:
    stem = Path(relative_path).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "recording"
    suffix = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem}__{suffix}"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = MANIFEST_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Corpus manifest is missing fields: " + ", ".join(sorted(missing))
            )
        rows = [dict(row) for row in reader]
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        identity = (
            row["patient_id"].strip(),
            row["recording_id"].strip(),
            Path(row["relative_path"]).as_posix(),
        )
        if not all(identity):
            raise ValueError(f"Manifest contains an incomplete recording identity: {row}")
        if identity in identities:
            raise ValueError(f"Duplicate corpus recording identity: {identity}")
        identities.add(identity)
    return sorted(
        rows,
        key=lambda row: (
            row["patient_id"].casefold(),
            Path(row["relative_path"]).as_posix().casefold(),
        ),
    )


def _selected_rows(
    rows: Iterable[dict[str, str]],
    patients: set[str] | None,
    recordings: set[str] | None,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        patient = row["patient_id"].strip()
        relative = Path(row["relative_path"]).as_posix()
        if patients is not None and patient.casefold() not in patients:
            continue
        if recordings is not None and not {
            row["recording_id"].casefold(),
            row["edf_filename"].casefold(),
            Path(relative).as_posix().casefold(),
            f"{patient}/{relative}".casefold(),
        }.intersection(recordings):
            continue
        selected.append(row)
    return selected


def _stage_entry(state: dict[str, Any], key: str, stage: str) -> dict[str, Any]:
    record = state["recordings"].setdefault(
        key,
        {
            "patient_id": key.split("::", 1)[0],
            "relative_path": key.split("::", 1)[1],
            "stages": {},
        },
    )
    return record["stages"].setdefault(stage, {"status": "Pending", "retryable": True})


def _initialize_recording(
    state_path: Path,
    state: dict[str, Any],
    key: str,
    patient_id: str,
    recording_id: str,
    relative_path: str,
) -> None:
    record = state["recordings"].setdefault(key, {})
    record.update(
        {
            "patient_id": patient_id,
            "recording_id": recording_id,
            "relative_path": relative_path,
        }
    )
    stages = record.setdefault("stages", {})
    for stage in STAGES:
        stages.setdefault(
            stage,
            {"status": "Pending", "created_at": _now(), "retryable": True},
        )
    _save_state(state_path, state)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    _atomic_write_json(path, state)


def _candidate_output_dir(
    stage_root: Path,
    identity_slug: str,
    input_checksum: str,
    config_checksum: str,
    prior_entry: dict[str, Any],
    same_work: bool,
) -> Path:
    if same_work and prior_entry.get("output_dir"):
        return Path(prior_entry["output_dir"])
    base = stage_root / (
        f"{identity_slug}__{input_checksum[:8]}__{config_checksum[:8]}"
    )
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}__{suffix}")
        suffix += 1
    return candidate


def _is_verified_complete(
    entry: dict[str, Any],
    input_checksum: str,
    config_checksum: str,
) -> bool:
    if (
        entry.get("status") != "Completed"
        or entry.get("input_checksum") != input_checksum
        or entry.get("configuration_checksum") != config_checksum
    ):
        return False
    output_path = Path(entry.get("output_path", ""))
    if not output_path.is_file():
        return False
    try:
        return sha256_file(output_path) == entry.get("output_checksum")
    except OSError:
        return False


def _validate_csv_output(path: Path, expected_rows: int | None = None) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Stage output is missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Stage output has no CSV header: {path}")
        row_count = sum(1 for _ in reader)
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            f"Stage output has {row_count} data rows; expected {expected_rows}: {path}"
        )
    return row_count


def _skip_stage(
    state_path: Path,
    state: dict[str, Any],
    key: str,
    stage: str,
    reason: str,
) -> None:
    entry = _stage_entry(state, key, stage)
    entry.update(
        {
            "status": "Skipped",
            "reason": reason,
            "updated_at": _now(),
            "retryable": False,
        }
    )
    _save_state(state_path, state)
    print(f"  {stage}: skipped ({reason})")


def _execute_stage(
    *,
    state_path: Path,
    state: dict[str, Any],
    key: str,
    stage: str,
    input_checksum: str,
    config: dict[str, Any],
    base_stage_root: Path,
    identity_slug: str,
    execute: Any,
    expected_rows: int | None = None,
) -> tuple[bool, Path | None, int | None]:
    entry = _stage_entry(state, key, stage)
    config_checksum = _canonical_hash(config)
    same_work = (
        entry.get("input_checksum") == input_checksum
        and entry.get("configuration_checksum") == config_checksum
        and entry.get("status") in {"Running", "Failed"}
    )
    if _is_verified_complete(entry, input_checksum, config_checksum):
        print(f"  {stage}: skipped (completed output verified)")
        return True, Path(entry["output_path"]), entry.get("row_count")

    output_dir = _candidate_output_dir(
        base_stage_root,
        identity_slug,
        input_checksum,
        config_checksum,
        entry,
        same_work,
    )
    entry.update(
        {
            "status": "Running",
            "input_checksum": input_checksum,
            "configuration_checksum": config_checksum,
            "configuration": config,
            "output_dir": str(output_dir),
            "started_at": _now(),
            "updated_at": _now(),
            "error": None,
            "retryable": True,
        }
    )
    _save_state(state_path, state)
    try:
        output_path, stage_rows = execute(output_dir)
        output_path = Path(output_path).resolve()
        row_count = (
            _validate_csv_output(output_path, expected_rows)
            if output_path.suffix.lower() == ".csv"
            else (stage_rows if isinstance(stage_rows, int) else None)
        )
        output_checksum = sha256_file(output_path)
        entry.update(
            {
                "status": "Completed",
                "output_path": str(output_path),
                "output_checksum": output_checksum,
                "row_count": row_count if row_count is not None else stage_rows,
                "completed_at": _now(),
                "updated_at": _now(),
                "error": None,
                "retryable": False,
            }
        )
        _save_state(state_path, state)
        print(f"  {stage}: completed ({output_path})")
        return True, output_path, entry.get("row_count")
    except Exception as exc:
        error = {
            "patient_id": state["recordings"][key]["patient_id"],
            "recording_id": state["recordings"][key]["recording_id"],
            "relative_path": key.split("::", 1)[1],
            "stage": stage,
            "message": f"{type(exc).__name__}: {exc}",
            "timestamp": _now(),
            "retryable": True,
        }
        entry.update(
            {
                "status": "Failed",
                "error": error["message"],
                "failed_at": error["timestamp"],
                "updated_at": error["timestamp"],
                "retryable": True,
            }
        )
        state["errors"].append(error)
        _save_state(state_path, state)
        print(f"  {stage}: FAILED ({error['message']})", file=sys.stderr)
        return False, None, None


def process_corpus(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    patients: Iterable[str] | None = None,
    recordings: Iterable[str] | None = None,
    dry_run: bool = False,
    configuration: PipelineConfiguration = PipelineConfiguration(),
) -> dict[str, Any]:
    """Run conversion, segmentation, and features in order for selected EDFs."""
    configuration.validate()
    manifest_path = manifest_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    rows = _selected_rows(
        _read_manifest(manifest_path),
        {item.casefold() for item in patients} if patients else None,
        {Path(item).as_posix().casefold() for item in recordings} if recordings else None,
    )
    if not rows:
        raise ValueError("No manifest recordings match the requested selection.")

    print(f"Selected recordings: {len(rows)}")
    for row in rows:
        print(
            f"  {row['patient_id']} | {row['recording_id']} | "
            f"{row['relative_path']} | {row['download_status']} | "
            f"{row['seizure_status']}"
        )
    if dry_run:
        return {"selected": len(rows), "completed": 0, "failed": 0, "dry_run": True}

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / STATE_FILENAME
    state = _load_state(state_path)
    total_completed = 0
    total_failed = 0

    for index, row in enumerate(rows, start=1):
        patient_id = row["patient_id"].strip()
        relative_path = Path(row["relative_path"]).as_posix()
        recording_id = row["recording_id"].strip()
        key = _record_key(patient_id, relative_path)
        slug = _identity_slug(relative_path)
        print(
            f"[{index}/{len(rows)}] {patient_id}/{recording_id} "
            f"({relative_path})"
        )
        _initialize_recording(
            state_path, state, key, patient_id, recording_id, relative_path
        )
        try:
            source_path = (dataset_root / relative_path).resolve()
            source_path.relative_to(dataset_root)
        except ValueError:
            source_path = Path("/__invalid_manifest_path__")
        row_status = row["download_status"].strip()
        if row_status != "downloaded_validated" or not source_path.is_file():
            reason = (
                f"manifest download_status={row_status}"
                if row_status != "downloaded_validated"
                else "EDF file is missing at its manifest relative path"
            )
            for stage in STAGES:
                _skip_stage(state_path, state, key, stage, reason)
            continue

        try:
            edf_checksum = sha256_file(source_path)
        except OSError as exc:
            for stage in STAGES:
                _skip_stage(state_path, state, key, stage, f"cannot read EDF: {exc}")
            continue

        patient_root = output_root / patient_id
        annotation_known = row["seizure_status"] in {
            "seizures_annotated",
            "no_seizures_annotated",
        } and row.get("annotation_agreement", "not_available") != "disagreement"
        annotation_input = _canonical_hash(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "relative_path": relative_path,
                "seizure_status": row["seizure_status"],
                "seizure_intervals": row["seizure_intervals"],
                "annotation_agreement": row.get("annotation_agreement", ""),
            }
        )
        conversion_ok, csv_path, _ = _execute_stage(
            state_path=state_path,
            state=state,
            key=key,
            stage="conversion",
            input_checksum=edf_checksum,
            config={"chunk_seconds": configuration.conversion_chunk_seconds},
            base_stage_root=patient_root / "csv",
            identity_slug=slug,
            execute=lambda output_dir: (
                (
                    (report := edf_to_csv.convert_edf_to_csv(
                        source_path,
                        output_dir,
                        chunk_seconds=configuration.conversion_chunk_seconds,
                    )).output_path,
                    report.exported_rows,
                )
            ),
        )
        if conversion_ok:
            total_completed += 1
        else:
            total_failed += 1

        if not annotation_known:
            reason = (
                f"annotation status is {row['seizure_status']!r} "
                f"(agreement={row.get('annotation_agreement', 'not_available')!r})"
            )
            _skip_stage(state_path, state, key, "segmentation", reason)
            _skip_stage(state_path, state, key, "feature_extraction", "segmentation was skipped")
            continue
        if not conversion_ok or csv_path is None:
            _skip_stage(state_path, state, key, "segmentation", "conversion did not complete")
            _skip_stage(state_path, state, key, "feature_extraction", "segmentation did not complete")
            continue

        segmentation_input = _canonical_hash(
            {"edf_sha256": edf_checksum, "annotation_sha256": annotation_input}
        )
        window_config = {
            "output_schema_version": 2,
            "window_seconds": configuration.window_seconds,
            "overlap_fraction": configuration.overlap_fraction,
            "label_policy": configuration.label_policy,
        }

        def segment_recording(output_dir: Path) -> tuple[Path, int]:
            output_dir.mkdir(parents=True, exist_ok=True)
            window_path = output_dir / "windows.csv"
            window_count, message = segment_windows.create_recording_manifest(
                edf_path=source_path,
                data_dir=dataset_root,
                annotation_source=manifest_path,
                output_path=window_path,
                patient_id=patient_id,
                configuration=segment_windows.WindowConfiguration(
                    configuration.window_seconds,
                    configuration.overlap_fraction,
                    configuration.label_policy,
                ),
            )
            print(f"    {message}")
            return window_path, window_count

        segmentation_ok, window_output_path, window_rows = _execute_stage(
            state_path=state_path,
            state=state,
            key=key,
            stage="segmentation",
            input_checksum=segmentation_input,
            config=window_config,
            base_stage_root=patient_root / "windows",
            identity_slug=slug,
            execute=segment_recording,
        )
        if segmentation_ok:
            total_completed += 1
        else:
            total_failed += 1
        if not segmentation_ok or window_output_path is None:
            _skip_stage(state_path, state, key, "feature_extraction", "segmentation did not complete")
            continue
        if not window_rows:
            _skip_stage(state_path, state, key, "feature_extraction", "segmentation produced no windows")
            continue

        feature_config = extract_features.FeatureConfiguration(
            feature_families=configuration.feature_families,
            wavelet=configuration.wavelet,
            wavelet_mode=configuration.wavelet_mode,
        )
        feature_input = _canonical_hash(
            {
                "csv_sha256": sha256_file(csv_path),
                "windows_sha256": sha256_file(window_output_path),
            }
        )
        feature_configuration = {
            **json.loads(feature_config.to_json()),
            "chunk_rows": configuration.feature_chunk_rows,
            "output_schema_version": 2,
        }
        feature_ok, _, _ = _execute_stage(
            state_path=state_path,
            state=state,
            key=key,
            stage="feature_extraction",
            input_checksum=feature_input,
            config=feature_configuration,
            base_stage_root=patient_root / "features",
            identity_slug=slug,
            expected_rows=window_rows,
            execute=lambda output_dir: extract_features.extract_feature_matrix(
                manifest_path=window_output_path,
                csv_dir=csv_path.parent,
                output_dir=output_dir,
                configuration=feature_config,
                chunk_rows=configuration.feature_chunk_rows,
            )[:2],
        )
        if feature_ok:
            total_completed += 1
        else:
            total_failed += 1

    return {
        "selected": len(rows),
        "completed": total_completed,
        "failed": total_failed,
        "dry_run": False,
        "state_path": str(state_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--patient",
        action="append",
        help="Patient ID to include; repeat to select multiple (default: all).",
    )
    parser.add_argument(
        "--recording",
        action="append",
        help="Recording ID, EDF basename, or relative path; repeat to select multiple.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=edf_to_csv.DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--window-seconds", type=float, default=segment_windows.DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=segment_windows.DEFAULT_OVERLAP_FRACTION)
    parser.add_argument("--label-policy", choices=segment_windows.POLICIES, default=segment_windows.DEFAULT_POLICY)
    parser.add_argument("--features", nargs="+", choices=extract_features.FEATURE_FAMILIES, default=list(extract_features.FEATURE_FAMILIES))
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-mode", default="symmetric")
    parser.add_argument("--feature-chunk-rows", type=int, default=extract_features.DEFAULT_CHUNK_ROWS)
    args = parser.parse_args()
    configuration = PipelineConfiguration(
        conversion_chunk_seconds=args.chunk_seconds,
        window_seconds=args.window_seconds,
        overlap_fraction=args.overlap,
        label_policy=args.label_policy,
        feature_families=tuple(args.features),
        wavelet=args.wavelet,
        wavelet_mode=args.wavelet_mode,
        feature_chunk_rows=args.feature_chunk_rows,
    )
    try:
        result = process_corpus(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            patients=args.patient,
            recordings=args.recording,
            dry_run=args.dry_run,
            configuration=configuration,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"CHB-MIT processing failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"Selected={result['selected']}, completed stages={result['completed']}, "
        f"failed stages={result['failed']}"
    )
    if "state_path" in result:
        print(f"State: {result['state_path']}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
