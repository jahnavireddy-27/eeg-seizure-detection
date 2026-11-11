#!/usr/bin/env python3
"""Assemble verified per-recording feature outputs into a corpus-wide matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote, unquote

import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_MANIFEST = PROJECT_ROOT / "results" / "chbmit_dataset_manifest.csv"
DEFAULT_STATE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "corpus_pipeline"
    / ".process_chbmit_state.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "assembled_features"
ASSEMBLED_FILENAME = "chbmit_feature_dataset.csv"
RECORDINGS_FILENAME = "recordings_manifest.csv"
SCHEMA_FILENAME = "feature_schema.json"
HASH_CHUNK_BYTES = 1024 * 1024
LEGACY_FEATURE_PATTERN = re.compile(r"^channel_(\d+)__(.*?)__(.+)$")
ALIGNED_FEATURE_PATTERN = re.compile(
    r"^channel__(.*?)__instance_(\d+)__(.+)$"
)
ASSEMBLED_METADATA_COLUMNS = (
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


def _identity(row: dict[str, str]) -> tuple[str, str, str]:
    patient = row.get("patient_id", "").strip()
    recording = row.get("recording_id", "").strip()
    relative = Path(row.get("relative_path", "")).as_posix()
    if not patient or not recording or not relative:
        raise ValueError(f"Incomplete composite recording identity: {row}")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Recording path escapes the dataset root: {relative!r}")
    return patient, recording, relative


def _identity_key(identity: tuple[str, str, str]) -> str:
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
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
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_csv_writer(path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return stream, Path(stream.name), csv.DictWriter(stream, fieldnames=fieldnames)


def _publish_temp(stream: Any, temporary_path: Path, destination: Path) -> None:
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()
    os.replace(temporary_path, destination)


def _read_csv_rows(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    stream = path.open(newline="", encoding="utf-8")
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        stream.close()
        raise ValueError(f"CSV has no header: {path}")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        stream.close()
        raise ValueError(f"CSV contains duplicate column names: {path}")

    def rows() -> Iterator[dict[str, str]]:
        try:
            yield from reader
        finally:
            stream.close()

    return list(reader.fieldnames), rows()


def _read_inventory(path: Path) -> list[dict[str, str]]:
    required = {"patient_id", "recording_id", "relative_path"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Corpus manifest is missing fields: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        identity = _identity(row)
        if identity in seen:
            raise ValueError(f"Duplicate corpus recording identity: {identity}")
        seen.add(identity)
        if row.get("edf_filename") and Path(identity[2]).name != row["edf_filename"]:
            raise ValueError(
                f"Manifest EDF filename disagrees with relative path: {identity}"
            )
    return sorted(rows, key=lambda row: tuple(part.casefold() for part in _identity(row)))


def _read_processing_state(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    recordings = state.get("recordings")
    if not isinstance(recordings, dict):
        raise ValueError(f"Processing state has no recordings object: {path}")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for state_key, row in recordings.items():
        if not isinstance(row, dict):
            continue
        record = {
            "patient_id": row.get("patient_id", ""),
            "recording_id": row.get("recording_id", ""),
            "relative_path": row.get("relative_path", ""),
        }
        try:
            identity = _identity(record)
        except ValueError:
            if "::" not in state_key:
                continue
            patient, relative = state_key.split("::", 1)
            identity = (patient, Path(relative).stem, Path(relative).as_posix())
        if identity in result:
            raise ValueError(f"Duplicate recording identity in processing state: {identity}")
        result[identity] = row
    return result


def _normalize_channel_label(label: str) -> str:
    return " ".join(label.strip().split()).upper()


def _feature_mapping(
    feature_columns: list[str],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    parsed_legacy: list[tuple[str, int, str, str]] = []
    aligned: dict[str, str] = {}
    aligned_labels: dict[str, set[str]] = {}
    for column in feature_columns:
        match = ALIGNED_FEATURE_PATTERN.fullmatch(column)
        if match:
            normalized = unquote(match.group(1))
            occurrence = int(match.group(2))
            feature_name = match.group(3)
            normalized = _normalize_channel_label(normalized)
            target = (
                f"channel__{quote(normalized, safe='')}__"
                f"instance_{occurrence:02d}__{feature_name}"
            )
            aligned[column] = target
            canonical_channel = target.rsplit("__", 1)[0]
            aligned_labels.setdefault(canonical_channel, set()).add(normalized)
            continue
        legacy = LEGACY_FEATURE_PATTERN.fullmatch(column)
        if not legacy:
            raise ValueError(
                f"Unrecognized feature column {column!r}; expected a channel feature."
            )
        index = int(legacy.group(1))
        original_label = unquote(legacy.group(2))
        feature_name = legacy.group(3)
        parsed_legacy.append((column, index, original_label, feature_name))

    if parsed_legacy and aligned:
        raise ValueError("A feature file mixes legacy and channel-aligned feature names.")
    if aligned:
        return aligned, aligned_labels
    labels_by_index: dict[int, str] = {}
    for _, index, label, _ in parsed_legacy:
        prior = labels_by_index.setdefault(index, label)
        if prior != label:
            raise ValueError(f"Channel index {index} maps to multiple labels.")
    occurrence_by_index: dict[int, int] = {}
    counts: dict[str, int] = {}
    for index in sorted(labels_by_index):
        normalized = _normalize_channel_label(labels_by_index[index])
        counts[normalized] = counts.get(normalized, 0) + 1
        occurrence_by_index[index] = counts[normalized]
    mapping = {}
    labels_by_canonical: dict[str, set[str]] = {}
    for column, index, original, feature_name in parsed_legacy:
        normalized = _normalize_channel_label(original)
        canonical_channel = (
            f"channel__{quote(normalized, safe='')}__"
            f"instance_{occurrence_by_index[index]:02d}"
        )
        mapping[column] = f"{canonical_channel}__{feature_name}"
        labels_by_canonical.setdefault(canonical_channel, set()).add(original)
    return mapping, labels_by_canonical


def _configured_feature_names(configuration: dict[str, Any]) -> set[str]:
    try:
        bands = tuple(
            (
                str(band["name"]),
                float(band["low"]),
                float(band["high_exclusive"]),
            )
            for band in configuration["bands_hz_half_open"]
        )
        feature_configuration = extract_features.FeatureConfiguration(
            wavelet=str(configuration["wavelet"]),
            wavelet_mode=str(configuration["wavelet_mode"]),
            bands=bands,
            feature_families=tuple(configuration["feature_families"]),
        )
        feature_configuration.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid serialized feature configuration: {exc}") from exc
    return set(extract_features._feature_value_names(feature_configuration))


def _recording_channel_details(
    feature_columns: list[str], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    channels: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        legacy = LEGACY_FEATURE_PATTERN.fullmatch(column)
        if legacy:
            index = int(legacy.group(1))
            original = unquote(legacy.group(2))
            target = mapping[column].rsplit("__", 1)[0]
            channels.setdefault(
                str(index),
                {
                    "channel_index": index,
                    "original_label": original,
                    "normalized_label": _normalize_channel_label(original),
                    "canonical_channel_id": target,
                },
            )
        else:
            aligned = ALIGNED_FEATURE_PATTERN.fullmatch(column)
            if not aligned:
                continue
            normalized = _normalize_channel_label(unquote(aligned.group(1)))
            occurrence = int(aligned.group(2))
            canonical = (
                f"channel__{quote(normalized, safe='')}__instance_{occurrence:02d}"
            )
            channels.setdefault(
                canonical,
                {
                    "original_label": normalized,
                    "normalized_label": normalized,
                    "instance": occurrence,
                    "canonical_channel_id": canonical,
                },
            )
    return list(channels.values())


def _validate_feature_file(
    path: Path,
    expected_identity: tuple[str, str, str],
    expected_rows: int | None,
    expected_annotation_status: str | None,
) -> tuple[list[str], dict[str, str], dict[str, set[str]], dict[str, Any], int]:
    headers, rows = _read_csv_rows(path)
    required_metadata = {
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
    }
    missing = required_metadata - set(headers)
    if missing:
        raise ValueError(f"Feature file is missing metadata fields: {sorted(missing)}")
    feature_columns = [
        column for column in headers if column not in extract_features.METADATA_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("Feature file contains no signal feature columns.")
    mapping, channel_labels = _feature_mapping(feature_columns)
    config_signature: str | None = None
    config_document: dict[str, Any] | None = None
    row_count = 0
    previous_start = -1.0
    previous_end: float | None = None
    constant_metadata: tuple[str, float, int] | None = None
    for row in rows:
        row_identity = _identity(row)
        if row_identity != expected_identity:
            raise ValueError(
                f"Feature row identity {row_identity} does not match expected "
                f"{expected_identity}."
            )
        if (
            Path(row["relative_path"]).name != row["recording_filename"]
            or Path(row["recording_filename"]).stem != row["recording_id"]
        ):
            raise ValueError("Feature row filename disagrees with composite identity.")
        if (
            expected_annotation_status is not None
            and row["annotation_status"] != expected_annotation_status
        ):
            raise ValueError(
                f"Feature annotation status {row['annotation_status']!r} does not match "
                f"corpus manifest status {expected_annotation_status!r}."
            )
        try:
            window_index = int(row["window_index"])
            start = float(row["start_time_seconds"])
            end = float(row["end_time_seconds"])
            sampling_frequency_hz = float(row["sampling_frequency_hz"])
            window_samples = int(row["window_samples"])
            config = json.loads(row["feature_config_json"])
            if not isinstance(config, dict):
                raise ValueError("feature_config_json must encode an object")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid feature metadata at row {row_count + 2}: {exc}") from exc
        if (
            window_index != row_count
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_start
            or end <= start
            or (row_count > 0 and start <= previous_start)
            or not math.isfinite(sampling_frequency_hz)
            or sampling_frequency_hz <= 0
            or window_samples <= 0
            or row["seizure_label"] not in {"seizure", "non_seizure", "unknown"}
        ):
            raise ValueError(f"Invalid or incomplete feature row {row_count + 2}.")
        signature = _canonical_json(config)
        if config_signature is not None and signature != config_signature:
            raise ValueError("Feature configuration changes within one recording.")
        config_signature = signature
        config_document = config
        row_constant_metadata = (
            row["label_policy"],
            sampling_frequency_hz,
            window_samples,
        )
        if constant_metadata is not None and row_constant_metadata != constant_metadata:
            raise ValueError("Label policy or sample geometry changes within recording.")
        constant_metadata = row_constant_metadata
        if previous_end is not None and not math.isclose(
            end - start,
            previous_end - previous_start,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("Window duration changes within recording.")
        for column in feature_columns:
            value = row.get(column, "")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric feature {column!r} in row {row_count + 2}."
                ) from exc
            if math.isinf(number):
                raise ValueError(
                    f"Infinite feature {column!r} in row {row_count + 2}."
                )
        row_count += 1
        previous_start = start
        previous_end = end
    if row_count == 0:
        raise ValueError("Feature file contains no window rows.")
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            f"Feature file has {row_count} rows; processing state records {expected_rows}."
        )
    expected_names = _configured_feature_names(config_document or {})
    actual_names_by_channel: dict[str, set[str]] = {}
    for source_column, target_column in mapping.items():
        channel_id, feature_name = target_column.rsplit("__", 1)
        actual_names_by_channel.setdefault(channel_id, set()).add(feature_name)
    if not actual_names_by_channel or any(
        names != expected_names for names in actual_names_by_channel.values()
    ):
        raise ValueError(
            "Feature columns do not match the serialized configuration for every channel."
        )
    return headers, mapping, channel_labels, json.loads(config_signature or "{}"), row_count


def assemble_feature_dataset(
    *,
    corpus_manifest: Path,
    state_path: Path,
    output_dir: Path,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    """Assemble verified feature files using bounded, per-recording CSV reads."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")
    corpus_manifest = corpus_manifest.expanduser().resolve()
    state_path = state_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    inventory = _read_inventory(corpus_manifest)
    state_recordings = _read_processing_state(state_path)
    staged: list[dict[str, Any]] = []
    referenced_paths: dict[Path, list[tuple[str, str, str]]] = {}
    inventory_identities = {_identity(row) for row in inventory}
    for row in inventory:
        identity = _identity(row)
        processing = state_recordings.get(identity)
        if processing is None:
            staged.append(
                {"identity": identity, "row": row, "status": "excluded",
                 "reason": "no processing state entry"}
            )
            continue
        stage = processing.get("stages", {}).get("feature_extraction", {})
        if stage.get("status") != "Completed":
            staged.append(
                {"identity": identity, "row": row, "status": "excluded",
                 "reason": f"feature extraction status is {stage.get('status', 'Pending')}"}
            )
            continue
        feature_path = Path(stage.get("output_path", "")).expanduser()
        if not feature_path.is_file():
            staged.append(
                {"identity": identity, "row": row, "status": "excluded",
                 "reason": "completed feature output is missing"}
            )
            continue
        try:
            actual_checksum = sha256_file(feature_path)
        except OSError as exc:
            staged.append(
                {"identity": identity, "row": row, "status": "excluded",
                 "reason": f"cannot read feature output: {exc}"}
            )
            continue
        if actual_checksum != stage.get("output_checksum"):
            staged.append(
                {"identity": identity, "row": row, "status": "excluded",
                 "reason": "feature output checksum does not match processing state"}
            )
            continue
        referenced_paths.setdefault(feature_path.resolve(), []).append(identity)
        staged.append(
            {
                "identity": identity,
                "row": row,
                "stage": stage,
                "feature_path": feature_path.resolve(),
                "status": "candidate",
                "reason": "",
            }
        )
    for identity in sorted(
        set(state_recordings) - inventory_identities,
        key=lambda item: tuple(part.casefold() for part in item),
    ):
        staged.append(
            {
                "identity": identity,
                "row": {
                    "edf_filename": Path(identity[2]).name,
                    "seizure_status": "unknown",
                },
                "status": "excluded",
                "reason": "processing state identity is absent from corpus manifest",
            }
        )

    duplicate_paths = {
        path
        for path, identities in referenced_paths.items()
        if len(identities) > 1
    }
    for item in staged:
        if item.get("feature_path") in duplicate_paths:
            item["status"] = "excluded"
            item["reason"] = "same feature output path is referenced by multiple identities"

    included: list[dict[str, Any]] = []
    reference_configuration: str | None = None
    for item in staged:
        if item["status"] != "candidate":
            continue
        try:
            corpus_status = item["row"].get("seizure_status", "")
            expected_annotation = {
                "seizures_annotated": "confirmed_seizure",
                "no_seizures_annotated": "confirmed_no_seizure",
            }.get(corpus_status)
            headers, mapping, channel_labels, config, rows_count = _validate_feature_file(
                item["feature_path"],
                item["identity"],
                item["stage"].get("row_count"),
                expected_annotation,
            )
            signature = _canonical_json(config)
            if reference_configuration is not None and signature != reference_configuration:
                raise ValueError("feature configuration differs from other recordings")
            reference_configuration = signature
            item.update(
                {
                    "status": "included",
                    "headers": headers,
                    "feature_mapping": mapping,
                    "channel_labels": channel_labels,
                    "feature_configuration": config,
                    "row_count": rows_count,
                    "feature_columns": [
                        column
                        for column in headers
                        if column not in extract_features.METADATA_COLUMNS
                    ],
                }
            )
            item["recording_channels"] = _recording_channel_details(
                item["feature_columns"], mapping
            )
            included.append(item)
        except (OSError, ValueError, csv.Error) as exc:
            item["status"] = "excluded"
            item["reason"] = str(exc)

    predictor_columns = sorted(
        {
            target
            for item in included
            for target in item["feature_mapping"].values()
        }
    )
    fieldnames = [*ASSEMBLED_METADATA_COLUMNS, *predictor_columns]
    all_channel_labels: dict[str, set[str]] = {}
    for item in included:
        for column, labels in item["channel_labels"].items():
            all_channel_labels.setdefault(column, set()).update(labels)
        for source_column, target_column in item["feature_mapping"].items():
            if target_column.startswith("channel__") and target_column not in predictor_columns:
                raise RuntimeError(f"Feature schema omitted mapped column {target_column}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / ASSEMBLED_FILENAME
    dataset_stream, dataset_temp, dataset_writer = _atomic_csv_writer(
        dataset_path, fieldnames
    )
    dataset_writer.writeheader()
    try:
        for item in included:
            path = item["feature_path"]
            mapping = item["feature_mapping"]
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                while True:
                    batch = list(itertools.islice(reader, chunk_rows))
                    if not batch:
                        break
                    for row in batch:
                        assembled = {
                            key: row.get(key, "")
                            for key in ASSEMBLED_METADATA_COLUMNS
                        }
                        assembled.update({column: "NaN" for column in predictor_columns})
                        for source_column, target_column in mapping.items():
                            assembled[target_column] = row[source_column]
                        dataset_writer.writerow(assembled)
        _publish_temp(dataset_stream, dataset_temp, dataset_path)
    except Exception:
        dataset_stream.close()
        dataset_temp.unlink(missing_ok=True)
        raise

    manifest_path = output_dir / RECORDINGS_FILENAME
    manifest_fields = [
        "patient_id",
        "recording_id",
        "relative_path",
        "edf_filename",
        "annotation_status",
        "status",
        "reason",
        "feature_path",
        "feature_sha256",
        "window_rows",
    ]
    manifest_stream, manifest_temp, manifest_writer = _atomic_csv_writer(
        manifest_path, manifest_fields
    )
    manifest_writer.writeheader()
    try:
        for item in staged:
            row = item["row"]
            stage = item.get("stage", {})
            manifest_writer.writerow(
                {
                    "patient_id": item["identity"][0],
                    "recording_id": item["identity"][1],
                    "relative_path": item["identity"][2],
                    "edf_filename": row.get("edf_filename", ""),
                    "annotation_status": row.get(
                        "seizure_status", row.get("annotation_status", "unknown")
                    ),
                    "status": item["status"],
                    "reason": item["reason"],
                    "feature_path": str(item.get("feature_path", "")),
                    "feature_sha256": stage.get("output_checksum", ""),
                    "window_rows": item.get("row_count", ""),
                }
            )
        _publish_temp(manifest_stream, manifest_temp, manifest_path)
    except Exception:
        manifest_stream.close()
        manifest_temp.unlink(missing_ok=True)
        raise

    schema = {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "recording_identity": ["patient_id", "recording_id", "relative_path"],
        "metadata_columns": list(ASSEMBLED_METADATA_COLUMNS),
        "target_column": "seizure_label",
        "annotation_status_column": "annotation_status",
        "predictor_columns": predictor_columns,
        "missing_feature_value": "NaN",
        "no_imputation_or_transformation": True,
        "channel_matching": {
            "normalization": "uppercase; trim and collapse whitespace only",
            "bipolar_derivations": "direction and electrode order are preserved",
            "duplicate_labels": (
                "same-label channels are distinguished by occurrence order "
                "within each recording"
            ),
            "channels": [
                {
                    "channel_id": key,
                    "original_labels": sorted(labels),
                }
                for key, labels in sorted(all_channel_labels.items())
            ],
        },
        "recordings": {
            _identity_key(item["identity"]): item["recording_channels"]
            for item in included
        },
        "unavailable_channels": {
            _identity_key(item["identity"]): sorted(
                set(all_channel_labels)
                - {
                    channel["canonical_channel_id"]
                    for channel in item["recording_channels"]
                }
            )
            for item in included
        },
        "feature_configuration": (
            included[0]["feature_configuration"] if included else None
        ),
        "included_recordings": len(included),
        "excluded_recordings": len(staged) - len(included),
        "recordings_manifest": str(manifest_path),
        "imputation_warning": (
            "Any learned imputation or transformation in later evaluation must be "
            "fit only on training patients within each cross-validation fold."
        ),
    }
    _atomic_write_json(output_dir / SCHEMA_FILENAME, schema)
    return {
        "included_recordings": len(included),
        "excluded_recordings": len(staged) - len(included),
        "windows": sum(item["row_count"] for item in included),
        "predictors": len(predictor_columns),
        "dataset_path": str(dataset_path),
        "recordings_manifest": str(manifest_path),
        "schema_path": str(output_dir / SCHEMA_FILENAME),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    try:
        result = assemble_feature_dataset(
            corpus_manifest=args.corpus_manifest,
            state_path=args.state,
            output_dir=args.output_dir,
            chunk_rows=args.chunk_rows,
        )
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        print(f"Feature dataset assembly failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Included recordings={result['included_recordings']}, "
        f"excluded recordings={result['excluded_recordings']}, "
        f"windows={result['windows']}, predictors={result['predictors']}"
    )
    print(f"Dataset: {result['dataset_path']}")
    print(f"Recordings manifest: {result['recordings_manifest']}")
    print(f"Schema: {result['schema_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
