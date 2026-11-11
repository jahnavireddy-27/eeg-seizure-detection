#!/usr/bin/env python3
"""Evaluate the existing Random Forest on the expanded CHB-MIT feature set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import train_random_forest as baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_MATRIX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "assembled_features"
    / "chbmit_feature_dataset.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "expanded_random_forest_baseline"
)
DEFAULT_SPLITS = 4
DEFAULT_FOLD_RANDOM_STATE = 42
IDENTITY_COLUMNS = ("patient_id", "recording_id", "relative_path")


@dataclass(frozen=True)
class RecordingFold:
    name: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_recordings: tuple[tuple[str, str, str], ...]
    test_recordings: tuple[tuple[str, str, str], ...]


def _recording_identity(row: dict[str, Any] | pd.Series) -> tuple[str, str, str]:
    patient = str(row["patient_id"]).strip()
    recording = str(row["recording_id"]).strip()
    relative_path = Path(str(row["relative_path"])).as_posix()
    if not patient or not recording or not relative_path or relative_path == ".":
        raise ValueError(f"Incomplete recording identity: {row}")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Recording path escapes dataset root: {relative_path!r}")
    return patient, recording, relative_path


def make_recording_grouped_folds(
    frame: pd.DataFrame,
    n_splits: int = DEFAULT_SPLITS,
    random_state: int = DEFAULT_FOLD_RANDOM_STATE,
) -> tuple[RecordingFold, ...]:
    """Stratify recording-level seizure presence, then assign whole recordings."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    missing = (set(IDENTITY_COLUMNS) | {"seizure_label"}) - set(frame.columns)
    if missing:
        raise ValueError("Feature matrix lacks columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Feature matrix contains no rows.")
    labels = frame["seizure_label"].map(baseline.LABEL_TO_INT)
    if labels.isna().any():
        raise ValueError("Recording folds require only verified binary seizure labels.")
    identities = [_recording_identity(row) for _, row in frame.iterrows()]
    labels_array = labels.to_numpy(dtype=np.int8)

    group_rows: dict[tuple[str, str, str], list[int]] = {}
    for index, identity in enumerate(identities):
        group_rows.setdefault(identity, []).append(index)
    recordings = sorted(group_rows, key=lambda item: tuple(part.casefold() for part in item))
    recording_seizure = np.asarray(
        [int(labels_array[group_rows[identity]].max() == 1) for identity in recordings],
        dtype=np.int8,
    )
    recording_counts = np.bincount(recording_seizure, minlength=2)
    if np.any(recording_counts < n_splits):
        raise ValueError(
            f"Cannot create {n_splits} folds with both recording classes in every "
            f"test fold: seizure_recordings={recording_counts[1]}, "
            f"non_seizure_recordings={recording_counts[0]}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    folds: list[RecordingFold] = []
    test_appearances: Counter[tuple[str, str, str]] = Counter()
    for fold_index, (train_group_indices, test_group_indices) in enumerate(
        splitter.split(np.zeros(len(recordings)), recording_seizure), start=1
    ):
        train_recordings = tuple(recordings[index] for index in train_group_indices)
        test_recordings = tuple(recordings[index] for index in test_group_indices)
        test_appearances.update(test_recordings)
        train_groups = set(train_recordings)
        test_groups = set(test_recordings)
        if train_groups & test_groups:
            raise RuntimeError(f"Fold {fold_index} splits a recording across partitions.")
        train_indices = np.asarray(
            [row for identity in train_recordings for row in group_rows[identity]],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [row for identity in test_recordings for row in group_rows[identity]],
            dtype=np.int64,
        )
        if set(labels_array[train_indices]) != {0, 1}:
            raise ValueError(f"Fold {fold_index} training partition lacks a class.")
        if set(labels_array[test_indices]) != {0, 1}:
            raise ValueError(f"Fold {fold_index} test partition lacks a class.")
        folds.append(
            RecordingFold(
                name=f"fold_{fold_index}",
                train_indices=train_indices,
                test_indices=test_indices,
                train_recordings=train_recordings,
                test_recordings=test_recordings,
            )
        )

    if set(test_appearances) != set(recordings) or any(
        appearances != 1 for appearances in test_appearances.values()
    ):
        raise RuntimeError("Each recording must appear in exactly one test fold.")
    return tuple(folds)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int8), minlength=2)
    return {"non_seizure": int(counts[0]), "seizure": int(counts[1])}


def _diagnostic_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    """Return metrics for groups that may contain only one true class."""
    if set(np.asarray(y_true, dtype=np.int8).tolist()) == {0, 1}:
        return baseline.calculate_metrics(y_true, y_pred, probability)
    matrix = baseline.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    ratio = baseline._ratio_or_none
    precision = ratio(tp, tp + fp)
    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    actual_classes = set(np.asarray(y_true).tolist())
    return {
        "confusion_matrix": {
            "labels": ["non_seizure", "seizure"],
            "values": matrix.tolist(),
        },
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": float(baseline.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(baseline.balanced_accuracy_score(y_true, y_pred))
            if len(actual_classes) == 2
            else None
        ),
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": None,
        "average_precision": None,
        "undefined_metrics": [
            key
            for key, value in (
                ("precision", precision),
                ("recall_sensitivity", sensitivity),
                ("specificity", specificity),
                ("f1_score", f1),
                ("roc_auc", None),
                ("average_precision", None),
            )
            if value is None
        ],
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"Refusing to write an empty report: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames or list(rows[0]),
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"), allow_nan=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def run_expanded_baseline(
    feature_matrix_path: Path = DEFAULT_FEATURE_MATRIX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    configuration: baseline.ForestConfiguration = baseline.ForestConfiguration(),
    n_splits: int = DEFAULT_SPLITS,
    fold_random_state: int = DEFAULT_FOLD_RANDOM_STATE,
) -> dict[str, Any]:
    """Run recording-held-out OOF evaluation without touching historical outputs."""
    configuration.validate()
    feature_matrix_path = Path(feature_matrix_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    frame, feature_columns = baseline.load_feature_matrix(feature_matrix_path)
    folds = make_recording_grouped_folds(frame, n_splits, fold_random_state)
    labels = frame["seizure_label"].map(baseline.LABEL_TO_INT).to_numpy(dtype=np.int8)
    identities = [_recording_identity(row) for _, row in frame.iterrows()]
    row_fold = np.full(len(frame), "", dtype=object)
    oof_prediction = np.full(len(frame), -1, dtype=np.int8)
    oof_probability = np.full(len(frame), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    recording_to_fold: dict[tuple[str, str, str], str] = {}

    for fold in folds:
        train_index, test_index = fold.train_indices, fold.test_indices
        if np.any(row_fold[test_index] != ""):
            raise RuntimeError(f"{fold.name} duplicates held-out window predictions.")
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_features = frame.iloc[train_index][feature_columns]
        test_features = frame.iloc[test_index][feature_columns]
        retained, excluded = baseline.deduplicate_training_features(train_features)
        model = baseline.build_random_forest(configuration)
        model.fit(train_features[retained], train_labels)
        if list(model.classes_) != [0, 1]:
            raise RuntimeError(f"{fold.name}: model classes are not [0, 1].")
        probabilities = model.predict_proba(test_features[retained])[:, 1]
        predictions = (
            probabilities >= configuration.probability_threshold
        ).astype(np.int8)
        metrics = baseline.calculate_metrics(test_labels, predictions, probabilities)
        row_fold[test_index] = fold.name
        oof_prediction[test_index] = predictions
        oof_probability[test_index] = probabilities

        train_rec_ids = set(fold.train_recordings)
        test_rec_ids = set(fold.test_recordings)
        if train_rec_ids & test_rec_ids:
            raise RuntimeError(f"{fold.name}: recording leakage detected.")
        if len(test_rec_ids) != len(fold.test_recordings):
            raise RuntimeError(f"{fold.name}: duplicate test recording assignment.")
        for identity in fold.test_recordings:
            recording_to_fold[identity] = fold.name
        train_recording_positive = sum(
            int(labels[[i for i, value in enumerate(identities) if value == rec]].max() == 1)
            for rec in fold.train_recordings
        )
        test_recording_positive = sum(
            int(labels[[i for i, value in enumerate(identities) if value == rec]].max() == 1)
            for rec in fold.test_recordings
        )
        train_count = _class_counts(train_labels)
        test_count = _class_counts(test_labels)
        fold_result = {
            "fold": fold.name,
            "train_recordings": json.dumps(
                [list(identity) for identity in fold.train_recordings], separators=(",", ":")
            ),
            "test_recordings": json.dumps(
                [list(identity) for identity in fold.test_recordings], separators=(",", ":")
            ),
            "test_recording_count": len(fold.test_recordings),
            "test_seizure_recordings": test_recording_positive,
            "test_non_seizure_recordings": len(fold.test_recordings) - test_recording_positive,
            "train_recording_count": len(fold.train_recordings),
            "train_seizure_recordings": train_recording_positive,
            "train_non_seizure_recordings": len(fold.train_recordings) - train_recording_positive,
            "train_windows": len(train_index),
            "train_non_seizure": train_count["non_seizure"],
            "train_seizure": train_count["seizure"],
            "test_windows": len(test_index),
            "test_non_seizure": test_count["non_seizure"],
            "test_seizure": test_count["seizure"],
            "predictor_count_before_deduplication": len(feature_columns),
            "predictor_count_after_deduplication": len(retained),
            "excluded_duplicate_count": len(excluded),
            "threshold": configuration.probability_threshold,
            **metrics,
        }
        fold_metrics.append(fold_result)
        fold_dir = output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        duplicate_rows = [
            {
                "excluded_feature": excluded_feature,
                "retained_feature": retained_feature,
                "detection_partition": "training_only",
            }
            for excluded_feature, retained_feature in excluded
        ]
        _write_csv(
            fold_dir / "excluded_duplicate_features.csv",
            duplicate_rows,
            ["excluded_feature", "retained_feature", "detection_partition"],
        )
        baseline._write_json(
            fold_dir / "metrics.json",
            {"random_forest": asdict(configuration), "results": fold_result},
        )
        baseline._plot_confusion_matrix(
            fold_dir / "confusion_matrix.png",
            metrics["confusion_matrix"]["values"],
            fold.name,
            "recording-grouped test recordings",
        )
        print(
            f"{fold.name}: test_recordings={len(fold.test_recordings)} "
            f"(seizure={test_recording_positive}, "
            f"non_seizure={len(fold.test_recordings) - test_recording_positive}), "
            f"windows={len(test_index)}, seizure_windows={test_count['seizure']}"
        )

        for identity in fold.test_recordings:
            recording_to_fold[identity] = fold.name

    if np.any(row_fold == "") or np.any(oof_prediction < 0) or not np.isfinite(oof_probability).all():
        raise RuntimeError("Out-of-fold predictions do not cover every window exactly once.")
    if len(recording_to_fold) != len(set(identities)):
        raise RuntimeError("Recording-to-fold assignment is not one-to-one.")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = frame[
        [
            "patient_id",
            "recording_id",
            "relative_path",
            "recording_filename",
            "window_index",
            "start_time_seconds",
            "end_time_seconds",
        ]
    ].copy()
    prediction_frame["test_fold"] = row_fold
    prediction_frame["y_true"] = labels
    prediction_frame["y_pred"] = oof_prediction
    prediction_frame["probability_non_seizure"] = 1.0 - oof_probability
    prediction_frame["probability_seizure"] = oof_probability
    prediction_frame.to_csv(
        output_dir / "out_of_fold_predictions.csv",
        index=False,
        float_format="%.17g",
    )

    assignment_rows = []
    for identity in sorted(set(identities), key=lambda item: tuple(x.casefold() for x in item)):
        indices = np.asarray([i for i, value in enumerate(identities) if value == identity])
        counts = _class_counts(labels[indices])
        assignment_rows.append(
            {
                "patient_id": identity[0],
                "recording_id": identity[1],
                "relative_path": identity[2],
                "test_fold": recording_to_fold[identity],
                "window_count": len(indices),
                "non_seizure_windows": counts["non_seizure"],
                "seizure_windows": counts["seizure"],
                "seizure_containing_recording": str(counts["seizure"] > 0).lower(),
            }
        )
    _write_csv(output_dir / "fold_assignments.csv", assignment_rows)

    pooled_metrics = baseline.calculate_metrics(labels, oof_prediction, oof_probability)
    pooled_document = {
        "partition": "all out-of-fold predictions",
        "windows": len(frame),
        "class_counts": _class_counts(labels),
        "recording_count": len(set(identities)),
        "seizure_recordings": sum(row["seizure_containing_recording"] == "true" for row in assignment_rows),
        "non_seizure_recordings": sum(row["seizure_containing_recording"] == "false" for row in assignment_rows),
        "threshold": configuration.probability_threshold,
        **pooled_metrics,
    }
    _write_json(output_dir / "pooled_metrics.json", pooled_document)
    baseline._plot_confusion_matrix(
        output_dir / "pooled_confusion_matrix.png",
        pooled_metrics["confusion_matrix"]["values"],
        "pooled",
        "all out-of-fold windows",
    )
    with (output_dir / "out_of_fold_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        prediction_rows = list(csv.DictReader(stream))
    if len(prediction_rows) != len(frame):
        raise RuntimeError("Saved out-of-fold prediction coverage failed validation.")
    expected_prediction_keys = {
        (*identity, str(frame.iloc[index]["window_index"]))
        for index, identity in enumerate(identities)
    }
    actual_prediction_keys = {
        (
            row["patient_id"],
            row["recording_id"],
            row["relative_path"],
            row["window_index"],
        )
        for row in prediction_rows
    }
    if (
        len(actual_prediction_keys) != len(prediction_rows)
        or actual_prediction_keys != expected_prediction_keys
    ):
        raise RuntimeError(
            "Saved out-of-fold predictions do not cover each composite "
            "recording/window identity exactly once."
        )

    diagnostic_groups: list[tuple[str, tuple[str, ...], str]] = []
    diagnostic_groups.extend(
        ("recording", identity, recording_to_fold[identity])
        for identity in sorted(set(identities), key=lambda item: tuple(x.casefold() for x in item))
    )
    diagnostic_rows: list[dict[str, Any]] = []
    for level, identity, assigned_fold in diagnostic_groups:
        mask = np.asarray([value == identity for value in identities], dtype=bool)
        diagnostic = _diagnostic_metrics(labels[mask], oof_prediction[mask], oof_probability[mask])
        patient_id, recording_id, relative_path = identity
        diagnostic_rows.append(
            {
                "level": level,
                "patient_id": patient_id,
                "recording_id": recording_id,
                "relative_path": relative_path,
                "test_fold": assigned_fold,
                "window_count": int(mask.sum()),
                "non_seizure_windows": int((labels[mask] == 0).sum()),
                "seizure_windows": int((labels[mask] == 1).sum()),
                **diagnostic,
            }
        )
    for patient_id in sorted({identity[0] for identity in identities}, key=str.casefold):
        mask = np.asarray([identity[0] == patient_id for identity in identities], dtype=bool)
        diagnostic = _diagnostic_metrics(labels[mask], oof_prediction[mask], oof_probability[mask])
        diagnostic_rows.append(
            {
                "level": "patient",
                "patient_id": patient_id,
                "recording_id": "",
                "relative_path": "",
                "test_fold": "multiple",
                "window_count": int(mask.sum()),
                "non_seizure_windows": int((labels[mask] == 0).sum()),
                "seizure_windows": int((labels[mask] == 1).sum()),
                **diagnostic,
            }
        )
    _write_csv(output_dir / "recording_patient_diagnostics.csv", diagnostic_rows)

    _write_csv(
        output_dir / "fold_metrics.csv",
        [
            {
                **result,
            }
            for result in fold_metrics
        ],
    )
    _write_json(
        output_dir / "run_configuration.json",
        {
            "random_forest": asdict(configuration),
            "validation": {
                "protocol": "four-fold recording-grouped stratified cross-validation",
                "stratification_unit": "recording-level seizure presence",
                "recording_identity": list(IDENTITY_COLUMNS),
                "n_splits": n_splits,
                "fold_random_state": fold_random_state,
                "test_recordings_per_fold": [
                    [list(identity) for identity in fold.test_recordings]
                    for fold in folds
                ],
                "patient_overlap_allowed": True,
                "patient_independent": False,
            },
            "predictor_selection": "channel-prefixed signal features only",
            "preprocessing": "none",
            "feature_selection": "none",
            "duplicate_detection": "exact equality on training partition per fold",
            "threshold_rule": "positive when P(seizure) >= 0.5 (unchanged baseline threshold)",
            "metrics": "fold-wise and pooled from all out-of-fold window predictions",
            "input_feature_matrix": str(feature_matrix_path),
        },
    )
    _write_report(output_dir, fold_metrics, pooled_document, assignment_rows)
    return {"folds": fold_metrics, "pooled": pooled_document, "output_dir": output_dir}


def _write_report(
    output_dir: Path,
    fold_metrics: list[dict[str, Any]],
    pooled: dict[str, Any],
    assignments: list[dict[str, Any]],
) -> None:
    lines = [
        "Expanded CHB-MIT Random Forest baseline",
        "========================================",
        "",
        "Validation: four-fold, recording-grouped stratified cross-validation.",
        "Stratification used recording-level seizure presence; each recording is",
        "assigned to one test fold only. Patients may occur in both train and test",
        "through different recordings. This is not patient-independent validation.",
        "The existing Random Forest settings and 0.5 threshold are unchanged.",
        "Exact duplicate predictors are detected on each training partition only.",
        "No learned preprocessing, feature selection, hyperparameter tuning, or",
        "threshold tuning was performed.",
        "",
        (
            f"Recordings: {pooled['recording_count']} "
            f"({pooled['seizure_recordings']} seizure-containing, "
            f"{pooled['non_seizure_recordings']} non-seizure)"
        ),
        (
            f"Windows: {pooled['windows']} "
            f"({pooled['class_counts']['seizure']} seizure, "
            f"{pooled['class_counts']['non_seizure']} non-seizure)"
        ),
        "",
        "Fold assignments",
        "-----------------",
    ]
    for fold in fold_metrics:
        test_records = json.loads(fold["test_recordings"])
        lines.append(
            f"{fold['fold']}: {len(test_records)} recordings; "
            f"{fold['test_seizure_recordings']} seizure-containing, "
            f"{fold['test_non_seizure_recordings']} non-seizure; "
            f"test windows seizure/non-seizure={fold['test_seizure']}/"
            f"{fold['test_non_seizure']}"
        )
        for patient, recording, relative in test_records:
            lines.append(f"  {patient} | {recording} | {relative}")
    metric_names = (
        "accuracy", "balanced_accuracy", "precision", "recall_sensitivity",
        "specificity", "f1_score", "roc_auc", "average_precision",
    )
    lines.extend(["", "Fold metrics", "------------"])
    for fold in fold_metrics:
        lines.append(f"{fold['fold']} confusion matrix [[TN, FP], [FN, TP]]: {fold['confusion_matrix']['values']}")
        lines.append("  " + ", ".join(f"{name}={fold[name]:.6f}" for name in metric_names))
    lines.extend(["", "Pooled out-of-fold metrics", "-------------------------"])
    lines.append(f"Confusion matrix [[TN, FP], [FN, TP]]: {pooled['confusion_matrix']['values']}")
    lines.append("  " + ", ".join(f"{name}={pooled[name]:.6f}" for name in metric_names))
    lines.extend(
        [
            "",
            "Limitations",
            "-----------",
            "Window metrics weight long recordings more heavily than short recordings.",
            "Recording-level and patient-level diagnostics are descriptive; windows",
            "from other recordings of the same patient can be present in training.",
            "The fixed 0.5 threshold and Random Forest configuration were not tuned.",
            "Results describe this selected three-patient subset and do not establish",
            "patient-independent generalization or reproduce the original paper.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, default=DEFAULT_FEATURE_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--fold-random-state", type=int, default=DEFAULT_FOLD_RANDOM_STATE)
    args = parser.parse_args()
    try:
        result = run_expanded_baseline(
            args.feature_matrix,
            args.output_dir,
            n_splits=args.folds,
            fold_random_state=args.fold_random_state,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Expanded Random Forest evaluation failed: {exc}", file=sys.stderr)
        return 1
    for fold in result["folds"]:
        print(
            f"{fold['fold']}: accuracy={fold['accuracy']:.4f} "
            f"balanced_accuracy={fold['balanced_accuracy']:.4f} "
            f"ROC-AUC={fold['roc_auc']:.4f} AP={fold['average_precision']:.4f}"
        )
    print(f"Pooled OOF metrics: {result['pooled']}")
    print(f"Artifacts: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
