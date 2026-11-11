#!/usr/bin/env python3
"""Run the first recording-wise Random Forest baseline on EEG feature rows."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_MATRIX = (
    PROJECT_ROOT / "data" / "processed" / "features" / "chb01_features.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "random_forest_baseline"
)
TEST_RECORDINGS = ("chb01_16", "chb01_18")
LABEL_TO_INT = {"non_seizure": 0, "seizure": 1}
PREDICTION_COLUMNS = (
    "patient_id",
    "recording_id",
    "recording_filename",
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "y_true",
    "y_pred",
    "probability_non_seizure",
    "probability_seizure",
)
REQUIRED_METADATA_COLUMNS = {
    "patient_id",
    "recording_id",
    "recording_filename",
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "seizure_label",
    "label_policy",
    "sampling_frequency_hz",
    "window_samples",
    "feature_config_json",
}
OPTIONAL_METADATA_COLUMNS = {
    "relative_path",
    "annotation_status",
}
METADATA_COLUMNS = REQUIRED_METADATA_COLUMNS | OPTIONAL_METADATA_COLUMNS
SIGNAL_FEATURE_COLUMN = re.compile(
    r"(?:channel_\d+__.+|channel__.+__instance_\d+)__(?:mean|std_population|maximum|minimum|"
    r"dominant_frequency_hz|spectral_centroid_hz|total_spectral_power|"
    r"band_power_[A-Za-z0-9_]+|wavelet_[AD]\d+_(?:mean|std_population|energy))"
)


@dataclass(frozen=True)
class ForestConfiguration:
    n_estimators: int = 300
    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = 1
    probability_threshold: float = 0.5

    def validate(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be positive or None.")
        if not math.isclose(self.probability_threshold, 0.5):
            raise ValueError("This baseline fixes the classification threshold at 0.5.")
        if self.class_weight != "balanced":
            raise ValueError('This baseline requires class_weight="balanced".')
        if self.n_jobs != 1:
            raise ValueError("Use n_jobs=1 for strict repeatability in this baseline.")


@dataclass(frozen=True)
class Fold:
    name: str
    test_recording: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_recordings: tuple[str, ...]


def load_feature_matrix_from_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        raise ValueError("Feature matrix contains no rows.")
    missing = REQUIRED_METADATA_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Feature matrix is missing required metadata: "
            + ", ".join(sorted(missing))
        )
    signal_columns = [
        column
        for column in frame.columns
        if SIGNAL_FEATURE_COLUMN.fullmatch(column)
    ]
    unexpected = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS and column not in signal_columns
    ]
    if unexpected:
        raise ValueError(
            "Unexpected non-signal columns must not enter the model: "
            + ", ".join(unexpected)
        )
    if not signal_columns:
        raise ValueError("Feature matrix has no channel-prefixed signal predictors.")
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Feature matrix has duplicate column names.")
    if not frame["recording_id"].notna().all() or not frame["patient_id"].notna().all():
        raise ValueError("Patient and recording identifiers cannot be missing.")
    if not frame["window_index"].notna().all():
        raise ValueError("Window indices cannot be missing.")
    invalid_labels = sorted(set(frame["seizure_label"].dropna()) - set(LABEL_TO_INT))
    if invalid_labels or frame["seizure_label"].isna().any():
        raise ValueError(
            f"Only verified seizure labels are accepted; invalid labels: {invalid_labels}"
        )
    try:
        values = frame[signal_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every channel-prefixed feature must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Signal feature matrix contains missing or non-finite values.")
    return frame, signal_columns


def load_feature_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature matrix does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    return load_feature_matrix_from_frame(frame)


def make_recording_folds(
    frame: pd.DataFrame,
    test_recordings: Sequence[str] = TEST_RECORDINGS,
) -> tuple[Fold, ...]:
    if frame["recording_id"].isna().any():
        raise ValueError("recording_id cannot be missing.")
    present = set(frame["recording_id"].astype(str))
    missing = set(test_recordings) - present
    if missing:
        raise ValueError(
            "Expected held-out seizure recordings are absent: "
            + ", ".join(sorted(missing))
        )
    if len(set(test_recordings)) != len(test_recordings):
        raise ValueError("Test recording IDs must be unique.")

    folds = []
    recording_ids = frame["recording_id"].astype(str).to_numpy()
    labels = frame["seizure_label"].map(LABEL_TO_INT).to_numpy(dtype=np.int8)
    for fold_number, test_recording in enumerate(test_recordings, start=1):
        test_mask = recording_ids == test_recording
        train_mask = ~test_mask
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        train_groups = set(recording_ids[train_indices])
        test_groups = set(recording_ids[test_indices])
        if train_groups & test_groups:
            raise RuntimeError(f"Recording leakage detected in fold {fold_number}.")
        if not len(train_indices) or not len(test_indices):
            raise ValueError(f"Fold {fold_number} has an empty partition.")
        train_counts = np.bincount(labels[train_indices], minlength=2)
        test_counts = np.bincount(labels[test_indices], minlength=2)
        if np.any(train_counts == 0):
            raise ValueError(
                f"Fold {fold_number} training partition lacks a class: "
                f"non_seizure={train_counts[0]}, seizure={train_counts[1]}."
            )
        if np.any(test_counts == 0):
            raise ValueError(
                f"Fold {fold_number} test partition lacks a class: "
                f"non_seizure={test_counts[0]}, seizure={test_counts[1]}."
            )
        folds.append(
            Fold(
                name=f"fold_{fold_number}",
                test_recording=test_recording,
                train_indices=train_indices,
                test_indices=test_indices,
                train_recordings=tuple(sorted(train_groups)),
            )
        )
    return tuple(folds)


def deduplicate_training_features(
    training_features: pd.DataFrame,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Keep the first exactly equal training feature and report each exclusion."""
    retained: list[str] = []
    excluded: list[tuple[str, str]] = []
    digest_groups: dict[str, list[str]] = {}
    arrays: dict[str, np.ndarray] = {}
    for column in training_features.columns:
        values = training_features[column].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {column!r} contains non-finite training values.")
        values[values == 0.0] = 0.0
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        match = next(
            (
                previous
                for previous in digest_groups.get(digest, [])
                if np.array_equal(values, arrays[previous])
            ),
            None,
        )
        if match is None:
            retained.append(column)
            arrays[column] = values
            digest_groups.setdefault(digest, []).append(column)
        else:
            excluded.append((column, match))
    if not retained:
        raise ValueError("No unique predictors remain after exact deduplication.")
    return retained, excluded


def build_random_forest(
    configuration: ForestConfiguration = ForestConfiguration(),
) -> RandomForestClassifier:
    configuration.validate()
    return RandomForestClassifier(
        n_estimators=configuration.n_estimators,
        criterion=configuration.criterion,
        max_depth=configuration.max_depth,
        min_samples_split=configuration.min_samples_split,
        min_samples_leaf=configuration.min_samples_leaf,
        max_features=configuration.max_features,
        bootstrap=configuration.bootstrap,
        class_weight=configuration.class_weight,
        random_state=configuration.random_state,
        n_jobs=configuration.n_jobs,
    )


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probability_seizure: Sequence[float],
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(y_pred, dtype=np.int8)
    probabilities = np.asarray(probability_seizure, dtype=np.float64)
    if true.ndim != 1 or predicted.shape != true.shape or probabilities.shape != true.shape:
        raise ValueError("Labels and probabilities must be same-length 1D arrays.")
    if not np.isin(true, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary values 0 or 1.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")
    if set(true.tolist()) != {0, 1}:
        raise ValueError("Metrics require both classes in the test labels.")

    matrix = confusion_matrix(true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision = _ratio_or_none(tp, tp + fp)
    recall = _ratio_or_none(tp, tp + fn)
    specificity = _ratio_or_none(tn, tn + fp)
    f1 = (
        _ratio_or_none(2 * tp, 2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else None
    )
    undefined = [
        metric
        for metric, value in (
            ("precision", precision),
            ("recall", recall),
            ("specificity", specificity),
            ("f1", f1),
        )
        if value is None
    ]
    return {
        "confusion_matrix": {
            "labels": ["non_seizure", "seizure"],
            "values": matrix.tolist(),
        },
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": float(roc_auc_score(true, probabilities)),
        "average_precision": float(average_precision_score(true, probabilities)),
        "undefined_metrics": undefined,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot_confusion_matrix(
    path: Path,
    matrix: Sequence[Sequence[int]],
    fold_name: str,
    test_recording: str,
) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    fig, axis = plt.subplots(figsize=(5.4, 4.5))
    image = axis.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["non-seizure", "seizure"],
        yticklabels=["non-seizure", "seizure"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{fold_name}: held-out {test_recording}",
    )
    threshold = values.max() / 2.0 if values.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{values[row, column]}",
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_baseline(
    feature_matrix_path: Path = DEFAULT_FEATURE_MATRIX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    configuration: ForestConfiguration = ForestConfiguration(),
) -> list[dict[str, Any]]:
    configuration.validate()
    frame, feature_columns = load_feature_matrix(feature_matrix_path)
    folds = make_recording_folds(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_fold: list[dict[str, Any]] = []
    prediction_fields = list(PREDICTION_COLUMNS)
    label_values = frame["seizure_label"].map(LABEL_TO_INT).to_numpy(dtype=np.int8)

    for fold in folds:
        fold_dir = output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_frame = frame.iloc[fold.train_indices]
        test_frame = frame.iloc[fold.test_indices]
        train_labels = label_values[fold.train_indices]
        test_labels = label_values[fold.test_indices]
        train_features = frame.iloc[fold.train_indices][feature_columns]
        test_features = frame.iloc[fold.test_indices][feature_columns]
        retained, excluded = deduplicate_training_features(train_features)
        model = build_random_forest(configuration)
        model.fit(train_features[retained], train_labels)
        if list(model.classes_) != [0, 1]:
            raise RuntimeError(
                f"{fold.name}: fitted model classes are {model.classes_}, expected [0, 1]."
            )
        probabilities = model.predict_proba(test_features[retained])[:, 1]
        predictions = (probabilities >= configuration.probability_threshold).astype(
            np.int8
        )
        metrics = calculate_metrics(test_labels, predictions, probabilities)

        duplicate_rows = [
            {
                "excluded_feature": excluded_column,
                "retained_feature": retained_column,
                "detection_partition": "training_only",
            }
            for excluded_column, retained_column in excluded
        ]
        _write_csv(
            fold_dir / "excluded_duplicate_features.csv",
            ("excluded_feature", "retained_feature", "detection_partition"),
            duplicate_rows,
        )

        predictions_frame = test_frame[
            [
                "patient_id",
                "recording_id",
                "recording_filename",
                "window_index",
                "start_time_seconds",
                "end_time_seconds",
            ]
        ].copy()
        predictions_frame["y_true"] = test_labels
        predictions_frame["y_pred"] = predictions
        predictions_frame["probability_non_seizure"] = 1.0 - probabilities
        predictions_frame["probability_seizure"] = probabilities
        if set(predictions_frame["recording_id"].astype(str)) != {fold.test_recording}:
            raise RuntimeError(f"{fold.name}: predictions contain a non-test recording.")
        predictions_frame.to_csv(
            fold_dir / "predictions.csv",
            index=False,
            columns=prediction_fields,
            float_format="%.17g",
        )

        train_counts = np.bincount(train_labels, minlength=2)
        test_counts = np.bincount(test_labels, minlength=2)
        result = {
            "fold": fold.name,
            "test_recording": fold.test_recording,
            "train_recordings": list(fold.train_recordings),
            "train_windows": int(len(train_labels)),
            "train_non_seizure": int(train_counts[0]),
            "train_seizure": int(train_counts[1]),
            "test_windows": int(len(test_labels)),
            "test_non_seizure": int(test_counts[0]),
            "test_seizure": int(test_counts[1]),
            "predictor_count_before_deduplication": len(feature_columns),
            "predictor_count_after_deduplication": len(retained),
            "excluded_duplicate_count": len(excluded),
            "threshold": configuration.probability_threshold,
            **metrics,
        }
        metrics_by_fold.append(result)
        _write_json(
            fold_dir / "metrics.json",
            {"random_forest": asdict(configuration), "results": result},
        )
        _plot_confusion_matrix(
            fold_dir / "confusion_matrix.png",
            metrics["confusion_matrix"]["values"],
            fold.name,
            fold.test_recording,
        )

    summary_fields = [
        "fold",
        "test_recording",
        "train_windows",
        "train_non_seizure",
        "train_seizure",
        "test_windows",
        "test_non_seizure",
        "test_seizure",
        "predictor_count_before_deduplication",
        "predictor_count_after_deduplication",
        "excluded_duplicate_count",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "average_precision",
    ]
    _write_csv(
        output_dir / "fold_comparison.csv",
        summary_fields,
        [
            {field: result[field] for field in summary_fields}
            for result in metrics_by_fold
        ],
    )
    _write_json(
        output_dir / "run_configuration.json",
        {
            "random_forest": asdict(configuration),
            "test_recordings": list(TEST_RECORDINGS),
            "predictor_selection": "channel-prefixed signal features only",
            "duplicate_detection": "exact equality on training partition per fold",
            "feature_selection": "none",
            "threshold_rule": "positive when P(seizure) >= 0.5",
        },
    )
    return metrics_by_fold


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the recording-wise Random Forest baseline."
    )
    parser.add_argument("--feature-matrix", type=Path, default=DEFAULT_FEATURE_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        results = run_baseline(
            feature_matrix_path=args.feature_matrix.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Random Forest baseline failed: {exc}", file=sys.stderr)
        return 1
    for result in results:
        print(
            f"{result['fold']} test={result['test_recording']} "
            f"train={result['train_seizure']}/{result['train_non_seizure']} "
            f"test={result['test_seizure']}/{result['test_non_seizure']} "
            f"accuracy={result['accuracy']:.4f} "
            f"balanced_accuracy={result['balanced_accuracy']:.4f} "
            f"ROC-AUC={result['roc_auc']:.4f} "
            f"average_precision={result['average_precision']:.4f}"
        )
    print(f"Artifacts: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
