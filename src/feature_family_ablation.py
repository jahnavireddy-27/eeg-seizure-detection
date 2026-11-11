#!/usr/bin/env python3
"""Run recording-wise EEG feature-family ablations against the saved baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from extract_features import SPECTRAL_FEATURES, TIME_FEATURES
from train_random_forest import (
    DEFAULT_FEATURE_MATRIX,
    DEFAULT_OUTPUT_DIR as DEFAULT_BASELINE_DIR,
    TEST_RECORDINGS,
    ForestConfiguration,
    build_random_forest,
    calculate_metrics,
    deduplicate_training_features,
    load_feature_matrix,
    make_recording_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "feature_family_ablation"
)
ABLATION_FAMILIES = ("time", "frequency", "wavelet")
COMBINED_REFERENCE = "all_features_combined_reference"
WAVELET_FEATURE = re.compile(r"wavelet_(?:A3|D3|D2|D1)_(?:mean|std_population|energy)")
PREDICTION_COLUMNS = {
    "recording_id",
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "y_true",
    "y_pred",
    "probability_seizure",
}
METRIC_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1_score",
    "roc_auc",
    "average_precision",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_family(column: str) -> str:
    """Return the feature-extractor family for a validated signal column."""
    if "__" not in column:
        raise ValueError(f"Not a channel feature column: {column!r}.")
    feature = column.rsplit("__", 1)[1]
    if feature in TIME_FEATURES:
        return "time"
    if (
        feature in SPECTRAL_FEATURES
        or feature.startswith("band_power_")
    ):
        return "frequency"
    if WAVELET_FEATURE.fullmatch(feature):
        return "wavelet"
    raise ValueError(f"Unrecognized feature family in column {column!r}.")


def select_feature_family_columns(
    columns: Sequence[str], family: str
) -> list[str]:
    """Select an ablation family or all validated signal predictors."""
    if family not in (*ABLATION_FAMILIES, "all"):
        raise ValueError(
            f"family must be one of {(*ABLATION_FAMILIES, 'all')}, got {family!r}."
        )
    classified = [(column, feature_family(column)) for column in columns]
    if family == "all":
        return [column for column, _ in classified]
    selected = [column for column, name in classified if name == family]
    if not selected:
        raise ValueError(f"No predictors found for feature family {family!r}.")
    return selected


def _load_configuration(baseline_dir: Path) -> tuple[dict[str, Any], ForestConfiguration]:
    path = baseline_dir / "run_configuration.json"
    if not path.is_file():
        raise FileNotFoundError(f"Saved baseline configuration not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if tuple(document.get("test_recordings", ())) != TEST_RECORDINGS:
        raise ValueError("Saved baseline folds do not match the required test recordings.")
    if document.get("threshold_rule") != "positive when P(seizure) >= 0.5":
        raise ValueError("Saved baseline threshold rule is not the required fixed 0.5.")
    configuration = ForestConfiguration(**document["random_forest"])
    configuration.validate()
    if not math.isclose(configuration.probability_threshold, 0.5):
        raise ValueError("Ablation requires the existing 0.5 probability threshold.")
    return document, configuration


def _load_reference(
    baseline_dir: Path,
    fold_name: str,
    expected_recording: str,
    configuration: ForestConfiguration,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_dir = baseline_dir / fold_name
    metrics_path = fold_dir / "metrics.json"
    predictions_path = fold_dir / "predictions.csv"
    metrics_document = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved_forest = ForestConfiguration(**metrics_document["random_forest"])
    if saved_forest != configuration:
        raise ValueError(f"{fold_name} saved forest settings differ from run configuration.")
    metrics = metrics_document["results"]
    if metrics["test_recording"] != expected_recording:
        raise ValueError(f"{fold_name} does not hold out {expected_recording}.")
    if not math.isclose(float(metrics["threshold"]), 0.5):
        raise ValueError(f"{fold_name} metrics do not use threshold 0.5.")
    for key in ("train_recordings", "train_windows", "test_windows"):
        if key not in metrics:
            raise ValueError(f"{fold_name} saved metrics lack {key}.")
    if expected_recording in metrics["train_recordings"]:
        raise ValueError(f"{fold_name} training recordings include its test recording.")
    frame = pd.read_csv(predictions_path)
    missing = PREDICTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{predictions_path} lacks {sorted(missing)}.")
    if len(frame) != int(metrics["test_windows"]):
        raise ValueError(f"{fold_name} saved prediction count does not match metrics.")
    if set(frame["recording_id"].astype(str)) != {expected_recording}:
        raise ValueError(f"{fold_name} predictions contain a non-test recording.")
    true = pd.to_numeric(frame["y_true"], errors="raise").to_numpy(dtype=np.int8)
    predicted = pd.to_numeric(frame["y_pred"], errors="raise").to_numpy(dtype=np.int8)
    probability = pd.to_numeric(
        frame["probability_seizure"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"{fold_name} saved probabilities are missing or invalid.")
    if not np.array_equal(predicted, (probability >= 0.5).astype(np.int8)):
        raise ValueError(f"{fold_name} labels disagree with the saved 0.5 threshold.")
    recomputed = calculate_metrics(true, predicted, probability)
    if recomputed["confusion_matrix"]["values"] != metrics["confusion_matrix"]["values"]:
        raise ValueError(f"{fold_name} reference confusion matrix failed verification.")
    for metric in METRIC_FIELDS:
        computed_value = recomputed[metric]
        saved_value = metrics[metric]
        if computed_value is None or saved_value is None:
            if computed_value is not saved_value:
                raise ValueError(f"{fold_name} reference metric {metric} failed verification.")
            continue
        if not math.isclose(float(computed_value), float(saved_value), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{fold_name} reference metric {metric} failed verification.")
    return frame, metrics


def _plot_confusion(
    path: Path, matrix: Sequence[Sequence[int]], title: str
) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.4, 4.5))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["non-seizure", "seizure"],
        yticklabels=["non-seizure", "seizure"],
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    midpoint = values.max() / 2.0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > midpoint else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_curves(
    fold_dir: Path,
    family_name: str,
    fold_name: str,
    test_recording: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, probability)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, label="ROC curve")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axis.scatter(
        [fp / (fp + tn)],
        [tp / (tp + fn)],
        color="crimson",
        label="Saved threshold ≥ 0.5",
        zorder=3,
    )
    axis.set(
        xlabel="False-positive rate (1 − specificity)",
        ylabel="True-positive rate (sensitivity)",
        title=f"{family_name}: {fold_name}, {test_recording}",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(fold_dir / "roc_curve.png", dpi=160)
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(y_true, probability)
    operating_precision = tp / (tp + fp) if tp + fp else float("nan")
    operating_recall = tp / (tp + fn) if tp + fn else float("nan")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.step(recall, precision, where="post", label="Precision-recall curve")
    axis.scatter(
        [operating_recall],
        [operating_precision],
        color="crimson",
        label="Saved threshold ≥ 0.5",
        zorder=3,
    )
    axis.set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision",
        title=f"{family_name}: {fold_name}, {test_recording}",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(fold_dir / "precision_recall_curve.png", dpi=160)
    plt.close(figure)


def _write_reference_fold(
    output_dir: Path,
    family_name: str,
    fold_name: str,
    test_recording: str,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    family_fold_dir = output_dir / family_name / fold_name
    family_fold_dir.mkdir(parents=True, exist_ok=True)
    true = predictions["y_true"].to_numpy(dtype=np.int8)
    predicted = predictions["y_pred"].to_numpy(dtype=np.int8)
    probability = predictions["probability_seizure"].to_numpy(dtype=np.float64)
    _plot_confusion(
        family_fold_dir / "confusion_matrix.png",
        metrics["confusion_matrix"]["values"],
        f"{family_name}: {fold_name}, {test_recording}",
    )
    _plot_curves(
        family_fold_dir,
        family_name,
        fold_name,
        test_recording,
        true,
        predicted,
        probability,
    )
    result = {
        "experiment": family_name,
        "fold": fold_name,
        "test_recording": test_recording,
        "predictors_before_duplicate_removal": int(
            metrics["predictor_count_before_deduplication"]
        ),
        "predictors_after_duplicate_removal": int(
            metrics["predictor_count_after_deduplication"]
        ),
        "excluded_duplicate_count": int(metrics["excluded_duplicate_count"]),
        "training_duration_seconds": None,
        "prediction_duration_seconds": None,
        "duration_note": "not available; reused saved baseline predictions",
        **{key: metrics[key] for key in METRIC_FIELDS},
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]["values"]),
        "train_windows": int(metrics["train_windows"]),
        "train_non_seizure": int(metrics["train_non_seizure"]),
        "train_seizure": int(metrics["train_seizure"]),
        "test_windows": int(metrics["test_windows"]),
        "test_non_seizure": int(metrics["test_non_seizure"]),
        "test_seizure": int(metrics["test_seizure"]),
        "train_recordings": json.dumps(metrics["train_recordings"]),
    }
    return result


def _run_family_fold(
    frame: pd.DataFrame,
    fold: Any,
    feature_columns: Sequence[str],
    family_name: str,
    configuration: ForestConfiguration,
    output_dir: Path,
) -> dict[str, Any]:
    train_recordings = set(
        frame.iloc[fold.train_indices]["recording_id"].astype(str)
    )
    test_recordings = set(frame.iloc[fold.test_indices]["recording_id"].astype(str))
    if train_recordings & test_recordings or test_recordings != {fold.test_recording}:
        raise RuntimeError(f"{fold.name}: recording-wise isolation check failed.")

    labels = frame["seizure_label"].map({"non_seizure": 0, "seizure": 1}).to_numpy(
        dtype=np.int8
    )
    train_counts = np.bincount(labels[fold.train_indices], minlength=2)
    test_counts = np.bincount(labels[fold.test_indices], minlength=2)
    if np.any(train_counts == 0) or np.any(test_counts == 0):
        raise ValueError(f"{fold.name} must contain both classes in train and test.")
    training = frame.iloc[fold.train_indices][list(feature_columns)]
    testing = frame.iloc[fold.test_indices][list(feature_columns)]
    retained, excluded = deduplicate_training_features(training)
    model = build_random_forest(configuration)

    training_started = time.perf_counter()
    model.fit(training[retained], labels[fold.train_indices])
    training_duration = time.perf_counter() - training_started

    prediction_started = time.perf_counter()
    probabilities = model.predict_proba(testing[retained])[:, 1]
    predictions = (probabilities >= configuration.probability_threshold).astype(np.int8)
    prediction_duration = time.perf_counter() - prediction_started
    metrics = calculate_metrics(
        labels[fold.test_indices], predictions, probabilities
    )

    family_fold_dir = output_dir / family_name / fold.name
    family_fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "excluded_feature": excluded_feature,
                "retained_feature": retained_feature,
                "detection_partition": "training_only",
            }
            for excluded_feature, retained_feature in excluded
        ],
        columns=("excluded_feature", "retained_feature", "detection_partition"),
    ).to_csv(family_fold_dir / "excluded_duplicate_features.csv", index=False)
    _plot_confusion(
        family_fold_dir / "confusion_matrix.png",
        metrics["confusion_matrix"]["values"],
        f"{family_name}: {fold.name}, {fold.test_recording}",
    )
    _plot_curves(
        family_fold_dir,
        family_name,
        fold.name,
        fold.test_recording,
        labels[fold.test_indices],
        predictions,
        probabilities,
    )
    return {
        "experiment": family_name,
        "fold": fold.name,
        "test_recording": fold.test_recording,
        "predictors_before_duplicate_removal": len(feature_columns),
        "predictors_after_duplicate_removal": len(retained),
        "excluded_duplicate_count": len(excluded),
        "training_duration_seconds": training_duration,
        "prediction_duration_seconds": prediction_duration,
        "duration_note": "measured with perf_counter; prediction includes predict_proba and thresholding",
        **{key: metrics[key] for key in METRIC_FIELDS},
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]["values"]),
        "train_windows": int(len(fold.train_indices)),
        "train_non_seizure": int(train_counts[0]),
        "train_seizure": int(train_counts[1]),
        "test_windows": int(len(fold.test_indices)),
        "test_non_seizure": int(test_counts[0]),
        "test_seizure": int(test_counts[1]),
        "train_recordings": json.dumps(list(fold.train_recordings)),
    }


def _write_summary(metrics: pd.DataFrame, path: Path) -> None:
    rows = []
    for experiment, group in metrics.groupby("experiment", sort=False):
        row: dict[str, Any] = {
            "experiment": experiment,
            "fold_count": int(len(group)),
        }
        for column in (
            "predictors_before_duplicate_removal",
            "predictors_after_duplicate_removal",
            "excluded_duplicate_count",
            *METRIC_FIELDS,
        ):
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"mean_{column}"] = (
                float(values.mean()) if values.notna().any() else None
            )
        for fold_number, fold_row in enumerate(
            group.sort_values("fold").itertuples(index=False), start=1
        ):
            row[f"fold_{fold_number}_test_recording"] = fold_row.test_recording
            row[f"fold_{fold_number}_confusion_matrix"] = fold_row.confusion_matrix
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    config: dict[str, Any],
    input_hashes: dict[str, str],
) -> None:
    lines = [
        "# Feature-family ablation",
        "",
        "## Protocol and integrity",
        "",
        "This is an exploratory within-patient, recording-wise evaluation on the "
        "available CHB01 recordings. Fold 1 holds out `chb01_16`; Fold 2 holds "
        "out `chb01_18`. Every recording remains wholly in train or test. No "
        "feature selection, threshold tuning, or test-informed preprocessing "
        "was performed.",
        "",
        "The three individual feature-family experiments fit a separate Random "
        "Forest per fold. The all-feature combined row is the existing saved "
        "baseline reference: it is read and independently checked, not retrained "
        "or replaced. Exact duplicate columns are detected separately on each "
        "training partition within each family; only exact equality is removed.",
        "",
        "No metadata, identifiers, times, indices, target labels, or label policy "
        "are predictors. All models use the unchanged saved baseline settings and "
        "the fixed `P(seizure) >= 0.5` threshold:",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "Input SHA-256:",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in input_hashes.items())
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Experiment | Fold | Test recording | Predictors before/after dedup | Train seizure/non-seizure | Test seizure/non-seizure | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | Average precision | TN, FP; FN, TP | Train s | Predict s |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in metrics.itertuples(index=False):
        matrix = json.loads(row.confusion_matrix)
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        training_duration = (
            f"{row.training_duration_seconds:.4f}"
            if pd.notna(row.training_duration_seconds)
            else "n/a"
        )
        prediction_duration = (
            f"{row.prediction_duration_seconds:.4f}"
            if pd.notna(row.prediction_duration_seconds)
            else "n/a"
        )
        formatted_metrics = [
            f"{value:.4f}" if pd.notna(value) else "undefined"
            for value in (
                row.accuracy,
                row.balanced_accuracy,
                row.recall_sensitivity,
                row.specificity,
                row.precision,
                row.f1_score,
                row.roc_auc,
                row.average_precision,
            )
        ]
        lines.append(
            f"| {row.experiment} | {row.fold} | `{row.test_recording}` | "
            f"{row.predictors_before_duplicate_removal}/{row.predictors_after_duplicate_removal} | "
            f"{row.train_seizure}/{row.train_non_seizure} | "
            f"{row.test_seizure}/{row.test_non_seizure} | "
            f"{' | '.join(formatted_metrics)} | {tn}, {fp}; {fn}, {tp} | "
            f"{training_duration} | {prediction_duration} |"
        )
    lines.extend(
        [
            "",
            "Per-fold metrics, including full confusion matrices and measured durations, "
            "are in `per_fold_metrics.csv`; unweighted two-fold means are in "
            "`experiment_summary.csv`. Baseline reference durations are unavailable "
            "because timing was not recorded in the existing artifacts.",
            "",
            "## Interpretation",
            "",
            "Compare the family-specific results with `all_features_combined_reference` "
            "as a descriptive ablation only. The mean is an unweighted mean of two "
            "recording-level folds, not a population estimate. A feature family that "
            "scores higher here is not universally superior; these folds contain one "
            "patient and just two seizure-containing recordings, and positive windows "
            "are sparse and overlapping.",
            "",
            "The paper does not define the exact FFT and wavelet feature summaries; the "
            "families use the project's documented reconstruction choices. The "
            "`f_classif` feature-selection experiment is not included.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    feature_matrix_path: Path = DEFAULT_FEATURE_MATRIX,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    configuration_document, configuration = _load_configuration(baseline_dir)
    frame, all_feature_columns = load_feature_matrix(feature_matrix_path)
    folds = make_recording_folds(frame)

    family_columns = {
        family: select_feature_family_columns(all_feature_columns, family)
        for family in ABLATION_FAMILIES
    }
    if set().union(*(set(columns) for columns in family_columns.values())) != set(
        all_feature_columns
    ):
        raise ValueError("Feature-family mapping does not cover all signal predictors.")
    if sum(map(len, family_columns.values())) != len(all_feature_columns):
        raise ValueError("A signal feature was mapped to more than one family.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reference_hashes: dict[str, str] = {}
    feature_hashes = {"feature_matrix": _sha256(feature_matrix_path)}

    for family in ABLATION_FAMILIES:
        for fold in folds:
            rows.append(
                _run_family_fold(
                    frame,
                    fold,
                    family_columns[family],
                    family,
                    configuration,
                    output_dir,
                )
            )

    for fold_number, fold in enumerate(folds, start=1):
        fold_name = f"fold_{fold_number}"
        baseline_fold_dir = baseline_dir / fold_name
        predictions_path = baseline_fold_dir / "predictions.csv"
        metrics_path = baseline_fold_dir / "metrics.json"
        reference_hashes[f"{fold_name}/predictions.csv"] = _sha256(predictions_path)
        reference_hashes[f"{fold_name}/metrics.json"] = _sha256(metrics_path)
        reference_predictions, reference_metrics = _load_reference(
            baseline_dir,
            fold_name,
            fold.test_recording,
            configuration,
        )
        if set(frame.iloc[fold.train_indices]["recording_id"].astype(str)) & {
            fold.test_recording
        }:
            raise RuntimeError(f"{fold_name}: test recording contributes to training.")
        if int(reference_metrics["train_windows"]) != len(fold.train_indices):
            raise ValueError(f"{fold_name} training rows differ from saved baseline.")
        if int(reference_metrics["test_windows"]) != len(fold.test_indices):
            raise ValueError(f"{fold_name} test rows differ from saved baseline.")
        if tuple(reference_metrics["train_recordings"]) != fold.train_recordings:
            raise ValueError(f"{fold_name} training recording groups differ.")
        expected_test = frame.iloc[fold.test_indices]
        expected_labels = expected_test["seizure_label"].map(
            {"non_seizure": 0, "seizure": 1}
        ).to_numpy(dtype=np.int8)
        if not np.array_equal(
            reference_predictions["y_true"].to_numpy(dtype=np.int8), expected_labels
        ):
            raise ValueError(f"{fold_name} reference labels do not match the feature matrix.")
        for column in ("window_index", "start_time_seconds", "end_time_seconds"):
            saved_values = pd.to_numeric(
                reference_predictions[column], errors="raise"
            ).to_numpy(dtype=np.float64)
            expected_values = pd.to_numeric(
                expected_test[column], errors="raise"
            ).to_numpy(dtype=np.float64)
            if not np.array_equal(saved_values, expected_values):
                raise ValueError(
                    f"{fold_name} reference {column} does not align with test windows."
                )
        rows.append(
            _write_reference_fold(
                output_dir,
                COMBINED_REFERENCE,
                fold_name,
                fold.test_recording,
                reference_predictions,
                reference_metrics,
            )
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "per_fold_metrics.csv", index=False)
    _write_summary(result, output_dir / "experiment_summary.csv")
    input_hashes = {
        "feature_matrix": feature_hashes["feature_matrix"],
        "baseline/run_configuration.json": _sha256(baseline_dir / "run_configuration.json"),
        **{
            f"baseline/{name}": digest
            for name, digest in reference_hashes.items()
        },
    }
    run_document = {
        "feature_families": list(ABLATION_FAMILIES),
        "combined_reference": (
            "existing saved Random Forest baseline predictions; not retrained"
        ),
        "test_recordings": list(TEST_RECORDINGS),
        "threshold_rule": "positive when P(seizure) >= 0.5",
        "random_forest": configuration_document["random_forest"],
        "duplicate_detection": (
            "exact equality on training partition, independently within each experiment and fold"
        ),
        "feature_selection": "none",
        "data_dependent_preprocessing": "none",
        "input_sha256": input_hashes,
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(run_document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "report.md",
        result,
        run_document,
        input_hashes,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, default=DEFAULT_FEATURE_MATRIX)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = run_ablation(
            args.feature_matrix.expanduser().resolve(),
            args.baseline_dir.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Feature-family ablation failed: {exc}", file=sys.stderr)
        return 1
    print(result[["experiment", "fold", "test_recording"]].to_string(index=False))
    print(f"Artifacts: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
