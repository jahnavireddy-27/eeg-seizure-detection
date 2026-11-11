"""Analyze saved Random Forest fold predictions without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PREDICTION_COLUMNS = {
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
}
ERROR_COLUMNS = [
    "fold",
    "patient_id",
    "recording_id",
    "recording_filename",
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "y_true",
    "y_pred",
    "probability_seizure",
]
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_predictions(path: Path, fold: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = PREDICTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no predictions.")
    numeric = [
        "window_index",
        "start_time_seconds",
        "end_time_seconds",
        "y_true",
        "y_pred",
        "probability_non_seizure",
        "probability_seizure",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains missing or non-finite numeric values.")
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} contains labels outside 0/1.")
    if not set(frame["y_pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} contains predictions outside 0/1.")
    probabilities = frame[
        ["probability_non_seizure", "probability_seizure"]
    ].to_numpy(dtype=float)
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{path} contains probabilities outside [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"{path} class probabilities do not sum to one.")
    if (frame["end_time_seconds"] <= frame["start_time_seconds"]).any():
        raise ValueError(f"{path} contains invalid window time bounds.")
    frame["fold"] = fold
    expected_recordings = {"fold_1": "chb01_16", "fold_2": "chb01_18"}
    if set(frame["recording_id"].astype(str)) != {expected_recordings[fold]}:
        raise ValueError(f"{path} contains unexpected recording IDs.")
    return frame


def _distribution_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for label, label_name in ((0, "non_seizure"), (1, "seizure")):
        values = frame.loc[frame["y_true"] == label, "probability_seizure"].to_numpy()
        if values.size == 0:
            raise ValueError(f"{frame['fold'].iloc[0]} has no {label_name} windows.")
        row: dict[str, Any] = {
            "fold": frame["fold"].iloc[0],
            "recording_id": frame["recording_id"].iloc[0],
            "true_class": label_name,
            "count": int(values.size),
            "minimum": float(values.min()),
        }
        for quantile in QUANTILES:
            row[f"q{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
        row["maximum"] = float(values.max())
        rows.append(row)
    return rows


def _plot_curves(frame: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    y_true = frame["y_true"].to_numpy(dtype=np.int8)
    probability = frame["probability_seizure"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=np.int8)
    fold = str(frame["fold"].iloc[0])
    recording = str(frame["recording_id"].iloc[0])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision, recall, _ = precision_recall_curve(y_true, probability)
    operating_precision = tp / (tp + fp) if tp + fp else float("nan")
    operating_recall = tp / (tp + fn) if tp + fn else float("nan")
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.step(recall, precision, where="post", label="Precision-recall curve")
    axis.scatter(
        [operating_recall],
        [operating_precision],
        color="crimson",
        zorder=3,
        label=f"Saved threshold ≥ {threshold:g}",
    )
    axis.set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision",
        title=f"{fold} — {recording} (AP={average_precision_score(y_true, probability):.4f})",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{fold}_precision_recall.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probability)
    operating_fpr = fp / (fp + tn) if fp + tn else float("nan")
    operating_tpr = tp / (tp + fn) if tp + fn else float("nan")
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, label="ROC curve")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.scatter(
        [operating_fpr],
        [operating_tpr],
        color="crimson",
        zorder=3,
        label=f"Saved threshold ≥ {threshold:g}",
    )
    axis.set(
        xlabel="False-positive rate (1 − specificity)",
        ylabel="True-positive rate (sensitivity)",
        title=f"{fold} — {recording} (ROC-AUC={roc_auc_score(y_true, probability):.4f})",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{fold}_roc.png", dpi=160)
    plt.close(fig)


def _fold_analysis(
    frame: pd.DataFrame, metrics_path: Path, threshold: float, output_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))["results"]
    y_true = frame["y_true"].to_numpy(dtype=np.int8)
    y_pred = frame["y_pred"].to_numpy(dtype=np.int8)
    probability = frame["probability_seizure"].to_numpy(dtype=float)
    implied_prediction = (probability >= threshold).astype(np.int8)
    if not np.array_equal(implied_prediction, y_pred):
        raise ValueError(f"{frame['fold'].iloc[0]} y_pred disagrees with saved threshold.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    observed = {"true_positive": int(tp), "false_negative": int(fn),
                "false_positive": int(fp), "true_negative": int(tn)}
    for key, value in observed.items():
        if int(stored[key]) != value:
            raise ValueError(
                f"{frame['fold'].iloc[0]} {key} mismatch: predictions={value}, "
                f"metrics={stored[key]}."
            )
    if int(stored["test_windows"]) != len(frame):
        raise ValueError(f"{frame['fold'].iloc[0]} prediction row count mismatch.")

    false_negatives = frame.loc[(frame["y_true"] == 1) & (frame["y_pred"] == 0)]
    false_positives = frame.loc[(frame["y_true"] == 0) & (frame["y_pred"] == 1)]
    negative_probabilities = frame.loc[
        frame["y_true"] == 0, "probability_seizure"
    ].to_numpy()
    fn_probabilities = false_negatives["probability_seizure"].to_numpy()
    fn_summary: dict[str, Any] = {
        "false_negative_count": int(len(false_negatives)),
        "false_positive_count": int(len(false_positives)),
        "probability_threshold": threshold,
    }
    if fn_probabilities.size:
        fn_summary.update(
            {
                "false_negative_probability_minimum": float(fn_probabilities.min()),
                "false_negative_probability_median": float(np.median(fn_probabilities)),
                "false_negative_probability_maximum": float(fn_probabilities.max()),
                "false_negative_max_distance_below_threshold": float(
                    threshold - fn_probabilities.max()
                ),
                "negative_windows_with_probability_at_least_fn": float(
                    np.mean(
                        negative_probabilities[:, None] >= fn_probabilities[None, :],
                        axis=0,
                    ).mean()
                ),
            }
        )
    else:
        fn_summary["false_negative_probability_minimum"] = None
        fn_summary["false_negative_probability_median"] = None
        fn_summary["false_negative_probability_maximum"] = None
        fn_summary["false_negative_max_distance_below_threshold"] = None
        fn_summary["negative_windows_with_probability_at_least_fn"] = None

    _plot_curves(frame, output_dir, threshold)
    return {
        "fold": str(frame["fold"].iloc[0]),
        "recording_id": str(frame["recording_id"].iloc[0]),
        **observed,
        "test_windows": int(len(frame)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        **fn_summary,
    }, _distribution_rows(frame)


def analyze(output_root: Path) -> Path:
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    threshold = float(config["random_forest"]["probability_threshold"])
    threshold_rule = config["threshold_rule"]
    if threshold_rule != f"positive when P(seizure) >= {threshold:g}":
        raise ValueError("Saved threshold rule and configuration threshold disagree.")

    all_frames = []
    fold_summaries = []
    distributions = []
    for fold in ("fold_1", "fold_2"):
        fold_dir = output_root / fold
        predictions_path = fold_dir / "predictions.csv"
        metrics_path = fold_dir / "metrics.json"
        frame = _load_predictions(predictions_path, fold)
        summary, rows = _fold_analysis(frame, metrics_path, threshold, diagnostics_dir)
        all_frames.append(frame)
        fold_summaries.append(summary)
        distributions.extend(rows)

    all_predictions = pd.concat(all_frames, ignore_index=True)
    false_negatives = all_predictions.loc[
        (all_predictions["y_true"] == 1) & (all_predictions["y_pred"] == 0),
        ERROR_COLUMNS,
    ].sort_values(["fold", "start_time_seconds", "window_index"])
    false_positives = all_predictions.loc[
        (all_predictions["y_true"] == 0) & (all_predictions["y_pred"] == 1),
        ERROR_COLUMNS,
    ].sort_values(["fold", "start_time_seconds", "window_index"])
    false_negatives.to_csv(diagnostics_dir / "false_negatives.csv", index=False)
    false_positives.to_csv(diagnostics_dir / "false_positives.csv", index=False)
    pd.DataFrame(distributions).to_csv(
        diagnostics_dir / "probability_distributions.csv", index=False
    )
    pd.DataFrame(fold_summaries).to_csv(
        diagnostics_dir / "fold_diagnostic_summary.csv", index=False
    )

    input_paths = [
        config_path,
        *(output_root / fold / "predictions.csv" for fold in ("fold_1", "fold_2")),
        *(output_root / fold / "metrics.json" for fold in ("fold_1", "fold_2")),
    ]
    input_hashes = {str(path.relative_to(output_root)): _sha256(path) for path in input_paths}
    report_path = diagnostics_dir / "analysis_report.md"
    _write_report(
        report_path,
        config,
        input_hashes,
        fold_summaries,
        distributions,
        false_negatives,
        false_positives,
    )
    return report_path


def _write_report(
    report_path: Path,
    config: dict[str, Any],
    input_hashes: dict[str, str],
    summaries: list[dict[str, Any]],
    distributions: list[dict[str, Any]],
    false_negatives: pd.DataFrame,
    false_positives: pd.DataFrame,
) -> None:
    threshold = config["random_forest"]["probability_threshold"]
    lines = [
        "# Random Forest baseline: prediction diagnostics",
        "",
        "## Scope and artifacts",
        "",
        "This is a post-hoc analysis of saved held-out predictions. No model was "
        "fit, no threshold was changed, and no feature selection or tuning was performed.",
        f"The saved decision rule is positive when seizure probability is >= {threshold}.",
        "Timestamps below are copied from each prediction CSV; no missing times or "
        "probabilities were imputed.",
        "",
        "Inputs (SHA-256):",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in input_hashes.items())
    lines.extend(
        [
            "",
            "Configuration:",
            "",
            "```json",
            json.dumps(config, indent=2, sort_keys=True),
            "```",
            "",
            "## Confusion-matrix verification",
            "",
            "| Fold | Recording | TP | FN | FP | TN | ROC-AUC | Average precision | Sensitivity | Specificity |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['fold']} | `{row['recording_id']}` | {row['true_positive']} | "
            f"{row['false_negative']} | {row['false_positive']} | {row['true_negative']} | "
            f"{row['roc_auc']:.4f} | {row['average_precision']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Independently recomputed counts match the saved per-fold metrics and "
            "the saved `y_pred` values agree with the configured threshold.",
            "",
            "## Probability distributions",
            "",
            "Quantiles are computed over seizure probabilities, separated by the true label.",
            "",
            "| Fold | True class | n | Min | 5% | 25% | Median | 75% | 95% | Max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in distributions:
        lines.append(
            f"| {row['fold']} | {row['true_class']} | {row['count']} | "
            f"{row['minimum']:.6g} | {row['q05']:.6g} | {row['q25']:.6g} | "
            f"{row['q50']:.6g} | {row['q75']:.6g} | {row['q95']:.6g} | "
            f"{row['maximum']:.6g} |"
        )
    fold2 = next(row for row in summaries if row["fold"] == "fold_2")
    lines.extend(
        [
            "",
            "## Fold 2 false-negative ranking context",
            "",
            f"Fold 2 has {fold2['false_negative_count']} false negatives. Their seizure "
            f"probabilities range from {fold2['false_negative_probability_minimum']:.6g} "
            f"to {fold2['false_negative_probability_maximum']:.6g}, with median "
            f"{fold2['false_negative_probability_median']:.6g}. The highest false-negative "
            f"probability is {fold2['false_negative_max_distance_below_threshold']:.6g} "
            f"below the existing {threshold:g} threshold.",
            f"Across false-negative/true-negative comparisons, a negative window has "
            f"probability at least as high as a false negative in "
            f"{fold2['negative_windows_with_probability_at_least_fn']:.1%} of comparisons. "
            "This comparison is descriptive of the held-out fold only; it is not used to "
            "change the threshold.",
            "",
            "## Ranking versus fixed-threshold classification",
            "",
            "ROC-AUC and average precision summarize how well scores rank positive windows "
            "above negative windows across score cutoffs. Sensitivity and specificity in "
            "the table are from the single saved 0.5 cutoff. Thus strong ranking metrics "
            "can coexist with missed positives at that operating point (as in Fold 2), "
            "while high sensitivity can coexist with false alarms (as in Fold 1). The "
            "large class imbalance also means average precision should be interpreted "
            "relative to the low seizure prevalence, and neither fold establishes clinical "
            "performance.",
            "",
            "The PR and ROC plots mark the operating point obtained from the saved labels "
            "and threshold; they do not choose a new cutoff.",
            "",
            "## Error tables",
            "",
            f"- False negatives: `{len(false_negatives)}` rows in `false_negatives.csv`.",
            f"- False positives: `{len(false_positives)}` rows in `false_positives.csv`.",
            "- Each row includes patient/recording, window index, recorded start/end times, "
            "true/predicted labels, and predicted seizure probability.",
            "",
            "## Limitations",
            "",
            "There are only two held-out seizure-containing recordings from one patient. "
            "Windows overlap by 50% and are not independent observations. This audit "
            "describes saved predictions, not uncertainty across patients or seizures, and "
            "does not support patient-independent or clinical generalization claims.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/evaluation/random_forest_baseline"),
        help="Directory containing run_configuration.json and fold prediction artifacts.",
    )
    args = parser.parse_args()
    report = analyze(args.output_root)
    print(f"Diagnostics saved under: {report.parent}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
