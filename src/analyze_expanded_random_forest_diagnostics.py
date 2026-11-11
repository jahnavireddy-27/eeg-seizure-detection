#!/usr/bin/env python3
"""Post-hoc diagnostics for saved expanded-cohort out-of-fold predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "expanded_random_forest_baseline"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "chbmit_dataset_manifest.csv"
THRESHOLD = 0.5
IDENTITY = ("patient_id", "recording_id", "relative_path")
OUTCOMES = ("TN", "FP", "FN", "TP")
PREDICTION_FIELDS = {
    *IDENTITY,
    "recording_filename",
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "test_fold",
    "y_true",
    "y_pred",
    "probability_non_seizure",
    "probability_seizure",
}
MANIFEST_FIELDS = {
    *IDENTITY,
    "recording_duration_seconds",
    "seizure_status",
    "summary_seizure_intervals",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def calculate_diagnostic_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Calculate fixed-threshold binary metrics; undefined denominators stay null."""
    truth = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(y_pred, dtype=np.int8)
    if truth.ndim != 1 or truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must be non-empty, equal-length vectors.")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels and predictions must contain only 0 and 1.")
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be finite and positive when provided.")
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    )
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    fpr = _safe_ratio(fp, fp + tn)
    return {
        "window_count": int(truth.size),
        "non_seizure_windows": int(np.count_nonzero(truth == 0)),
        "seizure_windows": int(np.count_nonzero(truth == 1)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": float((tn + tp) / truth.size),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "false_positive_rate": fpr,
        "recording_duration_seconds": duration_seconds,
        "false_positive_windows_per_recording_hour": (
            float(fp / (duration_seconds / 3600.0))
            if duration_seconds is not None
            else None
        ),
    }


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = PREDICTION_FIELDS - set(frame.columns)
    if missing:
        raise ValueError(f"OOF prediction file lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OOF prediction file is empty.")
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
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("OOF predictions contain missing or non-finite values.")
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise ValueError("OOF ground-truth values must be binary.")
    if not set(frame["y_pred"].unique()).issubset({0, 1}):
        raise ValueError("OOF prediction values must be binary.")
    if (frame["end_time_seconds"] <= frame["start_time_seconds"]).any():
        raise ValueError("OOF predictions contain invalid time bounds.")
    probabilities = frame[["probability_non_seizure", "probability_seizure"]].to_numpy(
        dtype=np.float64
    )
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("OOF probabilities must be within [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Class probabilities do not sum to one.")
    implied = (frame["probability_seizure"].to_numpy() >= THRESHOLD).astype(np.int8)
    if not np.array_equal(implied, frame["y_pred"].to_numpy(dtype=np.int8)):
        raise ValueError("Saved y_pred does not match the original 0.5 threshold.")
    window_key = [*IDENTITY, "window_index"]
    if frame.duplicated(window_key).any():
        raise ValueError("OOF data contains duplicate composite recording/window rows.")
    return frame.sort_values([*IDENTITY, "window_index"], kind="stable").reset_index(
        drop=True
    )


def _read_manifest(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    frame = pd.read_csv(path, dtype={name: str for name in IDENTITY})
    missing = MANIFEST_FIELDS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset manifest lacks columns: {sorted(missing)}")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        identity = tuple(str(row[column]) for column in IDENTITY)
        if identity in result:
            raise ValueError(f"Duplicate manifest identity: {identity}")
        result[identity] = row
    return result


def _parse_events(row: dict[str, Any], identity: tuple[str, str, str]) -> list[dict[str, Any]]:
    status = str(row["seizure_status"])
    try:
        intervals = json.loads(row["summary_seizure_intervals"] or "[]")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid summary intervals for {identity}: {exc}") from exc
    events = []
    for index, interval in enumerate(intervals, start=1):
        start = interval.get("start_seconds")
        end = interval.get("end_seconds")
        if start is None or end is None:
            raise ValueError(f"Incomplete seizure interval for {identity} event {index}.")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid seizure interval for {identity} event {index}.")
        events.append(
            {
                "event_id": f"{identity[1]}_event_{index:02d}",
                "patient_id": identity[0],
                "recording_id": identity[1],
                "relative_path": identity[2],
                "event_start_seconds": start,
                "event_end_seconds": end,
                "event_duration_seconds": end - start,
            }
        )
    if status == "no_seizures_annotated" and events:
        raise ValueError(f"Manifest marks {identity} as seizure-free but has intervals.")
    if status == "seizures_annotated" and not events:
        raise ValueError(f"Manifest marks {identity} as seizure-containing without intervals.")
    if status not in {"no_seizures_annotated", "seizures_annotated"}:
        raise ValueError(f"Annotations are not verified for {identity}: {status}")
    return events


def _overlaps(start: float, end: float, event_start: float, event_end: float) -> bool:
    """Positive-duration overlap for half-open time intervals."""
    return start < event_end and end > event_start


def match_seizure_events(
    events: Sequence[dict[str, Any]], predictions: pd.DataFrame
) -> list[dict[str, Any]]:
    """Mark an event detected if a positive OOF window overlaps its interval."""
    results: list[dict[str, Any]] = []
    for event in events:
        same_recording = predictions.loc[
            (predictions["patient_id"].astype(str) == str(event["patient_id"]))
            & (predictions["recording_id"].astype(str) == str(event["recording_id"]))
            & (predictions["relative_path"].astype(str) == str(event["relative_path"]))
        ]
        overlap = same_recording.loc[
            same_recording.apply(
                lambda row: _overlaps(
                    float(row["start_time_seconds"]),
                    float(row["end_time_seconds"]),
                    float(event["event_start_seconds"]),
                    float(event["event_end_seconds"]),
                ),
                axis=1,
            )
        ]
        positive_truth = overlap.loc[overlap["y_true"] == 1]
        false_negative = positive_truth.loc[positive_truth["y_pred"] == 0]
        positive_prediction = overlap.loc[overlap["y_pred"] == 1]
        result = dict(event)
        result.update(
            {
                "overlapping_ground_truth_windows": int(len(positive_truth)),
                "true_positive_windows": int((positive_truth["y_pred"] == 1).sum()),
                "false_negative_windows": int(len(false_negative)),
                "predicted_positive_overlapping_windows": int(len(positive_prediction)),
                "detected": bool(not positive_prediction.empty),
                "event_detection_definition": (
                    "at least one saved y_pred=1 window has positive-duration overlap"
                ),
            }
        )
        results.append(result)
    return results


def _validate_annotations(
    frame: pd.DataFrame,
    manifest: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str, str], float]]:
    identities = {
        tuple(str(row[column]) for column in IDENTITY)
        for row in frame[list(IDENTITY)].to_dict("records")
    }
    missing = identities - set(manifest)
    if missing:
        raise ValueError(f"OOF recordings are missing from manifest: {sorted(missing)}")
    events_by_recording: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    durations: dict[tuple[str, str, str], float] = {}
    for identity in identities:
        row = manifest[identity]
        events = _parse_events(row, identity)
        duration = float(row["recording_duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid EDF duration for {identity}.")
        events_by_recording[identity] = events
        durations[identity] = duration
        group = frame.loc[
            (frame["patient_id"].astype(str) == identity[0])
            & (frame["recording_id"].astype(str) == identity[1])
            & (frame["relative_path"].astype(str) == identity[2])
        ]
        intervals = events
        expected = np.asarray(
            [
                int(
                    any(
                        _overlaps(
                            float(item.start_time_seconds),
                            float(item.end_time_seconds),
                            event["event_start_seconds"],
                            event["event_end_seconds"],
                        )
                        for event in intervals
                    )
                )
                for item in group.itertuples(index=False)
            ],
            dtype=np.int8,
        )
        actual = group["y_true"].to_numpy(dtype=np.int8)
        if not np.array_equal(expected, actual):
            raise ValueError(
                f"OOF y_true labels do not match manifest seizure intervals for {identity}."
            )
    return events_by_recording, durations


def _outcome(frame: pd.DataFrame) -> pd.Series:
    names = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    return pd.Series(
        [names[(int(truth), int(pred))] for truth, pred in zip(frame.y_true, frame.y_pred)],
        index=frame.index,
        name="outcome",
    )


def _error_tables(
    frame: pd.DataFrame,
    durations: dict[tuple[str, str, str], float],
    events_by_recording: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    work = frame.copy()
    work["outcome"] = _outcome(work)
    work["false_positive"] = (work["outcome"] == "FP").astype(int)
    work["false_negative"] = (work["outcome"] == "FN").astype(int)
    group_columns = list(IDENTITY)
    recording_rows = []
    patient_rows = []
    for identity, group in work.groupby(group_columns, sort=True):
        identity = tuple(str(value) for value in identity)
        metrics = calculate_diagnostic_metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            durations[identity],
        )
        event_matches = match_seizure_events(events_by_recording[identity], group)
        recording_rows.append(
            {
                **dict(zip(IDENTITY, identity)),
                "patient_id": identity[0],
                "test_fold": str(group["test_fold"].iloc[0]),
                "seizure_event_count": len(event_matches),
                "detected_events": sum(item["detected"] for item in event_matches),
                "missed_events": sum(not item["detected"] for item in event_matches),
                **metrics,
            }
        )
    recording_frame = pd.DataFrame(recording_rows)
    for patient_id, group in work.groupby("patient_id", sort=True):
        patient_recordings = recording_frame.loc[
            recording_frame["patient_id"] == patient_id
        ]
        metrics = calculate_diagnostic_metrics(
            group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        )
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "recording_count": int(group["recording_id"].nunique()),
                "seizure_recordings": int(
                    (patient_recordings["seizure_event_count"] > 0).sum()
                ),
                "false_positive_windows": int(group["false_positive"].sum()),
                "false_negative_windows": int(group["false_negative"].sum()),
                **metrics,
            }
        )
    patient_frame = pd.DataFrame(patient_rows)

    error_counts = (
        work.groupby(["patient_id", "outcome"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(OUTCOMES), fill_value=0)
        .reset_index()
    )
    recording_errors = (
        work.groupby([*group_columns, "outcome"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(OUTCOMES), fill_value=0)
        .reset_index()
    )

    event_rows = []
    for identity, events in events_by_recording.items():
        group = work.loc[
            (work["patient_id"].astype(str) == identity[0])
            & (work["recording_id"].astype(str) == identity[1])
            & (work["relative_path"].astype(str) == identity[2])
        ]
        event_rows.extend(match_seizure_events(events, group))
    event_frame = pd.DataFrame(event_rows)

    fp = work.loc[work["outcome"] == "FP"].sort_values(
        [*group_columns, "window_index"], kind="stable"
    )
    cluster_rows = []
    for identity, group in fp.groupby(group_columns, sort=True):
        rows = list(group.sort_values("window_index").to_dict("records"))
        clusters: list[list[dict[str, Any]]] = []
        for row in rows:
            if (
                not clusters
                or int(row["window_index"]) != int(clusters[-1][-1]["window_index"]) + 1
            ):
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for cluster_index, cluster in enumerate(clusters, start=1):
            start = min(float(row["start_time_seconds"]) for row in cluster)
            end = max(float(row["end_time_seconds"]) for row in cluster)
            cluster_rows.append(
                {
                    **dict(zip(IDENTITY, (str(value) for value in identity))),
                    "cluster_id": f"{identity[1]}_fp_cluster_{cluster_index:02d}",
                    "window_count": len(cluster),
                    "classification": "isolated" if len(cluster) == 1 else "consecutive_cluster",
                    "first_window_index": int(cluster[0]["window_index"]),
                    "last_window_index": int(cluster[-1]["window_index"]),
                    "start_time_seconds": start,
                    "end_time_seconds": end,
                    "time_span_seconds": end - start,
                    "probability_minimum": min(float(row["probability_seizure"]) for row in cluster),
                    "probability_maximum": max(float(row["probability_seizure"]) for row in cluster),
                }
            )
    cluster_frame = pd.DataFrame(cluster_rows)
    false_negative_frame = work.loc[work["outcome"] == "FN"].copy()
    false_positive_frame = fp.copy()
    false_negative_frame = false_negative_frame.drop(
        columns=["false_positive", "false_negative"]
    )
    false_positive_frame = false_positive_frame.drop(
        columns=["false_positive", "false_negative"]
    )
    return (
        error_counts,
        recording_errors,
        recording_frame,
        patient_frame,
        event_frame,
        cluster_frame,
        false_negative_frame,
        false_positive_frame,
    )


def _probability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["outcome"] = _outcome(work)
    rows = []
    for outcome in OUTCOMES:
        values = work.loc[work["outcome"] == outcome, "probability_seizure"].to_numpy()
        rows.append(
            {
                "outcome": outcome,
                "count": int(len(values)),
                "minimum": float(values.min()) if len(values) else None,
                "q05": float(np.quantile(values, 0.05)) if len(values) else None,
                "q25": float(np.quantile(values, 0.25)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "q75": float(np.quantile(values, 0.75)) if len(values) else None,
                "q95": float(np.quantile(values, 0.95)) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
            }
        )
    return pd.DataFrame(rows)


def _fold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in frame.groupby("test_fold", sort=True):
        metrics = calculate_diagnostic_metrics(group.y_true, group.y_pred)
        y_true = group.y_true.to_numpy(dtype=np.int8)
        probability = group.probability_seizure.to_numpy(dtype=np.float64)
        metrics.update(
            {
                "fold": str(fold),
                "test_recordings": int(group["recording_id"].nunique()),
                "recording_ids": json.dumps(
                    sorted(
                        set(zip(group.patient_id.astype(str), group.recording_id.astype(str)))
                    ),
                    separators=(",", ":"),
                ),
                "roc_auc": float(roc_auc_score(y_true, probability)),
                "average_precision": float(average_precision_score(y_true, probability)),
                "seizure_recordings": int(
                    group.groupby(["patient_id", "recording_id", "relative_path"])
                    .y_true.max().sum()
                ),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _save_figures(
    frame: pd.DataFrame,
    recording_metrics: pd.DataFrame,
    events: pd.DataFrame,
    manifest: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    y_true = frame.y_true.to_numpy(dtype=np.int8)
    y_pred = frame.y_pred.to_numpy(dtype=np.int8)
    probability = frame.probability_seizure.to_numpy(dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    matrix = np.asarray([[tn, fp], [fn, tp]])
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["non-seizure", "seizure"],
        yticklabels=["non-seizure", "seizure"],
        xlabel="Predicted class", ylabel="Ground-truth class",
        title=f"Pooled OOF confusion matrix (threshold {THRESHOLD:g})",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probability)
    metric = average_precision_score(y_true, probability)
    operating_precision = _safe_ratio(tp, tp + fp)
    operating_recall = _safe_ratio(tp, tp + fn)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.step(recall, precision, where="post", label=f"PR curve (AP={metric:.3f})")
    ax.scatter([operating_recall], [operating_precision], color="crimson", zorder=3,
               label=f"Saved threshold {THRESHOLD:g}")
    ax.axhline(float(np.mean(y_true)), color="gray", linestyle="--", label="Seizure prevalence")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Recall (sensitivity)", ylabel="Precision",
           title="Pooled out-of-fold precision-recall curve")
    ax.grid(alpha=0.25); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=180); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probability)
    auc = roc_auc_score(y_true, probability)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.scatter([_safe_ratio(fp, fp + tn)], [operating_recall], color="crimson", zorder=3,
               label=f"Saved threshold {THRESHOLD:g}")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="False-positive rate (1 − specificity)",
           ylabel="True-positive rate (sensitivity)", title="Pooled out-of-fold ROC curve")
    ax.grid(alpha=0.25); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180); plt.close(fig)

    outcome = _outcome(frame)
    values = [frame.loc[outcome == name, "probability_seizure"].to_numpy() for name in OUTCOMES]
    labels = [f"{name}\nn={len(items):,}" for name, items in zip(OUTCOMES, values)]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    box = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    palette = {"TN": "#4C78A8", "FP": "#F58518", "FN": "#E45756", "TP": "#54A24B"}
    for patch, name in zip(box["boxes"], OUTCOMES):
        patch.set_facecolor(palette[name]); patch.set_alpha(0.75)
    ax.axhline(THRESHOLD, color="black", linestyle="--", label="Threshold 0.5")
    ax.set(ylabel="Predicted seizure probability", title="OOF probabilities by classification outcome",
           ylim=(-0.03, 1.03)); ax.grid(axis="y", alpha=0.2); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "probability_by_outcome.png", dpi=180); plt.close(fig)

    ordered = recording_metrics.sort_values(["patient_id", "recording_id"])
    names = [f"{p} · {r}" for p, r in zip(ordered.patient_id, ordered.recording_id)]
    y = np.arange(len(ordered))
    fig, (ax_sens, ax_fpr) = plt.subplots(1, 2, figsize=(13, max(7.5, len(ordered) * 0.34)), sharey=True)
    sensitivity = pd.to_numeric(ordered.sensitivity, errors="coerce").to_numpy(dtype=float)
    fpr_values = pd.to_numeric(ordered.false_positive_rate, errors="coerce").to_numpy(dtype=float)
    colors = {"chb01": "#4C78A8", "chb02": "#F58518", "chb03": "#54A24B"}
    for index, patient in enumerate(ordered.patient_id):
        ax_sens.barh(index, sensitivity[index] if np.isfinite(sensitivity[index]) else 0,
                     color=colors.get(patient, "gray"), alpha=0.8)
        ax_fpr.barh(index, fpr_values[index], color=colors.get(patient, "gray"), alpha=0.8)
    ax_sens.set(yticks=y, yticklabels=names, xlim=(0, 1.04), xlabel="Sensitivity",
                title="Per-recording sensitivity\n(blank means no seizure windows)")
    ax_fpr.set(xlim=(0, 1.04), xlabel="False-positive rate", title="Per-recording false-positive rate")
    for ax in (ax_sens, ax_fpr):
        ax.grid(axis="x", alpha=0.2); ax.invert_yaxis()
    fig.suptitle("Recording-level fixed-threshold diagnostics by patient")
    fig.tight_layout(); fig.savefig(output_dir / "recording_sensitivity_false_positive_rate.png", dpi=180); plt.close(fig)

    for patient_id in sorted(frame.patient_id.astype(str).unique()):
        patient_frame = frame.loc[frame.patient_id.astype(str) == patient_id]
        recordings = sorted(
            set(zip(patient_frame.recording_id.astype(str), patient_frame.relative_path.astype(str)))
        )
        fig, axes = plt.subplots(len(recordings), 1, figsize=(13, max(5.5, 1.55 * len(recordings))), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (recording_id, relative_path) in zip(axes, recordings):
            group = patient_frame.loc[
                (patient_frame.recording_id.astype(str) == recording_id)
                & (patient_frame.relative_path.astype(str) == relative_path)
            ]
            identity = (patient_id, recording_id, relative_path)
            manifest_row = manifest[identity]
            duration = float(manifest_row["recording_duration_seconds"])
            intervals = json.loads(manifest_row["summary_seizure_intervals"] or "[]")
            for event in intervals:
                start, end = float(event["start_seconds"]), float(event["end_seconds"])
                ax.axvspan(start, end, color="#E45756", alpha=0.28)
            positive = group.loc[group.y_pred == 1]
            for row in positive.itertuples(index=False):
                ax.broken_barh(
                    [(float(row.start_time_seconds), float(row.end_time_seconds - row.start_time_seconds))],
                    (0.37, 0.26), facecolors="#4C78A8", alpha=0.8,
                )
            event_rows = events.loc[
                (events.patient_id == patient_id) & (events.recording_id == recording_id)
            ]
            detected = int(event_rows.detected.sum()) if not event_rows.empty else 0
            total = len(intervals)
            ax.set(yticks=[], ylim=(0, 1), xlim=(0, duration),
                   ylabel=f"{recording_id}\n{detected}/{total} events")
            ax.grid(axis="x", alpha=0.16)
        axes[-1].set_xlabel("Recording-relative time (seconds)")
        from matplotlib.patches import Patch
        fig.legend(
            handles=[Patch(facecolor="#E45756", alpha=0.35, label="Summary seizure interval"),
                     Patch(facecolor="#4C78A8", alpha=0.8, label="Predicted-positive window (0.5 threshold)")],
            loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.005), frameon=False,
        )
        fig.suptitle(f"{patient_id}: annotation intervals and OOF positive windows", y=1.025)
        fig.tight_layout()
        fig.savefig(output_dir / f"seizure_timeline_{patient_id}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _write_report(
    path: Path,
    frame: pd.DataFrame,
    pooled: dict[str, Any],
    patient_errors: pd.DataFrame,
    recording_metrics: pd.DataFrame,
    events: pd.DataFrame,
    clusters: pd.DataFrame,
    folds: pd.DataFrame,
    probability: pd.DataFrame,
    input_hashes: dict[str, str],
) -> None:
    lines = [
        "# Expanded Random Forest: post-hoc OOF error analysis",
        "",
        "## Scope",
        "",
        "This report uses only the saved four-fold out-of-fold predictions, their ground-truth window labels and times, and the existing manifest's summary intervals and EDF durations. No signal samples were read; the model was not retrained; the 0.5 decision threshold was not changed.",
        "Event-level diagnostic definition: a summary seizure event is counted as detected if at least one saved predicted-positive window has positive-duration overlap with the event's annotated interval. An event with no such window is counted as missed. This is a descriptive matching rule, not a clinically validated alarm criterion. No detection latency or clinical utility is inferred.",
        "False positives are grouped by patient, recording, and consecutive-window clusters. They are not assigned to seizure events because their ground-truth window label is non-seizure and annotation overlap was checked against the manifest.",
        "",
        f"Pooled confusion matrix [[TN, FP], [FN, TP]]: `[[{pooled['true_negative']}, {pooled['false_positive']}], [{pooled['false_negative']}, {pooled['true_positive']}]]`.",
        f"Windows: {len(frame):,}; true seizure windows: {int(frame.y_true.sum()):,}; prevalence: {frame.y_true.mean():.3%}.",
        "",
        "## Errors by patient",
        "",
        "| Patient | Windows | FP | FN | TP | TN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in patient_errors.to_dict("records"):
        lines.append(
            f"| {row['patient_id']} | {row.get('TN', 0)+row.get('FP', 0)+row.get('FN', 0)+row.get('TP', 0)} | {row.get('FP', 0)} | {row.get('FN', 0)} | {row.get('TP', 0)} | {row.get('TN', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Seizure event outcomes and false negatives",
            "",
            f"Detected events: {int(events.detected.sum())}/{len(events)}; missed events: {int((~events.detected).sum())}.",
            "| Patient | Recording | Event | Interval (s) | Positive windows | FN windows | Status |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in events.to_dict("records"):
        status = "detected" if row["detected"] else "missed"
        lines.append(
            f"| {row['patient_id']} | {row['recording_id']} | {row['event_id']} | "
            f"{row['event_start_seconds']:g}–{row['event_end_seconds']:g} | "
            f"{row['overlapping_ground_truth_windows']} | {row['false_negative_windows']} | {status} |"
        )
    event_fn = events.sort_values(["false_negative_windows", "patient_id"], ascending=[False, True]).head(3)
    if not event_fn.empty:
        lines.extend(
            [
                "",
                "False negatives by event interval:",
                "",
            ]
        )
        lines.extend(
            f"- `{row.recording_id}` {row.event_start_seconds:g}–{row.event_end_seconds:g} s: "
            f"{row.false_negative_windows}/{row.overlapping_ground_truth_windows} positive windows missed."
            for row in event_fn.itertuples(index=False)
        )
    lines.extend(
        [
            "",
            "## Probability distributions by outcome",
            "",
            "| Outcome | Count | Min | Q05 | Q25 | Median | Q75 | Q95 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in probability.to_dict("records"):
        vals = ["—" if pd.isna(row[name]) else f"{row[name]:.4f}" for name in ("minimum", "q05", "q25", "median", "q75", "q95", "maximum")]
        lines.append(f"| {row['outcome']} | {row['count']} | " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## False-positive clusters",
            "",
            f"False-positive windows: {int((frame.y_true.eq(0) & frame.y_pred.eq(1)).sum())}; "
            f"clusters: {len(clusters)}; isolated windows: {int((clusters.window_count == 1).sum()) if not clusters.empty else 0}; "
            f"multi-window clusters: {int((clusters.window_count > 1).sum()) if not clusters.empty else 0}.",
            "A cluster means consecutive false-positive window indices within one recording; windows overlap by the configured 50% stride. The listed time span is from the first window start through the last window end.",
            "",
            "## Per-recording metrics and fold variation",
            "",
            "False-positive windows per recording-hour use manifest EDF duration; this is a window rate, not an alarm-episode rate. Sensitivity is not defined for recordings with no positive ground-truth windows; such values are blank.",
            "| Fold and test recordings | Recording count | Windows | FP | FN | Sensitivity | Specificity | ROC-AUC | AP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in folds.to_dict("records"):
        lines.append(
            f"| {row['fold']} ({row['recording_ids']}) | {row['test_recordings']} | {row['window_count']} | {row['false_positive']} | {row['false_negative']} | "
            f"{_fmt(row['sensitivity'])} | {_fmt(row['specificity'])} | {row['roc_auc']:.4f} | {row['average_precision']:.4f} |"
        )
    lines.extend(["", "Fold-specific error sources:"])
    for fold_name in sorted(recording_metrics.test_fold.unique()):
        fold_records = recording_metrics.loc[recording_metrics.test_fold == fold_name]
        fn_sources = fold_records.loc[fold_records.false_negative > 0].sort_values(
            "false_negative", ascending=False
        )
        fp_sources = fold_records.loc[fold_records.false_positive > 0].sort_values(
            "false_positive", ascending=False
        )
        fn_text = ", ".join(
            f"{row.recording_id} ({row.false_negative} FN)"
            for row in fn_sources.itertuples(index=False)
        ) or "none"
        fp_text = ", ".join(
            f"{row.recording_id} ({row.false_positive} FP windows)"
            for row in fp_sources.itertuples(index=False)
        ) or "none"
        lines.append(f"- {fold_name}: FN sources: {fn_text}; FP sources: {fp_text}.")
    top_fns = recording_metrics.sort_values(["false_negative", "recording_id"], ascending=[False, True]).head(4)
    top_fps = recording_metrics.sort_values(["false_positive", "recording_id"], ascending=[False, True]).head(4)
    lines.extend(["", "Largest false-negative contributors:"])
    lines.extend(
        f"- `{row.patient_id}/{row.recording_id}`: {row.false_negative} FN / {row.seizure_windows} positive windows; sensitivity {_fmt(row.sensitivity)}."
        for row in top_fns.itertuples(index=False)
        if row.seizure_windows > 0
    )
    lines.extend(["", "Largest false-positive contributors:"])
    lines.extend(
        f"- `{row.patient_id}/{row.recording_id}`: {row.false_positive} FP windows; {row.false_positive_windows_per_recording_hour:.2f} FP windows per recording-hour."
        for row in top_fps.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "Across folds, false-negative counts, sensitivity, and average precision are compared above. Fold differences reflect the particular held-out recordings, including their seizure event durations and positive-window counts; they are not independent patient-level estimates.",
            "",
            "## Inputs and limitations",
            "",
        ]
    )
    lines.extend(f"- `{name}` SHA-256: `{digest}`" for name, digest in input_hashes.items())
    lines.extend(
        [
            "",
            "The event rule above is a diagnostic overlap definition only. These window-level results use overlapping windows and a highly imbalanced target; they do not define alarm timing, latency, acceptable false-alarm burden, or clinical utility. The evaluation is recording-held-out with patient overlap, not patient-independent validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.4f}"


def analyze(
    evaluation_dir: Path = DEFAULT_EVALUATION_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Path:
    evaluation_dir = Path(evaluation_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = evaluation_dir / "diagnostics"
    config_path = evaluation_dir / "run_configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if float(config["random_forest"]["probability_threshold"]) != THRESHOLD:
        raise ValueError("Saved baseline threshold is not 0.5; diagnostics require the original threshold.")
    prediction_path = evaluation_dir / "out_of_fold_predictions.csv"
    frame = _load_predictions(prediction_path)
    manifest = _read_manifest(manifest_path)
    events_by_recording, durations = _validate_annotations(frame, manifest)
    all_events = [event for events in events_by_recording.values() for event in events]
    event_frame = pd.DataFrame(match_seizure_events(all_events, frame))
    tables = _error_tables(frame, durations, events_by_recording)
    patient_errors, recording_errors, recording_metrics, patient_metrics, _, clusters, false_negatives, false_positives = tables
    if not false_negatives.empty:
        event_map: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for event in all_events:
            identity = (event["patient_id"], event["recording_id"], event["relative_path"])
            event_map.setdefault(identity, []).append(event)
        overlapping_ids = []
        for row in false_negatives.itertuples(index=False):
            identity = (str(row.patient_id), str(row.recording_id), str(row.relative_path))
            ids = [
                event["event_id"]
                for event in event_map[identity]
                if _overlaps(
                    float(row.start_time_seconds),
                    float(row.end_time_seconds),
                    float(event["event_start_seconds"]),
                    float(event["event_end_seconds"]),
                )
            ]
            overlapping_ids.append(json.dumps(ids, separators=(",", ":")))
        false_negatives["overlapping_event_ids"] = overlapping_ids
    probability = _probability_summary(frame)
    folds = _fold_summary(frame)
    pooled = calculate_diagnostic_metrics(frame.y_true, frame.y_pred)
    pooled["roc_auc"] = float(roc_auc_score(frame.y_true, frame.probability_seizure))
    pooled["average_precision"] = float(average_precision_score(frame.y_true, frame.probability_seizure))
    pooled["confusion_matrix"] = confusion_matrix(frame.y_true, frame.y_pred, labels=[0, 1]).tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    patient_errors.to_csv(output_dir / "errors_by_patient.csv", index=False)
    recording_errors.to_csv(output_dir / "errors_by_recording.csv", index=False)
    recording_metrics.to_csv(output_dir / "per_recording_metrics.csv", index=False)
    patient_metrics.to_csv(output_dir / "per_patient_metrics.csv", index=False)
    event_frame.to_csv(output_dir / "seizure_event_detection.csv", index=False)
    clusters.to_csv(output_dir / "false_positive_clusters.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negative_windows.csv", index=False)
    false_positives.to_csv(output_dir / "false_positive_windows.csv", index=False)
    probability.to_csv(output_dir / "probability_by_outcome_summary.csv", index=False)
    folds.to_csv(output_dir / "fold_comparison.csv", index=False)
    (output_dir / "pooled_metrics.json").write_text(
        json.dumps(pooled, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    input_hashes = {
        "out_of_fold_predictions.csv": _sha256(prediction_path),
        "run_configuration.json": _sha256(config_path),
        (
            str(manifest_path.relative_to(PROJECT_ROOT))
            if manifest_path.is_relative_to(PROJECT_ROOT)
            else str(manifest_path)
        ): _sha256(manifest_path),
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        prior_hashes_path = output_dir / "input_hashes.json"
        report_exists = (output_dir / "analysis_report.md").exists()
        if report_exists or not prior_hashes_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite existing diagnostics: {output_dir}"
            )
        prior_hashes = json.loads(prior_hashes_path.read_text(encoding="utf-8"))
        if prior_hashes != input_hashes:
            raise FileExistsError(
                "Partial diagnostics belong to different input artifacts; refusing overwrite."
            )
    (output_dir / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _save_figures(frame, recording_metrics, event_frame, manifest, output_dir)
    report_path = output_dir / "analysis_report.md"
    _write_report(
        report_path, frame, pooled, patient_errors, recording_metrics, event_frame,
        clusters, folds, probability, input_hashes,
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        report = analyze(args.evaluation_dir, args.manifest)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Expanded OOF diagnostics failed: {exc}", file=sys.stderr)
        return 1
    print(f"Diagnostics saved under: {report.parent}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
