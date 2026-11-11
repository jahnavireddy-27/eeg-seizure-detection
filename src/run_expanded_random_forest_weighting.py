#!/usr/bin/env python3
"""Compare train-fold-only event and recording weighting against saved RF OOF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import analyze_expanded_random_forest_diagnostics as diagnostics
import train_expanded_random_forest as expanded
import train_random_forest as baseline

BASELINE_DIR = PROJECT_ROOT / "data/processed/evaluation/expanded_random_forest_baseline"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/evaluation/expanded_random_forest_weighting_experiment"
FEATURE_MATRIX = PROJECT_ROOT / "data/processed/assembled_features/chbmit_feature_dataset.csv"
MANIFEST = PROJECT_ROOT / "results/chbmit_dataset_manifest.csv"
IDENTITY = expanded.IDENTITY_COLUMNS
KEY = (*IDENTITY, "window_index")


def event_balanced_weights(
    fold: pd.DataFrame,
    labels: np.ndarray,
    events_by_recording: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Equalize positive class mass across events; negative windows are uniform."""
    labels = np.asarray(labels, dtype=np.int8)
    if len(fold) != len(labels) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Fold and binary labels must align and contain both classes.")
    weights = np.zeros(len(labels), dtype=np.float64)
    event_windows: dict[str, dict[int, float]] = {}
    for index, row in enumerate(fold.itertuples(index=False)):
        if labels[index] == 0:
            continue
        identity = tuple(str(getattr(row, name)) for name in IDENTITY)
        if identity not in events_by_recording:
            raise ValueError(f"No summary event mapping for positive window: {identity}")
        matches = [
            event for event in events_by_recording[identity]
            if diagnostics._overlaps(
                float(row.start_time_seconds), float(row.end_time_seconds),
                float(event["event_start_seconds"]), float(event["event_end_seconds"]),
            )
        ]
        if not matches:
            raise ValueError(f"Positive window has no summary event match: {identity}")
        for event in matches:
            event_windows.setdefault(str(event["event_id"]), {})[index] = 1.0 / len(matches)
    if not event_windows:
        raise ValueError("Training fold has no seizure event windows.")
    # Divide each window's contribution across all matched events.
    weights[labels == 1] = 0.0
    event_raw_masses = {event_id: sum(contributions.values()) for event_id, contributions in event_windows.items()}
    for event_id, contributions in event_windows.items():
        for index, contribution in contributions.items():
            weights[index] += contribution / (len(event_windows) * event_raw_masses[event_id])
    weights[labels == 0] = 1.0 / int(np.count_nonzero(labels == 0))
    weights[labels == 1] *= 0.5 / weights[labels == 1].sum()
    weights[labels == 0] *= 0.5 / weights[labels == 0].sum()
    weights *= len(labels)
    rows = []
    positive_mass = weights[labels == 1].sum()
    for event_id, contributions in sorted(event_windows.items()):
        unique = sorted(contributions)
        rows.append({"event_id": event_id, "positive_windows": len(unique),
                     "event_weight_sum": float(weights[unique].sum()),
                     "positive_class_weight_fraction": float(weights[unique].sum() / positive_mass)})
    return weights, rows


def recording_balanced_weights(fold: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Equalize class mass, then divide each class equally across recordings."""
    labels = np.asarray(labels, dtype=np.int8)
    if len(fold) != len(labels) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Fold and binary labels must align and contain both classes.")
    weights = np.zeros(len(labels), dtype=np.float64)
    identities = [tuple(str(row[name]) for name in IDENTITY) for row in fold.to_dict("records")]
    for label in (0, 1):
        recs = sorted({identity for identity, target in zip(identities, labels) if target == label})
        if not recs:
            raise ValueError(f"Training fold has no class {label} recordings.")
        for identity in recs:
            positions = [i for i, (item, target) in enumerate(zip(identities, labels))
                         if item == identity and target == label]
            weights[positions] = 1.0 / (len(recs) * len(positions))
        weights[labels == label] *= 0.5 / weights[labels == label].sum()
    return weights * len(labels)


def _metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    return baseline.calculate_metrics(y, pred, prob)


def run_experiment(
    feature_path: Path = FEATURE_MATRIX,
    manifest_path: Path = MANIFEST,
    baseline_dir: Path = BASELINE_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    data, features = baseline.load_feature_matrix(Path(feature_path))
    labels = data.seizure_label.map(baseline.LABEL_TO_INT).to_numpy(dtype=np.int8)
    folds = expanded.make_recording_grouped_folds(data, 4, 42)
    manifest = diagnostics._read_manifest(Path(manifest_path))
    train_truth = data[list(IDENTITY) + ["window_index", "start_time_seconds", "end_time_seconds"]].copy()
    train_truth["y_true"] = labels
    events_by_recording, _ = diagnostics._validate_annotations(train_truth, manifest)
    reference = pd.read_csv(Path(baseline_dir) / "out_of_fold_predictions.csv")
    required = {*KEY, "test_fold", "y_true", "y_pred", "probability_seizure"}
    if required - set(reference.columns):
        raise ValueError(f"Baseline OOF lacks columns: {sorted(required - set(reference.columns))}")
    reference_keys = reference[list(KEY)].astype(str).apply(tuple, axis=1)
    data_keys = data[list(KEY)].astype(str).apply(tuple, axis=1)
    if reference_keys.duplicated().any() or set(reference_keys) != set(data_keys):
        raise ValueError("Baseline OOF keys do not exactly match assembled feature windows.")
    ref = reference.set_index(list(KEY)).loc[pd.MultiIndex.from_frame(data[list(KEY)])].reset_index()
    if not np.array_equal(ref.y_true.to_numpy(dtype=np.int8), labels):
        raise ValueError("Baseline OOF labels differ from assembled dataset.")

    # Verify generated folds are identical to the persisted baseline assignment.
    for fold in folds:
        if set(ref.iloc[fold.test_indices].test_fold) != {fold.name}:
            raise ValueError(f"Generated assignment differs from reference in {fold.name}.")

    configuration_doc = json.loads((Path(baseline_dir) / "run_configuration.json").read_text())
    config = baseline.ForestConfiguration(**configuration_doc["random_forest"])
    variant_names = ("event_balanced", "recording_balanced")
    oof_prob = {name: np.full(len(data), np.nan) for name in variant_names}
    weight_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_index, test_index = fold.train_indices, fold.test_indices
        y_train, y_test = labels[train_index], labels[test_index]
        x_train, x_test = data.iloc[train_index][features], data.iloc[test_index][features]
        retained, _excluded = baseline.deduplicate_training_features(x_train)
        train_frame = data.iloc[train_index]
        w_event, event_rows = event_balanced_weights(train_frame, y_train, events_by_recording)
        w_recording = recording_balanced_weights(train_frame, y_train)
        for strategy, sample_weight in ((variant_names[0], w_event), (variant_names[1], w_recording)):
            if not np.isclose(sample_weight[y_train == 0].sum(), len(y_train) / 2) or not np.isclose(sample_weight[y_train == 1].sum(), len(y_train) / 2):
                raise RuntimeError(f"{strategy} class weight masses are unbalanced in {fold.name}.")
            model = baseline.build_random_forest(config)
            model.set_params(class_weight=None)
            model.fit(x_train[retained], y_train, sample_weight=sample_weight)
            prob = model.predict_proba(x_test[retained])[:, 1]
            oof_prob[strategy][test_index] = prob
            fold_rows.append({"strategy": strategy, "fold": fold.name,
                              "train_recordings": len(fold.train_recordings), "test_recordings": len(fold.test_recordings),
                              "train_windows": len(train_index), "test_windows": len(test_index),
                              "train_positive_weight_sum": float(sample_weight[y_train == 1].sum()),
                              "train_negative_weight_sum": float(sample_weight[y_train == 0].sum()),
                              **_metrics(y_test, (prob >= 0.5).astype(np.int8), prob)})
            for cls in (0, 1):
                rec_ids = [tuple(str(row[c]) for c in IDENTITY) for row in train_frame.to_dict("records")]
                for identity in sorted({r for r, lab in zip(rec_ids, y_train) if lab == cls}):
                    idx = np.asarray([i for i, (r, lab) in enumerate(zip(rec_ids, y_train)) if r == identity and lab == cls])
                    weight_rows.append({"fold": fold.name, "strategy": strategy, "weight_class": "seizure" if cls else "non_seizure",
                                        "patient_id": identity[0], "recording_id": identity[1], "relative_path": identity[2],
                                        "windows": int(len(idx)), "weight_sum": float(sample_weight[idx].sum())})
            if strategy == "event_balanced":
                for event in event_rows:
                    weight_rows.append({"fold": fold.name, "strategy": strategy, "weight_class": "event", **event})

    if any(not np.isfinite(values).all() for values in oof_prob.values()):
        raise RuntimeError("Weighted model predictions do not cover all windows exactly once.")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = []
    for strategy, probs in [("baseline_class_balanced", ref.probability_seizure.to_numpy(dtype=float)),
                            *((name, values) for name, values in oof_prob.items())]:
        pred = (probs >= 0.5).astype(np.int8)
        comparison.append({"strategy": strategy, "windows": len(labels), "threshold": 0.5, **_metrics(labels, pred, probs)})
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_comparison.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output_dir / "training_weight_summary.csv", index=False)
    pd.DataFrame(comparison).to_json(output_dir / "pooled_comparison.json", orient="records", indent=2)
    pred_frame = ref[list(KEY) + ["test_fold", "y_true"]].copy()
    for strategy, probs in oof_prob.items():
        pred_frame[f"{strategy}_probability_seizure"] = probs
        pred_frame[f"{strategy}_y_pred"] = (probs >= 0.5).astype(np.int8)
    pred_frame.to_csv(output_dir / "weighted_oof_predictions.csv", index=False, float_format="%.17g")

    # Per-recording diagnostics make event and long-recording changes interpretable.
    rows = []
    for identity, indices in data.groupby(list(IDENTITY), sort=True).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        rows.append({"patient_id": identity[0], "recording_id": identity[1], "relative_path": identity[2],
                     "test_fold": str(ref.iloc[idx[0]].test_fold), "windows": len(idx),
                     "actual_seizure_windows": int(labels[idx].sum()),
                     **{f"{key}_{strategy}": value for strategy, probs in [("baseline", ref.probability_seizure.to_numpy(dtype=float)), *oof_prob.items()]
                        for key, value in {"sensitivity": diagnostics.calculate_diagnostic_metrics(labels[idx], (probs[idx] >= .5).astype(np.int8))["sensitivity"],
                                           "false_positives": diagnostics.calculate_diagnostic_metrics(labels[idx], (probs[idx] >= .5).astype(np.int8))["false_positive"]}.items()}})
    pd.DataFrame(rows).to_csv(output_dir / "per_recording_comparison.csv", index=False)
    # Event-level matching uses exactly the same descriptive any-overlap rule.
    event_rows = []
    for strategy, probs in [("baseline_class_balanced", ref.probability_seizure.to_numpy(dtype=float)), *oof_prob.items()]:
        pred_frame_event = ref[list(IDENTITY) + ["start_time_seconds", "end_time_seconds"]].copy()
        pred_frame_event["y_true"] = labels
        pred_frame_event["y_pred"] = (probs >= .5).astype(np.int8)
        for events in events_by_recording.values():
            for result in diagnostics.match_seizure_events(events, pred_frame_event):
                event_rows.append({"strategy": strategy, **result})
    pd.DataFrame(event_rows).to_csv(output_dir / "event_detection_comparison.csv", index=False)
    config_out = {"reference": str(Path(baseline_dir) / "out_of_fold_predictions.csv"),
                  "features": str(feature_path), "folds_reused": True, "fold_seed": 42, "n_splits": 4,
                  "threshold": 0.5, "reference_strategy": "class_weight=balanced saved OOF (not retrained)",
                  "model_parameters": configuration_doc["random_forest"], "weighted_variants": {
                      "event_balanced": "class_weight=None; equal positive weight mass per summary event, uniform negative windows; equal total class masses",
                      "recording_balanced": "class_weight=None; equal weight mass per recording within each class; equal total class masses"},
                  "validation_note": "recording-held-out with patient overlap; not patient-independent validation"}
    (output_dir / "run_configuration.json").write_text(json.dumps(config_out, indent=2) + "\n")
    report = ["# Expanded Random Forest weighting experiment", "", "The saved four-fold baseline OOF predictions are the reference and were not retrained. Two weighting variants were fit with the same features, folds, Random Forest parameters, training-only duplicate removal, and threshold 0.5. All weights are derived separately within each training fold. No test-set tuning was performed.", "", "Validation is recording-held-out with patient overlap, not patient-independent validation. Event detection uses the existing descriptive any-overlap definition and is not a clinical alarm criterion.", "", "## Pooled comparison", "", "| Strategy | Accuracy | Balanced accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in comparison:
        report.append("| {strategy} | {accuracy:.4f} | {balanced_accuracy:.4f} | {precision:.4f} | {recall_sensitivity:.4f} | {specificity:.4f} | {f1_score:.4f} | {roc_auc:.4f} | {average_precision:.4f} |".format(**item))
    report += ["", "## Fold results", "", "See `fold_comparison.csv` for per-fold metrics and confusion matrices. `training_weight_summary.csv` documents the fold-specific weight contributions. `per_recording_comparison.csv` and `event_detection_comparison.csv` show where the results changed.", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return {"output_dir": str(output_dir), "pooled": comparison}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURE_MATRIX)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run_experiment(args.features, args.manifest, args.baseline_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
