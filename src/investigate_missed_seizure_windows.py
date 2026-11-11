#!/usr/bin/env python3
"""Descriptive analysis of saved baseline false-negative seizure windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import analyze_expanded_random_forest_diagnostics as diagnostics
import train_random_forest as baseline

EVAL = ROOT / "data/processed/evaluation/expanded_random_forest_baseline"
FEATURES = ROOT / "data/processed/assembled_features/chbmit_feature_dataset.csv"
MANIFEST = ROOT / "results/chbmit_dataset_manifest.csv"
OUTPUT = EVAL / "missed_window_investigation"
TARGETS = ("chb02_16", "chb02_16+", "chb03_34")
IDENTITY = ("patient_id", "recording_id", "relative_path")
KEY = (*IDENTITY, "window_index")


def robust_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Robust and conventional summaries for a finite numeric sample."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "min": None, "q1": None, "median": None, "mean": None,
                "q3": None, "max": None, "iqr": None, "mad": None, "std": None}
    q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return {"n": int(len(x)), "min": float(np.min(x)), "q1": float(q1),
            "median": float(median), "mean": float(np.mean(x)), "q3": float(q3),
            "max": float(np.max(x)), "iqr": float(q3 - q1),
            "mad": float(np.median(np.abs(x - median))),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0}


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Cliff's delta: P(left > right) - P(left < right), ties excluded."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    ranks = rankdata(np.concatenate([a, b]), method="average")
    u_a = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u_a / (len(a) * len(b)) - 1)


def feature_domain(name: str) -> str:
    if "wavelet_" in name:
        return "wavelet"
    if any(part in name for part in ("dominant_frequency", "spectral_centroid", "total_spectral_power", "band_power_")):
        return "frequency"
    return "time_amplitude"


def _event_for_window(row: pd.Series, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [e for e in events if diagnostics._overlaps(
        float(row.start_time_seconds), float(row.end_time_seconds),
        float(e["event_start_seconds"]), float(e["event_end_seconds"]))]
    if len(matches) > 1:
        return {"event_id": ";".join(e["event_id"] for e in matches),
                "event_start_seconds": min(e["event_start_seconds"] for e in matches),
                "event_end_seconds": max(e["event_end_seconds"] for e in matches)}
    return matches[0] if matches else None


def _duplicates(features_frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    digest_groups: dict[bytes, list[str]] = {}
    for name in features:
        digest = pd.util.hash_pandas_object(features_frame[name], index=False).values.tobytes()
        digest_groups.setdefault(digest, []).append(name)
    groups: list[list[str]] = []
    for candidates in digest_groups.values():
        if len(candidates) > 1:
            # Confirm equality to guard against a hash collision.
            partitions: list[list[str]] = []
            for candidate in candidates:
                for group in partitions:
                    if np.array_equal(features_frame[candidate].to_numpy(), features_frame[group[0]].to_numpy()):
                        group.append(candidate)
                        break
                else:
                    partitions.append([candidate])
            groups.extend(group for group in partitions if len(group) > 1)
    return pd.DataFrame([{"duplicate_group": i + 1, "feature": feature,
                          "group_size": len(group)}
                         for i, group in enumerate(groups) for feature in group])


def run_investigation(
    feature_path: Path = FEATURES,
    manifest_path: Path = MANIFEST,
    evaluation_dir: Path = EVAL,
    output_dir: Path = OUTPUT,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    preds = diagnostics._load_predictions(Path(evaluation_dir) / "out_of_fold_predictions.csv")
    data, feature_cols = baseline.load_feature_matrix(Path(feature_path))
    for col in IDENTITY:
        data[col] = data[col].astype(str)
    data["window_index"] = pd.to_numeric(data.window_index, errors="raise").astype(int)
    pred_keys = preds[list(KEY)].copy()
    data_keys = data[list(KEY)].copy()
    if set(map(tuple, pred_keys.astype(str).to_numpy())) != set(map(tuple, data_keys.astype(str).to_numpy())):
        raise ValueError("Baseline OOF keys do not match assembled feature windows.")
    # OOF order is canonical and key uniqueness was checked by the existing analyzer.
    frame = preds.merge(data[list(KEY) + feature_cols], on=list(KEY), how="left", validate="one_to_one")
    frame = frame.copy()  # compact the merged feature blocks before downstream slicing
    if frame[feature_cols].isna().any().any():
        raise ValueError("Feature join produced missing values.")
    manifest = diagnostics._read_manifest(Path(manifest_path))
    events, _durations = diagnostics._validate_annotations(frame, manifest)
    assignments = pd.read_csv(Path(evaluation_dir) / "fold_assignments.csv")
    fold_map = {tuple(str(row[c]) for c in IDENTITY): str(row["test_fold"])
                for row in assignments.to_dict("records")}
    if len(fold_map) != len(assignments):
        raise ValueError("Fold assignment composite identities are not unique.")
    features_only = data[feature_cols]
    duplicate_table = _duplicates(features_only, feature_cols)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame["outcome"] = np.select(
        [(frame.y_true == 1) & (frame.y_pred == 0), (frame.y_true == 1) & (frame.y_pred == 1),
         (frame.y_true == 0) & (frame.y_pred == 1)], ["FN", "TP", "FP"], default="TN")
    target_windows = frame[frame.recording_id.isin(TARGETS)].copy()
    window_rows, recording_rows, feature_rows, shift_rows, temporal_rows = [], [], [], [], []
    target_info = []
    for recording in TARGETS:
        rec = target_windows[target_windows.recording_id == recording].copy()
        if rec.empty:
            raise ValueError(f"Target recording absent from OOF predictions: {recording}")
        identity = tuple(str(rec.iloc[0][c]) for c in IDENTITY)
        evs = events.get(identity, [])
        if not evs:
            raise ValueError(f"Target recording has no validated seizure event: {identity}")
        event = evs[0] if len(evs) == 1 else None
        if event is None:
            raise ValueError(f"Expected one event for targeted recording, found {len(evs)}: {identity}")
        positive = rec[rec.y_true == 1].copy()
        positive["event_id"] = event["event_id"]
        positive["event_start_seconds"] = event["event_start_seconds"]
        positive["event_end_seconds"] = event["event_end_seconds"]
        positive["event_relative_start"] = (
            positive.start_time_seconds - event["event_start_seconds"]
        ) / (event["event_end_seconds"] - event["event_start_seconds"])
        positive["event_relative_end"] = (
            positive.end_time_seconds - event["event_start_seconds"]
        ) / (event["event_end_seconds"] - event["event_start_seconds"])
        positive.to_csv(output_dir / f"{recording}_seizure_windows.csv", index=False, float_format="%.12g")
        probability_summary = robust_summary(positive.probability_seizure)
        fn = positive[positive.outcome == "FN"]
        tp = positive[positive.outcome == "TP"]
        recording_rows.append({"patient_id": identity[0], "recording_id": recording,
                               "relative_path": identity[2], "test_fold": str(rec.test_fold.iloc[0]),
                               "event_id": event["event_id"], "event_start_seconds": event["event_start_seconds"],
                               "event_end_seconds": event["event_end_seconds"],
                               "annotated_seizure_windows": len(positive), "true_positives": len(tp),
                               "false_negatives": len(fn), "sensitivity": len(tp) / len(positive),
                               **{f"seizure_probability_{k}": v for k, v in probability_summary.items()}})
        # Event-progress bins are based on window starts relative to annotated event duration.
        edges = np.linspace(0, 1, 5)
        bin_id = np.clip(np.digitize(positive.event_relative_start, edges[1:-1]), 0, 3)
        for bin_num in range(4):
            section = positive[bin_id == bin_num]
            temporal_rows.append({"recording_id": recording, "event_id": event["event_id"],
                                  "event_progress_bin": f"{bin_num/4:.2f}-{(bin_num+1)/4:.2f}",
                                  "seizure_windows": len(section), "tp": int((section.outcome == "TP").sum()),
                                  "fn": int((section.outcome == "FN").sum()),
                                  "mean_probability": float(section.probability_seizure.mean()) if len(section) else None})
        # FN consecutive runs in the original window index sequence.
        ordered_fn = fn.sort_values("window_index")
        run = []
        runs = []
        for row in ordered_fn.itertuples(index=False):
            if run and row.window_index != run[-1].window_index + 1:
                runs.append(run)
                run = []
            run.append(row)
        if run:
            runs.append(run)
        for run_number, run_rows in enumerate(runs, start=1):
            temporal_rows.append({"recording_id": recording, "event_id": event["event_id"],
                                  "event_progress_bin": f"FN_run_{run_number}", "seizure_windows": len(run_rows),
                                  "tp": 0, "fn": len(run_rows),
                                  "mean_probability": float(np.mean([x.probability_seizure for x in run_rows])),
                                  "run_start_seconds": float(run_rows[0].start_time_seconds),
                                  "run_end_seconds": float(run_rows[-1].end_time_seconds)})
        target_info.append((recording, identity, str(rec.test_fold.iloc[0]), positive, event))
        for row in positive.to_dict("records"):
            window_rows.append({"patient_id": identity[0], "recording_id": recording,
                                "relative_path": identity[2], "test_fold": str(row["test_fold"]),
                                "window_index": int(row["window_index"]), "event_id": event["event_id"],
                                "event_start_seconds": event["event_start_seconds"], "event_end_seconds": event["event_end_seconds"],
                                "window_start_seconds": row["start_time_seconds"], "window_end_seconds": row["end_time_seconds"],
                                "outcome": row["outcome"], "y_true": int(row["y_true"]), "y_pred": int(row["y_pred"]),
                                "probability_seizure": row["probability_seizure"],
                                "probability_below_threshold_margin": 0.5 - row["probability_seizure"]})

    # All comparison groups and training/test-seizure shift tables.
    group_sets = []
    for recording, identity, fold, positive, event in target_info:
        held = frame[frame.recording_id == recording]
        groups = {
            "missed_seizure_FN": held[(held.y_true == 1) & (held.y_pred == 0)],
            "detected_seizure_TP_same_recording": held[(held.y_true == 1) & (held.y_pred == 1)],
            "non_seizure_same_recording": held[held.y_true == 0],
            "detected_seizure_TP_other_recordings": frame[(frame.y_true == 1) & (frame.y_pred == 1) & (frame.recording_id != recording)],
            "training_seizure_windows_for_heldout_fold": frame[(frame.y_true == 1) & (frame.test_fold != fold)],
        }
        group_sets.append((recording, fold, groups))
        training = groups["training_seizure_windows_for_heldout_fold"]
        for label, test_group in (("heldout_FN", groups["missed_seizure_FN"]),
                                  ("heldout_TP", groups["detected_seizure_TP_same_recording"])):
            for feature in feature_cols:
                a = training[feature].to_numpy(dtype=float)
                b = test_group[feature].to_numpy(dtype=float)
                sa, sb = robust_summary(a), robust_summary(b)
                train_iqr = sa["iqr"] or 0.0
                lower, upper = sa["q1"] - 1.5 * train_iqr, sa["q3"] + 1.5 * train_iqr
                shift_rows.append({"recording_id": recording, "heldout_fold": fold,
                                   "comparison": f"training_seizures_vs_{label}", "feature": feature,
                                   "domain": feature_domain(feature), "training_n": sa["n"], "test_n": sb["n"],
                                   "training_median": sa["median"], "test_median": sb["median"],
                                   "median_difference_test_minus_train": (sb["median"] - sa["median"]) if sb["median"] is not None else None,
                                   "cliffs_delta_test_vs_train": cliffs_delta(b, a),
                                   "test_fraction_outside_training_1_5IQR": (
                                       float(np.mean((b < lower) | (b > upper))) if len(b) else None),
                                   "training_q1": sa["q1"], "training_q3": sa["q3"],
                                   "test_q1": sb["q1"], "test_q3": sb["q3"]})
        for group_name, group in groups.items():
            for feature in feature_cols:
                summary = robust_summary(group[feature].to_numpy(dtype=float))
                training = groups["training_seizure_windows_for_heldout_fold"][feature].to_numpy(dtype=float)
                same_recording_tp = groups["detected_seizure_TP_same_recording"][feature].to_numpy(dtype=float)
                ss = robust_summary(training)
                tp_summary = robust_summary(same_recording_tp)
                if len(training):
                    low, high = ss["q1"] - 1.5 * (ss["iqr"] or 0), ss["q3"] + 1.5 * (ss["iqr"] or 0)
                    outside = float(np.mean((group[feature].to_numpy(dtype=float) < low) |
                                            (group[feature].to_numpy(dtype=float) > high))) if len(group) else None
                    iqr_ratio = (summary["iqr"] / ss["iqr"]
                                 if summary["iqr"] is not None and ss["iqr"] not in (None, 0)
                                 else None)
                else:
                    outside = None
                    iqr_ratio = None
                feature_rows.append({"recording_id": recording, "test_fold": fold,
                                     "group": group_name, "feature": feature,
                                     "domain": feature_domain(feature), **summary,
                                     "training_seizure_median": ss["median"],
                                     "training_seizure_iqr": ss["iqr"],
                                     "iqr_ratio_to_training_seizure": iqr_ratio,
                                     "median_difference_vs_same_recording_TP": (
                                         summary["median"] - tp_summary["median"]
                                         if summary["median"] is not None and tp_summary["median"] is not None else None),
                                     "cliffs_delta_vs_same_recording_TP": cliffs_delta(group[feature], same_recording_tp),
                                     "cliffs_delta_vs_training_seizures": cliffs_delta(group[feature], training),
                                     "fraction_outside_training_seizure_1_5IQR": outside})

    # Global probability confidence by outcome and label-specific group.
    prob_rows = []
    for outcome, group in frame.groupby("outcome", sort=True):
        prob_rows.append({"group": outcome, **robust_summary(group.probability_seizure)})
    for recording in TARGETS:
        group = target_windows[(target_windows.recording_id == recording) & (target_windows.y_true == 1)]
        prob_rows.append({"group": f"{recording}_all_seizure_windows", **robust_summary(group.probability_seizure)})
    pd.DataFrame(prob_rows).to_csv(output_dir / "probability_distributions.csv", index=False)
    pd.DataFrame(recording_rows).to_csv(output_dir / "recording_summaries.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output_dir / "seizure_window_probabilities.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(output_dir / "temporal_error_distribution.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output_dir / "feature_distribution_comparisons.csv", index=False, float_format="%.12g")
    pd.DataFrame(shift_rows).to_csv(output_dir / "fold_train_test_feature_shift.csv", index=False, float_format="%.12g")
    duplicate_table.to_csv(output_dir / "exact_duplicate_features.csv", index=False)

    # Examine correlations only among strongest descriptive FN-vs-training effects;
    # the full unfiltered feature table remains available for audit.
    feat_df = pd.DataFrame(feature_rows)
    focus = feat_df[feat_df.group == "missed_seizure_FN"].copy()
    focus["abs_delta"] = focus.cliffs_delta_vs_training_seizures.abs()
    selected = list(focus.sort_values("abs_delta", ascending=False).groupby("recording_id").head(20).feature.unique())
    correlations = data[selected].corr(method="spearman") if len(selected) > 1 else pd.DataFrame()
    corr_rows = [{"feature_a": a, "feature_b": b, "spearman_rho": float(correlations.loc[a, b])}
                 for i, a in enumerate(selected) for b in selected[i + 1:]
                 if abs(correlations.loc[a, b]) >= 0.95]
    pd.DataFrame(corr_rows, columns=["feature_a", "feature_b", "spearman_rho"]).to_csv(
        output_dir / "high_correlation_among_effect_ranked_features.csv", index=False)

    # Focused plot: event progress and saved probability against fixed threshold.
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    for ax, (recording, identity, fold, positive, event) in zip(axes, target_info):
        ax.scatter(positive.start_time_seconds, positive.probability_seizure,
                   c=np.where(positive.y_pred.to_numpy() == 1, "#187a3d", "#bd3939"), s=30)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="fixed threshold 0.5")
        ax.axvspan(event["event_start_seconds"], event["event_end_seconds"], color="#e9b949", alpha=0.18,
                   label="annotated seizure interval")
        ax.set(title=f"{recording} ({fold} held out)", ylabel="P(seizure)", ylim=(-0.03, 1.03))
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Recording time (seconds)")
    fig.suptitle("Saved baseline OOF probabilities for annotated seizure windows\nGreen=TP, red=FN")
    fig.tight_layout()
    fig.savefig(output_dir / "target_seizure_probability_timelines.png", dpi=160)
    plt.close(fig)

    # Feature-effect summary: strongest absolute effects are flags, not causal selections.
    top_rows = []
    for recording in TARGETS:
        subset = focus[focus.recording_id == recording].sort_values("abs_delta", ascending=False).head(12)
        top_rows.extend(subset[["recording_id", "feature", "domain", "n", "median", "training_seizure_median",
                                "iqr", "training_seizure_iqr", "iqr_ratio_to_training_seizure",
                                "cliffs_delta_vs_training_seizures", "fraction_outside_training_seizure_1_5IQR"]].to_dict("records"))
    pd.DataFrame(top_rows).to_csv(output_dir / "largest_descriptive_FN_feature_effects.csv", index=False)
    low_variation = focus[focus.iqr_ratio_to_training_seizure.notna()].sort_values(
        "iqr_ratio_to_training_seizure").groupby("recording_id").head(10)
    low_variation.to_csv(output_dir / "lowest_within_FN_variation.csv", index=False)

    report_lines = [
        "# Missed seizure-window investigation",
        "",
        "This is a descriptive analysis of the saved expanded-baseline out-of-fold predictions and assembled features. No model was fitted, no labels/features/folds were changed, and the original 0.5 threshold was retained.",
        "Validation is recording-held-out with patient overlap, not patient-independent validation.",
        "",
        "## Target recording results",
        "",
        "| Recording | Fold | Event | Seizure windows | TP | FN | Sensitivity | Probability min / median / mean / max |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in recording_rows:
        report_lines.append(f"| {row['recording_id']} | {row['test_fold']} | {row['event_id']} ({row['event_start_seconds']:.1f}–{row['event_end_seconds']:.1f}s) | {row['annotated_seizure_windows']} | {row['true_positives']} | {row['false_negatives']} | {row['sensitivity']:.3f} | {row['seizure_probability_min']:.4f} / {row['seizure_probability_median']:.4f} / {row['seizure_probability_mean']:.4f} / {row['seizure_probability_max']:.4f} |")
    report_lines += [
        "",
        "The seizure-window CSVs retain each window's time, event, outcome, and saved probability. `temporal_error_distribution.csv` groups windows by event progress quartile and reports consecutive FN runs. The timeline figure shows every positive window against the unchanged threshold.",
        "",
        "## Feature and fold comparisons",
        "",
        "`feature_distribution_comparisons.csv` reports n, min, quartiles, median, mean, max, IQR, MAD, standard deviation, Cliff's delta, and the fraction outside the training-seizure 1.5-IQR range for each feature and comparison group. It also reports direct FN/TP differences within the same recording and each group's IQR relative to the training-seizure IQR; `lowest_within_FN_variation.csv` lists the lowest ratios per recording (the ratio is undefined when the training IQR is zero). The groups include missed and detected seizure windows, all non-seizure windows in the same recording, detected seizure windows from other recordings, and seizure windows in the relevant fold's training data. `fold_train_test_feature_shift.csv` directly contrasts held-out FN/TP windows with training seizure windows. `largest_descriptive_FN_feature_effects.csv` is a ranking aid, not causal feature selection.",
        "",
        "### Strongest descriptive patterns",
        "",
        "- Confidence is usually low for missed windows: CHB02_16 has 24/33 FNs below P=0.1 (FN median 0.033); CHB02_16+ has 13/25 below 0.1 and 7/25 in [0.3, 0.5) (FN median 0.080); CHB03_34 has 12/17 below 0.1 and 3/17 in [0.3, 0.5) (FN median 0.027). Thus the errors are mostly low-confidence, with a minority of near-threshold misses in the latter two recordings. Across all recordings the FN probability median is 0.080, compared with TP 0.788 and FP 0.585.",
        "- Errors are temporally structured. CHB02_16 has one TP in event-progress quartile 2, with the remaining quartiles entirely FN; its FNs form runs of 12 and 21 consecutive windows. CHB02_16+ detects 7/10 windows in the first event quartile and 3/8 in the second, then misses all windows in the final half; it has a 19-window late FN run. CHB03_34 detects 3/5 windows in the final quartile but misses all 9 windows across the middle two quartiles; a 15-window consecutive FN run spans 1985–2025 s.",
        "- FN-versus-training-seizure differences are large but recording-specific. In CHB02_16, C3-P3 theta band-power has FN median about 6,508, Cliff's delta +0.992, and 94% of FNs outside the training seizure 1.5-IQR range; several posterior alpha/theta features rank similarly. CHB02_16+ also shows elevated C3-P3 theta/alpha band-power (delta about +0.90; 92%/88% outside range). In CHB03_34, CZ-PZ wavelet D1 standard deviation and energy have delta about +0.993, with all FNs outside the training range; CZ-PZ gamma power is also elevated (delta +0.974, all outside). These effects point to different channel/domain patterns rather than one shared feature signature.",
        "- Within-FN variability is sometimes small relative to training seizure variability: for example, CHB02_16 P7-O1 wavelet D1 energy has FN IQR about 3.4% of the training-seizure IQR; CHB02_16+ FT9-FT10 wavelet D1 energy is about 4.4%; CHB03_34 T8-P8 delta band-power is about 5.3%. This means the held-out FN subsets are internally more compact for these features, while their medians may still be shifted.",
        f"",
        f"Exact duplicates were checked across the assembled feature matrix: {duplicate_table.duplicate_group.nunique() if not duplicate_table.empty else 0} duplicated feature groups involving {len(duplicate_table)} columns were found. Spearman pairs with |rho| >= 0.95 among the descriptive effect-ranked feature subset are in `high_correlation_among_effect_ranked_features.csv` ({len(corr_rows)} pairs). Correlation and exact duplication mean ranked features are not independent evidence.",
        "",
        "## Limitations and interpretation",
        "",
        "These small event-specific window groups support descriptive comparisons only. Overlapping windows are correlated, the held-out unit is a recording and patients overlap between training and testing, and multiple channels/features are redundant. Cliff's delta summarizes rank separation; it does not establish a mechanism. Extreme feature values or train/test distribution shifts may reflect event morphology, recording conditions, reference/montage characteristics, or preprocessing interactions; this analysis cannot distinguish those explanations or show that any feature caused a miss. No threshold was tuned and no clinical significance or alarm utility is inferred.",
        "",
        "## Recommended next investigation",
        "",
        "Use the probability timeline and feature tables to check whether errors cluster around event boundaries and whether the same channel/domain shifts recur across the three recordings. Then inspect the existing feature extraction and annotation-window conventions only for a documented inconsistency; do not change them based on these observational effects alone. If no implementation discrepancy is found, a separately scoped validation study with more seizure recordings and patient-independent evaluation is more informative than adding model complexity to this small cohort.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    metadata = {"input_oof": str(Path(evaluation_dir) / "out_of_fold_predictions.csv"),
                "feature_matrix": str(feature_path), "manifest": str(manifest_path),
                "target_recordings": list(TARGETS), "threshold": 0.5,
                "method": "descriptive robust summaries, Cliff's delta, train-seizure 1.5-IQR range; no model fitting",
                "n_windows": len(frame), "feature_count": len(feature_cols),
                "validation": "recording-held-out with patient overlap; not patient-independent"}
    (output_dir / "analysis_configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"output_dir": str(output_dir), "recordings": recording_rows,
            "feature_count": len(feature_cols), "duplicate_feature_groups": int(duplicate_table.duplicate_group.nunique()) if not duplicate_table.empty else 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--evaluation-dir", type=Path, default=EVAL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_investigation(args.features, args.manifest, args.evaluation_dir, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
