# EEG Seizure Detection

A Python pipeline for classifying seizure and non-seizure windows in scalp EEG recordings. It processes CHB-MIT recordings, extracts time-, frequency-, and wavelet-domain features, and evaluates a Random Forest model. The workflow provides the code implementation of my Research publication of signal-feature-classifier approach described in [“Analysis and Detection of Seizures Using EEG Signal”](https://doi.org/10.1109/AISP61711.2024.10870633).

## Dataset

The evaluation uses 19 recordings from three CHB-MIT patients: CHB01, CHB02, and CHB03. Eight recordings contain seizures; eleven contain no annotated seizure. The source EDF files and generated datasets are not included in this repository.

The four-fold evaluation contains 26,284 windows: 206 seizure windows (0.78%) and 26,078 non-seizure windows.

## Pipeline

1. Inventory EDF recordings and patient summaries; compare annotations with available seizure sidecars.
2. Convert EEG signals to tabular files in bounded chunks.
3. Divide recordings into five-second windows with 50% overlap. Label a window as seizure when it overlaps a validated seizure interval.
4. Extract per-channel features:
   - **Time:** mean, population standard deviation, maximum, minimum.
   - **Frequency:** dominant non-DC frequency, spectral centroid, total power, and delta/theta/alpha/beta/gamma band power.
   - **Wavelet:** approximation and detail coefficient summaries from a level-three `db4` transform.
5. Train and evaluate a Random Forest.

## Model and evaluation

The baseline Random Forest has 300 trees, Gini criterion, unrestricted depth, `max_features="sqrt"`, bootstrap sampling, `class_weight="balanced"`, and random seed 42. Exact duplicate features are identified using training-fold data only. The classification threshold is `P(seizure) >= 0.5`.

Evaluation uses four-fold stratified cross-validation with whole recordings held out. Each recording appears in one test fold. Patients occur in both training and test partitions through different recordings, so this is recording-held-out evaluation with patient overlap, not patient-independent evaluation.

## Results

Metrics are pooled over all out-of-fold window predictions.

| Metric | Result |
|---|---:|
| Accuracy | 99.44% |
| Balanced accuracy | 73.23% |
| Sensitivity | 46.60% |
| Specificity | 99.86% |
| Precision | 72.73% |
| F1 score | 56.80% |
| ROC-AUC | 97.42% |
| Average precision | 57.76% |

Rows in the confusion matrix are actual labels; columns are predicted labels.

| | Predicted non-seizure | Predicted seizure |
|---|---:|---:|
| **Actual non-seizure** | 26,042 | 36 |
| **Actual seizure** | 110 | 96 |

### ROC curve

The red marker shows the operating point at the fixed 0.5 threshold.

![Pooled out-of-fold ROC curve, AUC 0.9742](figures/pooled_oof_roc_curve.png)

### Precision-recall curve

The dashed line indicates seizure-window prevalence.

![Pooled out-of-fold precision-recall curve, average precision 0.5776](figures/pooled_oof_precision_recall_curve.png)

### Confusion matrix

![Pooled out-of-fold confusion matrix](figures/pooled_oof_confusion_matrix.png)

Seizure windows are rare in this dataset, so accuracy and specificity should be considered together with sensitivity: the model missed 110 of 206 seizure windows at the fixed threshold.

## Run the pipeline

Use Python 3.10 or newer. Obtain CHB-MIT data separately and place the selected patient folders, EDF recordings, summaries, and available sidecars under `data/raw/chbmit/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/dataset_manifest.py \
  --dataset-root data/raw/chbmit \
  --output results/chbmit_dataset_manifest.csv

python src/process_chbmit.py

python src/assemble_feature_dataset.py \
  --corpus-manifest results/chbmit_dataset_manifest.csv \
  --state data/processed/corpus_pipeline/.process_chbmit_state.json \
  --output-dir data/processed/assembled_features \
  --chunk-rows 4096

python src/train_expanded_random_forest.py
```

The baseline reads `data/processed/assembled_features/chbmit_feature_dataset.csv` and writes results under `data/processed/evaluation/expanded_random_forest_baseline/`. Data and generated evaluation files are ignored by Git.

## Limitations

- The evaluation covers 19 recordings from three patients, not the full CHB-MIT database.
- Patient overlap across folds means the results do not measure performance on unseen patients.
- Overlapping windows are correlated, and the reported metrics are window-level.
- The fixed-threshold sensitivity is 46.60%; this model has not been clinically validated.
