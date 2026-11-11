# EEG Seizure Detection

An independent reconstruction and expanded evaluation of an EEG seizure-detection pipeline using signal processing and Random Forest classification. The original implementation was unavailable; reconstruction choices are documented in [the methodology notes](docs/original_methodology.md). This project does not claim an exact reproduction of the original paper.

## Project scope

The evaluated subset contains 19 CHB-MIT recordings from CHB01, CHB02, and CHB03. Source EEG data and derived datasets are not included. Obtain the dataset under its applicable terms and place the selected patient folders, summary files, EDF recordings, and available seizure sidecars under `data/raw/chbmit/`.

The pipeline audits annotations, converts EEG recordings to tabular signals in bounded chunks, creates a window manifest, extracts per-channel features, assembles a corpus matrix, and evaluates a Random Forest. The evaluated pipeline uses 5-second windows with 50% overlap and a reconstruction-specific `any_overlap` seizure-label policy. It extracts time-domain statistics, frequency-domain summaries and band power, and level-3 wavelet statistics. Feature definitions and other choices not specified by the paper are recorded in the methodology notes.

## Environment

Use Python 3.10 or newer. Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the expanded evaluation

After placing source data under `data/raw/chbmit/`, build the manifest, process available recordings, assemble the feature matrix, and run the four-fold evaluation:

```bash
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

The manifest preserves summary-listed but unavailable recordings and records annotation validation status. Processing and assembly use eligible local recordings and write generated data under ignored `data/` and `results/` paths. The Random Forest runner reads `data/processed/assembled_features/chbmit_feature_dataset.csv` and writes evaluation artifacts under `data/processed/evaluation/expanded_random_forest_baseline/`.

## Evaluation

The baseline uses four-fold stratified cross-validation at the recording level, with whole recordings held out and stratification on recording-level seizure presence. Fold assignment is deterministic (seed 42). The selected cohort has 8 seizure-containing and 11 non-seizure recordings. Saved out-of-fold predictions cover 26,284 windows: 206 seizure windows and 26,078 non-seizure windows (0.78% seizure windows).

The Random Forest configuration uses 300 trees, Gini criterion, unrestricted depth, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, bootstrap sampling, `class_weight="balanced"`, seed 42, and one worker. Exact duplicate predictors are identified using only each training fold. No hyperparameters or classification thresholds were tuned against held-out predictions; the threshold is `P(seizure) >= 0.5`.

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

Confusion matrix (rows are actual negative/positive; columns are predicted negative/positive):

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 26,042 | 36 |
| Actual positive | 110 | 96 |

The high accuracy and specificity coexist with low seizure-window sensitivity because seizure windows are rare. These are window-weighted metrics; longer recordings contribute more windows. Folds hold out recordings, but patients can occur in both train and test partitions through different recordings. This is recording-held-out validation with patient overlap, not patient-independent validation. The subset contains only three patients and does not establish clinical performance or generalization to the full database.

An exploratory event-/recording-weighting comparison did not improve seizure-window sensitivity. Post-hoc analysis found recording-specific clusters of missed windows. These diagnostics are descriptive and do not establish causation or clinical utility.

## Other commands

The annotation audit supports a selected patient:

```bash
python src/annotation_audit.py --patient chb01
```

The CHB01 recording-wise baseline and feature-family ablation are separate exploratory analyses:

```bash
python src/train_random_forest.py
python src/feature_family_ablation.py
```

Run the tests with:

```bash
python -m pytest -q
```

Tests create temporary fixtures; source EEGs and generated results remain local. Do not interpret the legacy CHB01 folds or the expanded recording-wise folds as patient-independent evaluation.
