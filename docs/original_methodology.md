# Original EEG seizure-detection methodology

## Purpose and source

This document records the methods described in the paper **“Analysis and
Detection of Seizures Using EEG Signal.”** It is a reconstruction of the
reported methodology, not recovered source code. The paper does not provide
enough implementation detail to reproduce every result exactly.

The paper is the primary source for the sections below. Statements are
paraphrased; they distinguish what the paper says from assumptions that would
otherwise be needed to implement it.

## Methods explicitly reported

### Dataset and EDF conversion

In its dataset description, the paper identifies the CHB-MIT Scalp EEG
Database and describes 22 patients across 23 cases. It reports 664 EDF files,
198 annotated seizures, and 129 files containing seizures. It describes the
recordings as sampled at 256 samples per second with 16-bit resolution and
recorded using the international 10-20 method. The paper also notes that
recordings can include ECG or VNS signals.

The conversion section describes using `pyedflib` to open an EDF and obtain
its duration, sampling frequency, signal labels, and samples. It describes
using `pandas` to build a table with a time column and a signal column for
each channel, then writing that table to CSV.

These are descriptions of the paper's dataset and conversion approach. They
do not establish that every database EDF was included in the reported
experiments or that the current partial local download reproduces the paper's
experimental cohort.

#### EDF-to-CSV implementation decisions in this reconstruction

The paper does not specify the exact CSV time origin, signal units, chunk
strategy, or channel-type selection rule. This implementation therefore makes
the following explicit choices:

- The time column starts at 0 seconds and is computed as sample index divided
  by sampling frequency. It has one row per sample and does not add a sample
  at the nominal recording-duration endpoint.
- Conversion reads fixed-duration chunks and writes them to one CSV so that
  all signals from a full recording need not be held in memory at once.
- The output name is the exact source EDF filename followed by `.csv`
  (for example, `chb01_02.edf.csv`), under `data/processed/csv/`.
- Signal values remain in the physical units provided by pyEDFlib; they are
  not rescaled. EDF channel units are logged.
- EDF does not have a reliably populated universal signal-type field. The
  converter selects explicit `EEG` labels and labels matching recognized
  scalp-electrode names/pairs, excludes clearly auxiliary labels, and logs
  excluded or unclassified labels. This is a label-based reconstruction
  choice, not a channel-selection rule reported by the paper.
- Duplicate EEG labels are preserved exactly as stored in the EDF header.
  The current CHB01 EDFs contain two `T8-P8` signals; both are retained as
  separate columns with the identical header label.
- A single time column requires EEG channels to share sample frequency and
  sample count. Heterogeneous channels cause an explicit error rather than
  silent resampling or alignment.
- A temporary CSV is checked against EDF labels, sampling frequency, sample
  count, time interval, and non-finite value counts before it replaces the
  deterministic output.

### Feature extraction

The paper describes three feature groups:

1. **Time domain:** mean, standard deviation, maximum amplitude, and minimum
   amplitude computed from EEG samples.
2. **Frequency domain:** an FFT/DFT representation, with high-magnitude
   frequency components treated as significant features.
3. **Wavelet domain:** a level-3 discrete wavelet decomposition, with
   wavelet coefficients used as features.

#### Unspecified feature details and reconstruction implementation

The paper does not fully define the reported features. The window size and
overlap; channel-wise versus pooled computation; standard-deviation
convention; FFT detrending, taper, normalization, bins, power calculation,
frequency bands, and spectral summaries; and wavelet family, boundary mode,
coefficient ordering, and coefficient summaries are not established. The
following choices belong only to this reconstruction and must not be cited as
the exact paper implementation:

- Features are computed independently for every 5-second channel window from
  the existing segmentation manifest. The converted CSV's ordered signal
  columns are matched exactly to the manifest; channel indices accompany
  labels so duplicate labels remain separate. This does not imply the paper
  used those windows or channel-wise calculation.
- Time-domain features are arithmetic mean, population standard deviation
  (`ddof=0`), signed maximum, and signed minimum.
- Frequency features use a mean-subtracted, untapered window and a real FFT
  (`rfft`). The dominant component is the largest-magnitude non-DC bin;
  spectral centroid is magnitude-weighted over positive-frequency bins. A
  one-sided periodogram density is calculated as `|X|^2 / (fs * N)`, with
  interior positive-frequency bins doubled, excluding DC and (for even
  lengths) Nyquist. Total power is the sum of this density times `fs/N`
  across all one-sided bins, including DC and Nyquist. This normalization
  makes the sum equal to the mean-square value of the mean-subtracted
  window, subject to floating-point precision.
- Default band power uses bin-center frequencies and half-open boundaries:
  delta `[0.5, 4)` Hz, theta `[4, 8)` Hz, alpha `[8, 13)` Hz, beta
  `[13, 30)` Hz, and gamma `[30, 80)` Hz. Power is the sum of the
  one-sided density times bin width for bins whose centers are within the
  band. Bands are configurable; no claim is made that these exact boundaries
  appeared in the paper.
- Wavelet features use PyWavelets' level-3 DWT with configurable wavelet
  family (default Daubechies-4, `db4`) and symmetric boundary extension.
  Coefficients are ordered `cA3`, `cD3`, `cD2`, `cD1`; mean, population
  standard deviation, and sum-of-squared-coefficients energy are calculated
  for each set.
- Missing/non-finite signal values fail extraction explicitly; no
  imputation or channel removal is used. The output is a deterministic,
  wide feature matrix with one row per manifest window, a feature-configuration
  record, and one indexed feature group per channel.

These are reproducible implementation decisions, not reported paper details.
Feature selection, classification, splitting, and metric calculation are not
part of this feature-extraction stage.

### Feature selection

The paper says it uses scikit-learn `SelectKBest` with the `f_regression`
score function. It says features are ranked for relevance to a “dummy target”
and that statistically significant frequency-domain and wavelet features
are identified.

### Classification and reported evaluation

The paper describes a Random Forest classifier and gives general explanations
of decision trees, bootstrap aggregation, random feature subsets, and
majority voting for classification. It does not provide a complete
experiment-specific Random Forest configuration in the supplied paper.

The paper discusses accuracy, precision, recall, F1-score, support, a
confusion matrix, ROC-AUC, a precision-recall curve, and training/testing
accuracy and loss curves. It reports:

| Feature group | Reported accuracy |
|---|---:|
| Time domain | 85.83% |
| Frequency domain | 86.67% |
| Wavelet transform | 72.50% |

It also reports a confusion matrix with 84 true negatives, 21 true positives,
3 false positives, and 12 false negatives; ROC-AUC 0.79; and average
precision 0.71. The confusion-matrix counts sum to 120 and imply 87.5%
accuracy, which does not directly match the feature-group accuracy table.
The paper does not explain whether these numbers describe the same experiment,
feature set, or evaluation subset.

## Important ambiguity: `f_regression` and a “dummy target”

`SelectKBest(f_regression, k=...)` scores each input feature against the
provided numeric target. `f_regression` is a univariate linear-regression
F-test scorer; it does not create a meaningful target by itself. The paper
does not specify:

- How the “dummy target” is constructed, whether it is constant, random,
  encoded from seizure labels, or something else.
- Whether a different target is used for the frequency and wavelet feature
  sets.
- The value of `k`, a selection threshold, or how the reported “statistical
  significance” is used.
- Whether feature selection is fit using training data only or using the
  full dataset.

These omissions materially affect the method. A constant target has no
feature relationship and, with current scikit-learn's default finite-score
handling, produces zero scores rather than useful discriminative ranking.
An arbitrary generated target would rank features by accidental association.
If the intended target is the actual binary seizure label encoded numerically,
that is not a dummy target; it is supervised feature selection, and the
selector must be fit only on each training fold to avoid leakage. A
classification-oriented scorer such as `f_classif` would be a methodological
alternative, not something the paper explicitly reports.

These API details are checked against the current scikit-learn documentation:
[`f_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
and
[`SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).

**No dummy target will be invented, and no selector will be implemented as
part of this documentation task.** Before implementing the baseline, the
target meaning and selector fitting scope need an explicit decision. Any
statistically sound alternative must be recorded separately from the
paper-faithful baseline.

## Model-readiness audit and implemented exploratory baseline

The paper reports a Random Forest classifier, but does not specify enough
settings or evaluation details to reconstruct its model or split exactly.
The first local baseline below is an explicit reconstruction choice, not a
claim that the paper used this protocol. Its results describe this partial
CHB01 dataset only.

### Current data audit

The existing manifest and feature matrix contain:

- 10,073 windows from 7 recordings belonging to 1 patient (`chb01`).
- 10,014 `non_seizure` and 59 `seizure` windows (about 0.59% positive).
- 563 feature-matrix columns: 11 metadata columns and 552 per-channel
  feature columns (24 for each of 23 channel positions).
- No missing or non-finite per-channel feature values, and no constant
  per-channel feature columns in this generated matrix.
- 552 unique feature-column names; no naming collisions. Channel index plus
  escaped channel label differentiates repeated labels.

Per-recording labels are:

| Recording ID | Windows | Seizure windows | Non-seizure windows |
|---|---:|---:|---:|
| `chb01_02` | 1,439 | 0 | 1,439 |
| `chb01_06` | 1,439 | 0 | 1,439 |
| `chb01_09` | 1,439 | 0 | 1,439 |
| `chb01_12` | 1,439 | 0 | 1,439 |
| `chb01_13` | 1,439 | 0 | 1,439 |
| `chb01_16` | 1,439 | 22 | 1,417 |
| `chb01_18` | 1,439 | 37 | 1,402 |

All adjacent windows are separated by a 640-sample stride for 1,280-sample
windows, so consecutive windows share 640 samples (50%). Randomly splitting
windows can place features computed partly from the same raw EEG samples in
both training and test sets. The overlapping windows also remain strongly
temporally and label-related even where they do not straddle the exact same
samples.

The feature matrix keeps `patient_id`, `recording_id`, `recording_filename`,
window index/times, `seizure_label`, `label_policy`, sampling frequency,
window size, and feature configuration alongside predictors. Only columns
with the `channel_` feature naming pattern are signal-derived candidate
predictors. `seizure_label` is the target, not a predictor; IDs, times, policy,
window configuration, and feature configuration must also be excluded from a
model's predictor matrix. Window index and recording-relative times can encode
event position and should not be used as shortcuts for the label. In
particular, do not pass all CSV columns to a model or selector and rely on
automatic type handling.

There are 25 exact duplicate feature-column pairs in the generated matrix:
24 pairs are the feature values for channel positions 14 and 22, both labeled
`T8-P8`. Inspection of all seven converted CSVs shows those two signal columns
are sample-for-sample identical. The remaining pair is dominant frequency for
`T7-P7` and `P7-T7`, which have identical dominant-bin values; reversing a
bipolar derivation changes polarity but not FFT magnitude. These are redundant
predictors, not label-derived columns or column-name collisions. They should
be documented for feature interpretation and handled explicitly by the
baseline's training-only duplicate check, not silently renamed or removed.

### Implemented recording-wise exploratory folds

This subsection records the two-fold protocol implemented by
`src/train_random_forest.py`. It remains an exploratory reconstruction choice.

Use the two available seizure-containing recordings as the two held-out test
recordings. In each fold, all windows from one held-out recording are in test;
the other seizure recording and all five available non-seizure recordings are
in training. This keeps every recording wholly within one partition.

| Fold | Training recordings | Training windows (seizure / non-seizure) | Test recording | Test windows (seizure / non-seizure) |
|---|---|---:|---|---:|
| 1 | `chb01_18` and `chb01_02`, `chb01_06`, `chb01_09`, `chb01_12`, `chb01_13` | 8,634 (37 / 8,597) | `chb01_16` | 1,439 (22 / 1,417) |
| 2 | `chb01_16` and `chb01_02`, `chb01_06`, `chb01_09`, `chb01_12`, `chb01_13` | 8,634 (22 / 8,612) | `chb01_18` | 1,439 (37 / 1,402) |

Both folds have both classes in their training and test partitions;
no fold is discarded or altered to obtain this property. `recording_id` (or
the corresponding `recording_filename`) is the grouping key, and the current
manifest has one patient ID per recording with all window indices retained
together. The two positive recordings are distinct recording groups, so this
design can assess transfer to a different recording from the same patient.
It cannot establish patient-independent performance: the dataset has only one
patient, only two seizure-containing recordings, and only one positive
recording available for training in each fold. A single event per seizure
recording, the small positive-window counts, severe class imbalance, and
overlapping windows also make the estimates highly uncertain. This is an
exploratory recording-wise protocol, not evidence of generalization or a
reproduction of the paper's unspecified split.

### Feature-selection decision for this first baseline

The paper's “dummy target” cannot be reconstructed reliably from the
description: it does not say whether the target was constant, random,
label-derived, or something else. Those interpretations have materially
different behavior, and inventing one would not be a faithful reproduction.
Therefore this first Random Forest baseline does not run
`SelectKBest(f_regression)` and does not invent a target.

If feature selection is considered in a later, separate experiment,
`f_classif` using the actual binary expert-derived window labels is a
classification-appropriate univariate ANOVA score and is more meaningful than
an arbitrary dummy target. That is a methodological deviation from the paper's
named `f_regression` procedure, not its reconstruction. Any selector must be
fit only on each fold's training data, then applied to that fold's held-out
recording. Fitting or ranking features on all windows before splitting would
let test-distribution information affect feature selection and bias evaluation.
Because overlapping windows are dependent and the positive training data in
each proposed fold comes from only one recording, nominal univariate
significance scores should also be interpreted cautiously.

No value of `k`, selection scope (per feature family versus jointly), or
nested tuning procedure is chosen here. No feature selection is applied in the
implemented baseline.

### Implemented Random Forest and metrics

The standalone baseline runner uses the generated feature matrix and these
fixed configuration choices:

- `RandomForestClassifier` with 300 estimators, Gini criterion,
  `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`,
  `max_features="sqrt"`, bootstrap sampling, and `class_weight="balanced"`.
- Fixed `random_state=42` and `n_jobs=1` for reproducibility.
- A fixed positive-class decision rule of `P(seizure) >= 0.5`. Neither
  hyperparameters nor threshold are tuned against test data.
- Predictors are limited to signal feature columns prefixed `channel_`.
  Patient/recording IDs, window index and times, seizure target, labeling
  policy, sampling/window metadata, and serialized feature configuration are
  not predictors.
- In each fold, exact duplicate columns are identified using the training
  partition only. The first in original column order is retained; subsequent
  exact copies are excluded and written to a fold-specific report. No
  correlated-but-non-identical features are removed. This per-fold handling
  avoids consulting held-out values to decide feature identity.

Reported test metrics are defined as follows: confusion matrix with rows as
true labels and columns as predictions in `[non_seizure, seizure]` order;
accuracy as correct predictions divided by all predictions; balanced accuracy
as the mean of seizure sensitivity and non-seizure specificity; precision as
`TP/(TP+FP)`; recall/sensitivity as `TP/(TP+FN)`; specificity as
`TN/(TN+FP)`; F1 as `2TP/(2TP+FP+FN)`; ROC-AUC from seizure probabilities;
and average precision from seizure probabilities and binary labels. A metric
whose denominator is zero is represented as null and listed as undefined,
not silently set to zero. Both-class prechecks guarantee ROC-AUC and average
precision have both true classes available.

Predicted class probabilities and true labels are saved for every held-out
window. Confusion-matrix plots, per-fold metrics, duplicate-feature reports,
the fold-comparison CSV, and run configuration are written under
`data/processed/evaluation/random_forest_baseline/`.

The generated feature matrix contains 25 exact duplicate feature-column pairs.
The current fold-wise training-only detection retains one column per duplicate
group, reports each excluded column and its retained counterpart, and retains
527 of 552 features per fold.
Most duplicates arise from the two sample-identical `T8-P8`
channels; the remaining duplicate pair is the dominant-frequency feature for
the reversed `T7-P7` / `P7-T7` derivations. The source feature matrix is not
modified. Duplicate removal is not a substitute for feature selection.

The first run produced the following held-out results; these are reported to
make the implementation auditable, not as reproduction of the paper's scores:

| Fold / held-out recording | Confusion matrix (TN, FP; FN, TP) | Accuracy | Balanced accuracy | Precision | Recall / sensitivity | Specificity | F1 | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 / `chb01_16` | (1304, 113; 0, 22) | 0.9215 | 0.9601 | 0.1630 | 1.0000 | 0.9203 | 0.2803 | 0.9923 | 0.8323 |
| Fold 2 / `chb01_18` | (1402, 0; 25, 12) | 0.9826 | 0.6622 | 1.0000 | 0.3243 | 1.0000 | 0.4898 | 0.9858 | 0.9668 |

These metrics use the fixed 0.5 threshold. Fold 1's 113 false positives and
fold 2's 25 false negatives illustrate why accuracy alone is insufficient
under severe class imbalance. High ranking metrics do not eliminate the
limitations of two held-out recordings/events or support generalization
claims.

This evaluation is recording-wise but not patient-wise. Both test recordings
and all training recordings belong to `chb01`, so the folds cannot establish
patient-independent validation. Only one seizure-containing recording is
available for training in each fold, each test fold contains a single
annotated seizure event, and there are only 22 or 37 positive windows against
1,402–1,417 negative windows. The substantial overlap between adjacent
windows also makes window observations dependent. Thus the reported metrics
are exploratory, potentially high-variance, and not a clinical performance
claim.
The protocol must not be presented as the paper's actual split or as exact
reproduction of its reported scores.

### Feature-family ablation reconstruction

The standalone `src/feature_family_ablation.py` experiment compares the
reconstruction's time-domain, frequency-domain, and wavelet-domain feature
families separately under the same two recording-wise folds and saved Random
Forest configuration. The source feature matrix is not changed. Its signal
columns are partitioned by the feature suffixes defined in the feature
extractor: four time features per channel (92 total), eight frequency
features per channel (184 total), and twelve wavelet features per channel
(276 total). All metadata and target columns remain excluded.

Each individual family is fitted and evaluated independently per fold. Exact
duplicate columns are removed using only that experiment's training
partition; non-identical correlated features are retained. The test
recordings are `chb01_16` and `chb01_18`, with all windows from each recording
kept together. The same saved 300-tree balanced Random Forest, random seed
42, and fixed `P(seizure) >= 0.5` rule are used. No feature selection,
hyperparameter tuning, threshold tuning, or test-informed preprocessing is
performed.

The all-family combined result is not newly trained: it is read from the
existing Random Forest baseline's saved held-out predictions and metrics,
independently checked, and presented as a reference. Since that baseline did
not save durations, its training and prediction durations remain unavailable;
durations for the three new family experiments are measured at execution.
Artifacts are written separately to
`data/processed/evaluation/feature_family_ablation/`, including per-fold
metrics, two-fold summary, confusion matrices, ROC and precision-recall plots,
duplicate reports, run configuration, and report.

This comparison is exploratory and within-patient. There is only one patient
and two seizure-containing recordings, with a single seizure-bearing
recording available for training in each fold. Overlapping windows are
dependent and seizure-window counts are small. A higher score for any family
in these folds does not imply universal or patient-independent superiority.
The feature-family definitions themselves are reconstruction choices because
the paper does not fully specify the exact FFT and wavelet feature summaries.
This experiment does not include the separate `f_classif` feature-selection
experiment.

## Ambiguous or missing implementation details

| Area | What is not established by the paper |
|---|---|
| Experimental cohort | Which patients, cases, recordings, channels, or seizure events were actually used for the reported scores; whether the study used the entire database. |
| EDF conversion | Exact CSV schema, channel inclusion/exclusion, units, handling of discontinuities, missing samples, and whether time columns are per-recording. |
| Preprocessing | No reproducible filter settings, artifact handling, rereferencing, normalization, resampling, or channel-quality policy are specified in the supplied methods. |
| Time-domain features | The feature window/chunk size, overlap, channel-wise versus pooled calculation, and whether maximum/minimum are signed extrema or absolute magnitudes are not consistently clear from the equations. |
| FFT features | Window length, taper, detrending, one- versus two-sided spectrum, normalization, frequency resolution, selected bins/bands, and how magnitudes become a fixed-length feature vector are unspecified. The frequency and power definitions above are reconstruction choices. |
| Wavelet features | The wavelet family, decomposition/boundary mode, exact level-3 coefficient sets, coefficient statistics/selection, and feature-vector shape are unspecified. The `db4`/symmetric mode and coefficient summaries above are reconstruction choices. The text calls this a discrete wavelet decomposition but displays a continuous-wavelet formula. |
| Labels and chunking | The seizure/non-seizure label construction, handling of windows crossing seizure boundaries, treatment of preictal/postictal periods, chunk size, overlap, and class balancing are unspecified. |
| Feature selection | Dummy-target construction, `k`, scoring/p-value use, feature-set scope, and fitting within training data are unspecified. The selection description names frequency and wavelet features, while the results also report time-domain accuracy; the relationship is not explained. |
| Random Forest | Number of trees, criterion, feature sampling, depth, leaf constraints, class weights, random seed, and other model parameters are not reported as experiment settings. |
| Evaluation split | The paper does not clearly specify a train/test ratio, split unit, grouping by patient or recording, stratification, cross-validation, random seed, or leakage controls. No random or patient-wise split can be inferred from the text. |
| Metrics | Averaging convention, positive class, decision threshold, probability method for ROC-AUC, and the relationship between the confusion matrix and feature-group scores are not clear. |
| Training curves | The paper describes accuracy/loss over 50 epochs, but a conventional Random Forest fit is not epoch-based. The curves' generation procedure is not specified. |

## Remaining reconstruction decisions

The preceding sections record current implementation choices. These do not
resolve what the original paper did. For future experiments, keep the
remaining decisions explicit and separate from any paper-faithful claims:

1. Establish the original paper's actual patient/recording/channel cohort,
   if it can be recovered; the current local cohort is only seven recordings
   from one patient.
2. Resolve the paper's “dummy target” before implementing its named
   `SelectKBest(f_regression)` procedure. Do not invent a target.
3. If an exploratory supervised selector is approved, choose its feature-set
   scope and `k`, and fit it strictly inside each training partition.
4. Do not attribute this baseline's chosen window labeling, FFT/wavelet
   features, Random Forest hyperparameters, recording-wise folds, threshold,
   or metric conventions to the paper.
5. Keep preprocessing alternatives separate from the documented baseline
   unless the original method is verified in source material.

## Current segmentation reconstruction (not paper-reported)

The paper does not specify a window duration, overlap, window alignment,
partial-window handling, or rule for assigning seizure labels to windows.
Consequently, none of the current segmentation settings should be attributed
to the paper.

The separate `src/segment_windows.py` stage uses these explicit choices:

- It reads seizure intervals only from passing rows in
  `results/chb01_annotation_audit.csv`. Confirmed no-seizure recordings remain
  distinct from recordings whose annotations are missing, unavailable, or
  invalid; the latter receive `unknown` window labels.
- Its baseline geometry is 5 seconds with 50% overlap, represented as 1,280
  samples and a 640-sample stride at 256 Hz. Boundaries are deterministic and
  anchored at sample zero. Incomplete trailing windows are omitted.
- Window sample intervals and expert seizure intervals use half-open
  semantics `[start, end)`. Thus a window ending exactly at onset has no
  overlap, and one beginning exactly at offset has no overlap.
- Its default `any_overlap` rule labels a window as seizure when its overlap
  with an expert interval has positive duration. It also exposes
  `majority_overlap` (strictly more than half the window) and
  `center_in_seizure` policies for explicit comparisons.
- The output is a metadata manifest, not a copied or transformed EEG dataset.
  The window size, overlap, stride, policy, and ordered channel-label list are
  recorded in each row; labels are serialized as JSON so repeated labels are
  preserved. EDF headers are checked for equal EEG sampling frequencies and
  sample counts; no channel is resampled or dropped to make windows fit.

These settings are intended as a reproducible baseline reconstruction while
the paper remains underspecified. They do not establish that the authors used
these settings.

### CHB-MIT annotation handling and identity

The shared `src/annotation_utils.py` parser is used by the annotation audit,
EDF inspection, and corpus manifest. It parses integer and decimal
recording-relative summary times, multiple seizure intervals, and each
patient's own summary file. Values are retained as parsed without rounding at
ingestion. WFDB `.edf.seizures` files are parsed only through `wfdb.rdann`;
sample markers are converted to seconds only when the applicable sampling
frequency is known.

Corpus-manifest and segmentation lookups identify recordings by patient ID,
recording ID, and dataset-relative EDF path, rather than a basename alone.
The older patient-scoped annotation audit CSV remains supported. A usable
summary is sufficient when a sidecar is absent; absence of a sidecar is not
an annotation failure. When both sources exist, the audit and manifest retain
summary and sidecar intervals separately and record agreement, disagreement,
or an unavailable comparison. A disagreement is ambiguous and therefore
produces an unknown annotation for segmentation; neither source is silently
selected as truth.

A summary explicitly declaring zero seizures is distinguished from a missing
summary, missing seizure count, incomplete times, or unreadable sidecar. Those
conditions remain unknown/unavailable and are not converted to negative
labels. Invalid, reversed, negative, overlapping, duplicate, or
out-of-duration intervals are reported; non-passing audit rows remain unknown
to the legacy segmentation workflow. These are data-integrity rules for this
reconstruction, not additional methods attributed to the paper.

## Corpus processing and resume behavior

`src/process_chbmit.py` is an operational wrapper around the existing EDF
converter, window segmenter, and feature extractor; it does not introduce new
feature definitions, labels, or model behavior. It selects records from the
corpus manifest using patient ID and dataset-relative recording path, and
processes each EDF independently in conversion, segmentation, then feature
extraction order. Patient-scoped output directories under
`data/processed/corpus_pipeline/` isolate same-named recordings and leave the
legacy CHB01 outputs and evaluation artifacts untouched.

The state JSON stores each stage's status, input SHA-256, canonical
configuration checksum, output path and SHA-256, row count, timestamps, and
error/retry information. State writes use a temporary file and atomic replace.
After restart, a completed stage is reused only when its input and
configuration match and its output still exists with the recorded checksum.
Running or failed work is retried, and changed inputs/configurations receive a
separate output directory rather than replacing an earlier result. Missing
EDFs and annotation-unavailable or conflicting records receive explicit
`Skipped` stages and reasons; only validated seizure or validated no-seizure
annotations are passed into segmentation.

Memory handling is stage-specific. EDF-to-CSV conversion reads chunks,
segmentation writes each window row as it is generated, and feature extraction
reads one recording's data in chunks. The orchestrator is serial and does not
load the corpus or multiple signal recordings simultaneously. The separate
exploratory `src/preprocessing.py` stage loads one complete recording for
continuous filtering; before that allocation it estimates three float64
signal copies and rejects recordings above a default 2-GiB estimate. This is
an estimate guard, not a hard cap on all process memory. Streaming window
metadata output avoids retaining all windows from all recordings in memory.
None of these engineering choices are methodologies attributed to the paper.

## Multi-patient features and dataset assembly

The feature extractor keys window rows by the composite
`(patient_id, recording_id, relative_path)` identity; a recording basename is
not treated as unique. It validates one channel schema per recording but no
longer requires every recording to have the same ordered montage. Existing
feature calculations and the legacy column names remain unchanged when all
recordings have the same ordered labels. For differing layouts, output
columns use a deterministic label-aligned schema.

Channel matching uppercases labels and trims/collapses whitespace only. It
does not reorder electrodes, infer equivalence, or merge reversed bipolar
derivations. Duplicate normalized labels use occurrence order within each
recording, which preserves repeated channels but cannot establish their
physical equivalence across montages. Original labels, per-recording channel
indices, and the mapping to predictor names are persisted in
`feature_schema.json`. Feature columns absent from a recording are `NaN`, not
zero.

`src/assemble_feature_dataset.py` reads per-recording feature outputs from the
P2 state file, verifies their recorded output checksums and row identities,
and streams one file at a time in bounded batches. It checks row count,
window sequence, required metadata, and consistent feature configuration.
Missing, corrupt, incomplete, duplicate, or incompatible outputs are
identified in `recordings_manifest.csv`; corpus identity uses patient,
recording ID, and relative path together. The assembled schema explicitly
separates predictor columns from patient/recording identity, window metadata,
target labels, and annotation status. No imputation, normalization, feature
selection, or model fitting occurs. Any learned transformation or imputation
in later validation must be fitted on training patients only within each
cross-validation fold.

The currently downloaded CHB01 EDFs were inspected and all seven have the
same 23-channel order, including a repeated `T8-P8` label. This observation
applies only to the local partial download and is not an assumption about the
complete CHB-MIT corpus. The extractor and assembly schema are designed to
preserve montage differences rather than infer a universal montage.
The existing Random Forest runner remains the CHB01 recording-wise baseline;
the assembled corpus matrix must not be passed to it as a substitute for a
patient-wise evaluation. A separate patient-wise split and evaluation
implementation is still needed before full-corpus model fitting.

## Research-integrity boundary

This project is a reconstruction/reimplementation, not a recovery of the
paper's original source code. The paper's reported results are not evidence
that this partial CHB01 download will reproduce those results or generalize
to other patients. The present project must not claim either conclusion.

The current work does not implement patient-wise splitting. The first
recording-wise baseline is implemented, but its results are only exploratory
and do not establish generalization. The segmentation manifest records labels
derived from validated expert-provided annotations; it does not create
synthetic labels. Dataset inspection, annotation auditing, exploratory
preprocessing, feature extraction, and model evaluation remain distinct
stages. Baseline reproduction and methodological improvements should be
tracked as separate experiments with their own recorded parameters and
limitations.

## Unresolved methodological questions

1. What exactly did the paper mean by “dummy target” for `f_regression`? Is
   there an original notebook, code listing, or author clarification that
   identifies it?
2. What is the intended feature-selection `k`, and was it applied separately
   to each feature family? Was the selector fit on training data only?
3. Which patients/EDF files and channels formed the actual 120-example
   confusion-matrix evaluation set, and how does that reconcile with the
   larger database description and reported per-feature accuracies?
4. What window/chunk length, overlap, seizure-boundary labeling rule, and
   class-balancing procedure did the authors use? These are not established by
   the paper. This reconstruction separately records the choices in the
   segmentation section above; they must not be presented as the authors'
   settings.
5. Which FFT bins or frequency summaries and which wavelet family/coefficient
   summaries were used?
6. What preprocessing, if any, was applied before extracting features?
7. What Random Forest settings, split unit/proportion, validation protocol,
   seed, positive-class convention, and metric averaging did the paper use?
   The local baseline's settings are reconstruction choices and do not answer
   this question.
8. How were the reported 50-epoch accuracy/loss curves generated for the
   Random Forest workflow?
9. What publication venue, year, and DOI should be recorded in the final
   bibliography? Those details are not identified in the supplied PDF copy.
