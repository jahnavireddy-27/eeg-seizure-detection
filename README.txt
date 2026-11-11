# EEG-Based Seizure Detection Using Machine Learning

## Project Overview

This project explores the detection of epileptic seizures from electroencephalogram (EEG) signals using signal processing and machine learning techniques.

This project was developed using the CHB-MIT Scalp EEG Database. The pipeline processes EEG recordings, extracts time-domain, frequency-domain, and wavelet-domain features, and uses a Random Forest classifier to distinguish seizure from non-seizure activity.

## Dataset

**CHB-MIT Scalp EEG Database**

The dataset is publicly available through PhysioNet:

https://physionet.org/content/chbmit/1.0.0/

This implementation uses 19 EEG recordings from 3 patients (CHB01, CHB02, and CHB03). The raw dataset is not included in this repository and must be downloaded separately.

## Methodology

- EEG recordings are divided into 5-second windows with 50% overlap.
- Time-domain, frequency-domain, and wavelet-domain features are extracted from each window.
- Extracted features are used to train a Random Forest classifier.
- Performance is evaluated using four-fold recording-held-out cross-validation.

## Evaluation Results

The baseline Random Forest model achieved the following pooled out-of-fold results:

| Metric | Score |
|---|---:|
| Accuracy | 99.44% |
| Balanced Accuracy | 73.23% |
| Sensitivity | 46.60% |
| Specificity | 99.86% |
| Precision | 72.73% |
| F1-score | 56.80% |
| ROC-AUC | 97.42% |
| Average Precision | 57.76% |

**Confusion Matrix**

| | Predicted Non-Seizure | Predicted Seizure |
|---|---:|---:|
| Actual Non-Seizure | 26,042 | 36 |
| Actual Seizure | 110 | 96 |

A total of 26,284 windows were evaluated, including 206 seizure-positive windows.

Research Publication : https://ieeexplore.ieee.org/document/10870633 
