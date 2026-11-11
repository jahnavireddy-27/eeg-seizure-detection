import numpy as np
import pandas as pd
import pytest

from src import run_expanded_random_forest_weighting as weighting


def _fold():
    return pd.DataFrame([
        {"patient_id": "p1", "recording_id": "r1", "relative_path": "p1/r1.edf", "window_index": 0,
         "start_time_seconds": 0.0, "end_time_seconds": 5.0},
        {"patient_id": "p1", "recording_id": "r1", "relative_path": "p1/r1.edf", "window_index": 1,
         "start_time_seconds": 5.0, "end_time_seconds": 10.0},
        {"patient_id": "p1", "recording_id": "r2", "relative_path": "p1/r2.edf", "window_index": 0,
         "start_time_seconds": 0.0, "end_time_seconds": 5.0},
        {"patient_id": "p1", "recording_id": "r2", "relative_path": "p1/r2.edf", "window_index": 1,
         "start_time_seconds": 5.0, "end_time_seconds": 10.0},
        {"patient_id": "p2", "recording_id": "r3", "relative_path": "p2/r3.edf", "window_index": 0,
         "start_time_seconds": 0.0, "end_time_seconds": 5.0},
        {"patient_id": "p2", "recording_id": "r3", "relative_path": "p2/r3.edf", "window_index": 1,
         "start_time_seconds": 5.0, "end_time_seconds": 10.0},
    ])


def test_event_balanced_weights_equal_event_masses_and_class_masses():
    frame = _fold()
    labels = np.array([1, 1, 0, 0, 0, 0], dtype=np.int8)
    events = {
        ("p1", "r1", "p1/r1.edf"): [{"event_id": "e1", "event_start_seconds": 0, "event_end_seconds": 10}],
        ("p1", "r2", "p1/r2.edf"): [],
        ("p2", "r3", "p2/r3.edf"): [],
    }
    weights, rows = weighting.event_balanced_weights(frame, labels, events)
    assert np.isclose(weights[labels == 1].sum(), 3.0)
    assert np.isclose(weights[labels == 0].sum(), 3.0)
    assert rows[0]["event_id"] == "e1"
    assert np.isclose(rows[0]["positive_class_weight_fraction"], 1.0)
    assert np.allclose(weights[:2], [1.5, 1.5])


def test_recording_balanced_weights_equalize_each_recording_within_class():
    frame = _fold()
    labels = np.array([1, 1, 0, 0, 0, 0], dtype=np.int8)
    weights = weighting.recording_balanced_weights(frame, labels)
    assert np.isclose(weights[labels == 1].sum(), 3.0)
    assert np.isclose(weights[labels == 0].sum(), 3.0)
    assert np.isclose(weights[:2].sum(), 3.0)
    assert np.isclose(weights[2:4].sum(), 1.5)
    assert np.isclose(weights[4:].sum(), 1.5)


def test_event_balanced_weights_reject_positive_window_without_event():
    frame = _fold()
    labels = np.array([1, 1, 0, 0, 0, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="summary event"):
        weighting.event_balanced_weights(frame, labels, {})
