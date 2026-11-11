import numpy as np

from src import investigate_missed_seizure_windows as investigation


def test_robust_summary_uses_finite_values_and_reports_dispersion():
    result = investigation.robust_summary([1, 2, 3, 4, np.nan])
    assert result["n"] == 4
    assert result["median"] == 2.5
    assert result["iqr"] == 1.5
    assert result["mad"] == 1.0


def test_cliffs_delta_reports_order_and_ties():
    assert investigation.cliffs_delta([3, 4], [1, 2]) == 1.0
    assert investigation.cliffs_delta([1, 2], [3, 4]) == -1.0
    assert investigation.cliffs_delta([1, 1], [1, 1]) == 0.0
    assert investigation.cliffs_delta([], [1]) is None


def test_feature_domain_maps_existing_feature_families():
    assert investigation.feature_domain("channel__C3-P3__mean") == "time_amplitude"
    assert investigation.feature_domain("channel__C3-P3__band_power_alpha") == "frequency"
    assert investigation.feature_domain("channel__C3-P3__wavelet_D3_energy") == "wavelet"
