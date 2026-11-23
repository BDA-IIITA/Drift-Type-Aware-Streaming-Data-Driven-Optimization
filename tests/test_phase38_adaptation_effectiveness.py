import numpy as np
import pandas as pd

from griddrift.forecasting.adaptation_analysis import (
    build_feature_matrix_for_issues,
    compare_absolute_errors,
)
from griddrift.forecasting.features import DirectFeatureSpec, build_feature_row


def test_vectorized_features_match_canonical_builder_and_ignore_future_values():
    index = pd.date_range("2024-01-01", periods=800, freq="h", tz="UTC")
    demand = pd.Series(50_000 + np.sin(np.arange(len(index)) / 8) * 1000, index=index)
    issues = index[300:305]
    spec = DirectFeatureSpec()
    features = build_feature_matrix_for_issues(demand, issues, spec)
    expected = build_feature_row(demand, issues[0], spec)
    assert np.allclose(features.iloc[0].loc[list(expected)].to_numpy(), list(expected.values()))
    changed = demand.copy()
    changed.loc[changed.index > issues[-1]] = 1_000_000
    altered = build_feature_matrix_for_issues(changed, issues, spec)
    pd.testing.assert_frame_equal(features, altered)


def test_paired_error_comparison_detects_improvement_and_degradation():
    actual = np.full(168, 50_000.0)
    previous = actual + 1_000
    improved = actual + 100
    worse = actual + 2_000
    improved_result = compare_absolute_errors(actual, previous, improved, bootstrap_replicates=250)
    worse_result = compare_absolute_errors(actual, previous, worse, bootstrap_replicates=250)
    assert improved_result["effect"] == "improved"
    assert worse_result["effect"] == "worsened"
    assert improved_result["sample_count"] == 168


def test_paired_error_comparison_marks_short_window_inconclusive():
    actual = np.full(12, 50_000.0)
    result = compare_absolute_errors(actual, actual + 10, actual, bootstrap_replicates=100)
    assert result["effect"] == "inconclusive"
