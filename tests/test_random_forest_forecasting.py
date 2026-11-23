import numpy as np
import pandas as pd
import pytest

from griddrift.forecasting.features import DirectFeatureSpec, build_feature_row
from griddrift.forecasting.random_forest import (
    DirectRandomForestForecaster,
    train_direct_random_forest,
)


@pytest.fixture
def demand():
    index = pd.date_range("2023-01-01", periods=600, freq="h", tz="UTC")
    values = 50_000 + 4_000 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    return pd.Series(values, index=index, name="demand_mw")


def tiny_parameters():
    return {"n_estimators": 4, "max_depth": 5, "n_jobs": 1, "min_samples_leaf": 1}


def test_random_forest_prediction_has_aligned_24_hour_horizon(demand):
    result = train_direct_random_forest(
        demand.iloc[:500], train_end_issue_time=demand.index[350],
        parameters=tiny_parameters(),
    )
    issue = demand.index[400]
    prediction = result.model.predict_horizon(demand.iloc[:401], issue)
    assert len(prediction) == 24
    assert prediction.index.equals(
        pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="h")
    )
    assert (prediction >= 0).all()


def test_features_ignore_values_after_issue_time(demand):
    spec = DirectFeatureSpec()
    issue = demand.index[300]
    extended = demand.copy()
    extended.loc[extended.index > issue] = 999_999
    assert build_feature_row(demand.loc[:issue], issue, spec) == build_feature_row(extended, issue, spec)


def test_training_uses_chronological_validation_and_full_target_embargo(demand):
    result = train_direct_random_forest(
        demand.iloc[:550], train_end_issue_time=demand.index[400],
        validation_fraction=0.2, parameters=tiny_parameters(),
    )
    assert result.model.train_end_issue_time <= demand.index[400]
    assert result.train_samples > result.validation_samples
    assert (
        result.validation_train_end_issue_time + pd.Timedelta(hours=24)
        < result.validation_first_issue_time
    )
    assert result.validation_metrics["mae_mw"] >= 0
    assert result.final_fit_samples > result.train_samples


def test_checkpoint_round_trip_and_fixed_seed_reproducibility(demand, tmp_path):
    kwargs = {
        "train_end_issue_time": demand.index[350],
        "parameters": tiny_parameters(),
        "random_state": 7,
    }
    first = train_direct_random_forest(demand.iloc[:500], **kwargs)
    repeat = train_direct_random_forest(demand.iloc[:500], **kwargs)
    assert first.model.model_version == repeat.model.model_version
    issue = demand.index[400]
    expected = first.model.predict_horizon(demand.iloc[:401], issue)
    pd.testing.assert_series_equal(repeat.model.predict_horizon(demand.iloc[:401], issue), expected)

    first.model.save(tmp_path, first.validation_metrics)
    restored = DirectRandomForestForecaster.load(tmp_path)
    pd.testing.assert_series_equal(restored.predict_horizon(demand.iloc[:401], issue), expected)
    assert restored.model_version == first.model.model_version
    assert (tmp_path / "metadata.json").exists()


def test_training_rejects_misaligned_or_short_labels(demand):
    from griddrift.forecasting.features import build_supervised_dataset

    X, y = build_supervised_dataset(demand.iloc[:300])
    with pytest.raises(ValueError, match="issue timestamps"):
        from griddrift.forecasting.random_forest import fit_direct_random_forest_samples

        fit_direct_random_forest_samples(X, y.iloc[::-1], parameters=tiny_parameters())
