import numpy as np
import pandas as pd
import pytest

from griddrift.data.synthetic import make_synthetic_scenario
from griddrift.forecasting.features import DirectFeatureSpec, build_supervised_dataset
from griddrift.forecasting.mlp import train_direct_mlp


def _demand(periods=24 * 16):
    frame, _ = make_synthetic_scenario(periods=periods, seed=11)
    return frame["demand_mw"]


def test_feature_rows_do_not_use_future_demand():
    demand = _demand()
    X, _ = build_supervised_dataset(demand)
    issue = X.index[5]
    corrupted = demand.copy()
    corrupted.loc[corrupted.index > issue] = 1_000_000.0
    corrupted_X, _ = build_supervised_dataset(corrupted)
    pd.testing.assert_series_equal(X.loc[issue], corrupted_X.loc[issue])


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_direct_mlp_has_24_outputs_and_chronological_scaler_fit():
    demand = _demand()
    result = train_direct_mlp(
        demand,
        hidden_layer_sizes=(12,),
        max_iter=40,
        random_state=5,
    )
    prediction = result.model.predict_horizon(demand, demand.index[-1])
    assert prediction.shape == (24,)
    assert prediction.index[0] == demand.index[-1] + pd.Timedelta(hours=1)
    assert prediction.index[-1] == demand.index[-1] + pd.Timedelta(hours=24)
    assert result.model.train_end_issue_time < demand.index[-24]
    assert np.all(prediction.to_numpy() >= 0)
