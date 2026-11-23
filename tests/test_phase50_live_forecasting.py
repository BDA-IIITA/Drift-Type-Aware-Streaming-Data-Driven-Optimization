from types import SimpleNamespace

import numpy as np
import pandas as pd

from griddrift.dashboard import model_selection
from griddrift.dispatch import load_dispatch_config
from griddrift.forecasting.features import DirectFeatureSpec
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.forecasting.xgboost import DirectXGBoostForecaster
from griddrift.simulation import integrated
from griddrift.simulation.integrated import IntegratedConfig, IntegratedSimulation


class _PredictingXGBoost(DirectXGBoostForecaster):
    def predict_horizon(self, history, issue_time):
        issue = pd.Timestamp(issue_time)
        assert history.index.max() <= issue
        targets = pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="1h")
        base = float(history.loc[:issue].iloc[-1])
        return pd.Series(base + np.arange(24), index=targets, name="predicted_demand_mw")


class _PredictingMLP(DirectMLPForecaster):
    def predict_horizon(self, history, issue_time):
        issue = pd.Timestamp(issue_time)
        targets = pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="1h")
        return pd.Series(500.0, index=targets, name="predicted_demand_mw")


def _xgb_model():
    return _PredictingXGBoost(
        models=[], feature_spec=DirectFeatureSpec(), model_version="xgb-test-v1",
        feature_columns=[], train_end_issue_time=pd.Timestamp("2025-01-05T04:00:00Z"),
        parameters={}, multi_output=True,
    )


def test_selected_forecaster_loads_only_matching_checkpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        DirectXGBoostForecaster, "load",
        classmethod(lambda cls, path: calls.append(("xgboost", path)) or "xgb-model"),
    )
    monkeypatch.setattr(
        DirectMLPForecaster, "load",
        classmethod(lambda cls, path: calls.append(("mlp", path)) or "mlp-model"),
    )
    assert model_selection.load_selected_forecaster(
        model_selection.ADAPTIVE_XGBOOST,
        mlp_checkpoint="mlp-dir", xgboost_checkpoint="xgb-dir",
    ) == "xgb-model"
    assert model_selection.load_selected_forecaster(
        model_selection.ADAPTIVE_MLP,
        mlp_checkpoint="mlp-dir", xgboost_checkpoint="xgb-dir",
    ) == "mlp-model"
    assert calls == [("xgboost", "xgb-dir"), ("mlp", "mlp-dir")]


def test_xgboost_live_forecast_is_full_issue_plus_24_and_past_only(monkeypatch):
    index = pd.date_range("2025-01-01", periods=500, freq="1h", tz="UTC")
    demand = pd.Series(1000 + np.arange(len(index)), index=index, name="demand_mw")
    assets, _ = load_dispatch_config("configs/assets.yaml")
    calls = []

    def fake_optimize_dispatch(demand_values, resources, active_assets, state, **kwargs):
        calls.append((np.asarray(demand_values), pd.DatetimeIndex(kwargs["timestamps"])))
        return SimpleNamespace(total_cost_eur=0.0)

    monkeypatch.setattr(integrated, "optimize_dispatch", fake_optimize_dispatch)
    issue = index[220]
    simulation = IntegratedSimulation(
        demand, _xgb_model(), assets,
        config=IntegratedConfig(policy="adaptive"),
        start_timestamp=issue,
        model_family="xgboost",
    )

    expected = pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="1h")
    assert simulation.model_family == "xgboost"
    assert simulation.active_model.model_version == "xgb-test-v1"
    assert simulation.forecast.index.equals(expected)
    assert simulation.forecast.tolist() == [float(demand.loc[issue]) + n for n in range(24)]
    assert calls[-1][1].equals(expected)
    assert np.array_equal(calls[-1][0], simulation.forecast.to_numpy() * simulation.config.demand_scale)
    assert simulation.timestamp == issue
    assert simulation.actual_history.index.max() == issue
    assert simulation.detector is not None  # The detector is per-family and is wired in Stage 4.


def test_xgboost_rejects_misaligned_forecast_horizon(monkeypatch):
    index = pd.date_range("2025-01-01", periods=500, freq="1h", tz="UTC")
    demand = pd.Series(1000.0, index=index, name="demand_mw")
    assets, _ = load_dispatch_config("configs/assets.yaml")
    model = _xgb_model()

    def wrong_horizon(history, issue_time):
        targets = pd.date_range(issue_time + pd.Timedelta(hours=2), periods=24, freq="1h")
        return pd.Series(np.ones(24), index=targets)

    model.predict_horizon = wrong_horizon
    monkeypatch.setattr(
        integrated, "optimize_dispatch",
        lambda *args, **kwargs: SimpleNamespace(total_cost_eur=0.0),
    )
    try:
        IntegratedSimulation(
            demand, model, assets,
            config=IntegratedConfig(policy="adaptive"), start_timestamp=index[220],
            model_family="xgboost",
        )
    except ValueError as exc:
        assert "exactly 24 UTC targets" in str(exc)
    else:
        raise AssertionError("misaligned XGBoost targets should be rejected")


def test_mlp_and_xgboost_contexts_do_not_share_detector_or_registries(monkeypatch):
    index = pd.date_range("2025-01-01", periods=500, freq="1h", tz="UTC")
    demand = pd.Series(1000.0, index=index, name="demand_mw")
    assets, _ = load_dispatch_config("configs/assets.yaml")
    monkeypatch.setattr(
        integrated, "optimize_dispatch",
        lambda *args, **kwargs: SimpleNamespace(total_cost_eur=0.0),
    )
    xgb_sim = IntegratedSimulation(
        demand, _xgb_model(), assets,
        config=IntegratedConfig(policy="adaptive"), start_timestamp=index[220], model_family="xgboost",
    )
    mlp_model = _PredictingMLP(
        model=None, feature_scaler=None, target_scaler=None, feature_spec=DirectFeatureSpec(),
        model_version="mlp-test-v1", feature_columns=[], train_end_issue_time=index[100],
    )
    mlp_sim = IntegratedSimulation(
        demand, mlp_model, assets,
        config=IntegratedConfig(policy="adaptive"), start_timestamp=index[220], model_family="mlp",
    )
    assert xgb_sim.detector is not mlp_sim.detector
    assert xgb_sim.static_registry is not mlp_sim.static_registry
    assert xgb_sim.adaptive_registry is not mlp_sim.adaptive_registry
    assert xgb_sim.adaptation_events is not mlp_sim.adaptation_events
