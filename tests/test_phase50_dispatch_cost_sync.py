import numpy as np
import pandas as pd
import pytest

from griddrift.dashboard.views import (
    daily_planned_cost_eur,
    demand_chart_frame,
    dispatch_download_frame,
    validate_forecast_plan_alignment,
)
from griddrift.dispatch import load_dispatch_config
from griddrift.forecasting.features import DirectFeatureSpec
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.forecasting.xgboost import DirectXGBoostForecaster
from griddrift.simulation import IntegratedConfig, IntegratedSimulation
from scripts.streamlit_app import _dispatch_table


class _TestMLP(DirectMLPForecaster):
    def predict_horizon(self, history, issue_time):
        targets = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=24, freq="h")
        return pd.Series(float(history.iloc[-1]) - 500.0, index=targets, name="predicted_demand_mw")


class _TestXGBoost(DirectXGBoostForecaster):
    def predict_horizon(self, history, issue_time):
        targets = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=24, freq="h")
        return pd.Series(float(history.iloc[-1]) + 500.0, index=targets, name="predicted_demand_mw")


def _simulation(model_family):
    index = pd.date_range("2025-01-01", periods=500, freq="h", tz="UTC")
    demand = pd.Series(
        50_000.0 + 1_500.0 * np.sin(np.arange(len(index)) * 2 * np.pi / 24),
        index=index,
        name="demand_mw",
    )
    spec = DirectFeatureSpec()
    if model_family == "xgboost":
        model = _TestXGBoost(
            models=[], feature_spec=spec, model_version="xgb-cost-test-v1", feature_columns=[],
            train_end_issue_time=index[100], parameters={}, multi_output=True,
        )
    else:
        model = _TestMLP(
            model=None, feature_scaler=None, target_scaler=None, feature_spec=spec,
            model_version="mlp-cost-test-v1", feature_columns=[], train_end_issue_time=index[100],
        )
    assets, _ = load_dispatch_config("configs/assets.yaml")
    return IntegratedSimulation(
        demand, model, assets,
        config=IntegratedConfig(demand_scale=0.001, policy="adaptive"),
        start_timestamp=index[300], model_family=model_family,
    )


@pytest.mark.parametrize("model_family", ["mlp", "xgboost"])
def test_selected_model_forecast_is_the_same_plan_and_cost_horizon(model_family):
    simulation = _simulation(model_family)
    validate_forecast_plan_alignment(simulation)
    forecast_index = pd.date_range(simulation.timestamp + pd.Timedelta(hours=1), periods=24, freq="h")

    assert simulation.forecast.index.equals(forecast_index)
    assert simulation.plan.schedule.index.equals(forecast_index)
    np.testing.assert_allclose(
        simulation.plan.schedule.demand_mw.to_numpy(),
        simulation.forecast.to_numpy() * simulation.config.demand_scale,
    )

    chart = demand_chart_frame(simulation)
    assert chart.loc[forecast_index, "predicted_demand_mw"].notna().all()
    assert chart.loc[forecast_index, "actual_demand_mw"].isna().all()
    downloaded = dispatch_download_frame(simulation)
    assert pd.DatetimeIndex(downloaded.timestamp).equals(forecast_index)
    table = _dispatch_table(simulation)
    assert pd.DatetimeIndex(table.timestamp).equals(forecast_index)
    np.testing.assert_allclose(
        table["Predicted demand (MW)"].to_numpy(),
        simulation.forecast.to_numpy(),
        atol=0.001,
    )

    cost_columns = [
        "cost_solar_eur", "cost_wind_eur", "cost_hydro_eur", "cost_conventional_eur",
        "cost_battery_degradation_eur", "cost_emergency_eur", "cost_unserved_voll_eur",
    ]
    assert simulation.plan.schedule.cost_total_eur.sum() == pytest.approx(
        simulation.plan.schedule[cost_columns].to_numpy().sum()
    )
    today = simulation.timestamp.date()
    expected_daily = simulation.plan.schedule.loc[
        simulation.plan.schedule.index.date == today, "cost_total_eur"
    ].sum()
    assert daily_planned_cost_eur(simulation) == pytest.approx(expected_daily)


def test_plan_alignment_guard_rejects_a_different_target_timestamp():
    simulation = _simulation("xgboost")
    simulation.plan.schedule.index = simulation.plan.schedule.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="identical target timestamps"):
        validate_forecast_plan_alignment(simulation)
