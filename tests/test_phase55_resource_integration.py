from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from griddrift.data.hydro import load_reservoir_parameters_json
from griddrift.data.renewables import RenewableResourceProvider
from griddrift.dispatch import load_dispatch_config
from griddrift.simulation import IntegratedConfig, IntegratedSimulation


@dataclass
class ForecastSpec:
    lookback_hours: int = 168
    horizon_hours: int = 24


class SimpleModel:
    def __init__(self, index):
        self.feature_spec = ForecastSpec()
        self.model_version = "phase55-test"
        self.train_end_issue_time = index[-120]

    def predict_horizon(self, history, issue_time):
        target = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=24, freq="h")
        return pd.Series(np.full(24, float(history.iloc[-1])), index=target)


def _resource_data(index):
    return pd.DataFrame(
        {
            "target_timestamp_utc": index,
            "solar_actual_mw": np.full(len(index), 2_000.0),
            "wind_actual_mw": np.full(len(index), 3_000.0),
            "solar_day_ahead_forecast_mw": np.full(len(index), 1_000.0),
            "wind_day_ahead_forecast_mw": np.full(len(index), 4_000.0),
            "solar_forecast_issue_timestamp_utc": [pd.NaT] * len(index),
            "wind_forecast_issue_timestamp_utc": [pd.NaT] * len(index),
        }
    )


def test_integrated_replay_uses_scaled_forecasts_actuals_hydro_and_same_24h_plan():
    index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    renewable = RenewableResourceProvider(_resource_data(index))
    parameters = load_reservoir_parameters_json(
        "data/processed/pecd_de00_reservoir_parameters.json"
    )
    inflow = pd.Series(np.full(len(index), 100.0), index=index)
    assets, _ = load_dispatch_config("configs/assets.yaml")
    simulation = IntegratedSimulation(
        demand,
        SimpleModel(index),
        assets,
        config=IntegratedConfig(demand_scale=0.001, policy="adaptive"),
        renewable_provider=renewable,
        hydro_parameters=parameters,
        hydro_inflow_profile=inflow,
    )

    schedule = simulation.plan.schedule
    assert simulation.resource_data_mode == "entsoe_pecd"
    assert simulation.timestamp >= SimpleModel(index).train_end_issue_time + pd.Timedelta(hours=24)
    assert len(simulation.forecast) == len(schedule) == 24
    assert simulation.forecast.index.equals(schedule.index)
    assert simulation.assets.hydro.capacity_mw == pytest.approx(1.297)
    assert simulation.assets.hydro.reservoir_capacity_mwh == pytest.approx(258.0)
    assert simulation.state.hydro_remaining_mwh == pytest.approx(129.0)
    assert schedule.solar_available_mw.eq(1.0).all()
    assert schedule.wind_available_mw.eq(4.0).all()
    assert schedule.hydro_generation_limit_mw.eq(1.297).all()
    assert schedule.hydro_inflow_mw.eq(0.1).all()
    assert simulation.resource_forecast_details.solar_forecast_issue_time_assumed.all()

    starting_hydro = simulation.state.hydro_remaining_mwh
    simulation.step()
    realized = simulation.realized[-1]
    assert realized["resource_data_source"] == "ENTSO-E realized generation"
    assert realized["solar_available_mw"] == pytest.approx(2.0)
    assert realized["wind_available_mw"] == pytest.approx(3.0)
    assert realized["hydro_inflow_mw"] == pytest.approx(0.1)
    assert realized["ending_hydro_remaining_mwh"] == pytest.approx(
        starting_hydro + 0.1 - realized["hydro_mw"] - realized["hydro_spill_mw"]
    )


def test_historical_provider_requires_hydro_inputs_and_complete_target_coverage():
    index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    assets, _ = load_dispatch_config("configs/assets.yaml")
    with pytest.raises(ValueError, match="requires PECD"):
        IntegratedSimulation(
            demand,
            SimpleModel(index),
            assets,
            config=IntegratedConfig(demand_scale=0.001),
            renewable_provider=RenewableResourceProvider(_resource_data(index)),
        )
