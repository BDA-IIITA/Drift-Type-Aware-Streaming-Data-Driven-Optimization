from types import SimpleNamespace

import numpy as np
import pandas as pd

from griddrift.data import RenewableResourceProvider
from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch
from scripts import streamlit_app


def test_dashboard_can_load_processed_renewable_demand_schema():
    path = str(streamlit_app.DEFAULT_RENEWABLE_DATA)
    demand = streamlit_app._read_demand(path)
    bounds = streamlit_app._read_demand_bounds(path)

    assert len(demand) == 8_760
    assert demand.index[0] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert bounds == (demand.index[0], demand.index[-1])


def test_dashboard_historical_resource_mode_loads_all_three_inputs():
    provider, parameters, inflow = streamlit_app._load_resource_inputs(
        streamlit_app.HISTORICAL_RESOURCE_MODE,
        {"renewable_forecast_issue_policy": "previous_day_00_utc_assumption"},
    )

    assert isinstance(provider, RenewableResourceProvider)
    assert parameters.zone == "DE00"
    assert inflow.index[0] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert len(inflow) == 8_760
    assert streamlit_app._load_resource_inputs(
        streamlit_app.SIMULATED_RESOURCE_MODE, {}
    ) == (None, None, None)


def test_dashboard_limits_historical_replay_to_full_forecast_coverage():
    provider = RenewableResourceProvider(streamlit_app.DEFAULT_RENEWABLE_DATA)
    resource_index = provider.target_timestamps
    demand_index = pd.date_range("2024-12-01", periods=10_000, freq="h", tz="UTC")
    demand = pd.Series(np.ones(len(demand_index)), index=demand_index)

    limited = streamlit_app._limit_demand_to_resource_horizon(demand, provider)

    assert limited.index[-1] == resource_index[-1] - pd.Timedelta(hours=24)
    assert limited.iloc[0] == demand.iloc[0]


def test_plan_download_exposes_hydro_inflow_and_separate_storage_state():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    assets, _ = load_dispatch_config("configs/assets.yaml")
    plan = optimize_dispatch(
        np.full(24, 10.0),
        {"solar": np.zeros(24), "wind": np.zeros(24)},
        assets,
        DispatchState(battery_soc_mwh=3.0, hydro_remaining_mwh=100.0),
        timestamps=index,
        hydro_inflow_mw=np.full(24, 0.2),
        terminal_soc_policy="none",
    )
    sim = SimpleNamespace(
        forecast=pd.Series(np.full(24, 10.0), index=index), plan=plan, assets=assets
    )

    table = streamlit_app._dispatch_table(sim)

    assert "Hydro inflow (MW)" in table
    assert "Hydro remaining (MWh)" in table
    assert "Battery SOC (MWh)" in table
    assert table["Hydro inflow (MW)"].eq(0.2).all()
