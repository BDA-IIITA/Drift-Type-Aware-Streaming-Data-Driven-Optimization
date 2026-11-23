from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch
from griddrift.simulation import load_integrated_settings
from scripts import streamlit_app

FINAL_DIR = Path("artifacts/experiments/phase55/final_integration")


def test_final_processed_inputs_are_utc_aligned_and_have_explicit_hydro_scenario_fields():
    renewable = pd.read_csv(FINAL_DIR / "entsoe_de_lu_2025_renewables_hourly.csv")
    hydro = pd.read_csv(FINAL_DIR / "pecd_de00_synthetic_hourly_hydro_2025.csv")
    renewable_time = pd.DatetimeIndex(pd.to_datetime(renewable.target_timestamp_utc, utc=True))
    hydro_time = pd.DatetimeIndex(pd.to_datetime(hydro.target_timestamp_utc, utc=True))
    assert len(renewable) == len(hydro) == 8760
    assert renewable_time.equals(hydro_time)
    assert "hydro_water_reservoir_actual_mw" in renewable
    assert renewable.hydro_water_reservoir_actual_mw.notna().any()
    assert {
        "synthetic_2025_hydro_inflow_mwh",
        "hydro_turbine_generation_limit_mw",
        "maximum_reservoir_energy_mwh",
        "provenance",
        "scenario_random_seed",
    }.issubset(hydro.columns)
    assert hydro.synthetic_2025_hydro_inflow_mwh.ge(0).all()


def test_integration_config_selects_historical_resources_for_dashboard_by_default():
    settings = load_integrated_settings("configs/integration.yaml")
    assert settings["renewable_data_mode"] == streamlit_app.HISTORICAL_RESOURCE_MODE
    provider, params, inflow = streamlit_app._load_resource_inputs(
        settings["renewable_data_mode"], settings
    )
    assert provider.target_timestamps[0] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert params.zone == "DE00"
    assert len(inflow) == 8760


def test_minimum_reservoir_storage_is_an_optimizer_constraint():
    assets, _ = load_dispatch_config("configs/assets.yaml")
    hydro = replace(
        assets.hydro,
        capacity_mw=0.0,
        initial_energy_budget_mwh=5.0,
        reservoir_capacity_mwh=10.0,
        minimum_storage_mwh=5.0,
    )
    assets = replace(assets, hydro=hydro)
    result = optimize_dispatch(
        [10.0], {"solar": [0.0], "wind": [0.0]}, assets,
        DispatchState(assets.battery.initial_soc_mwh, 5.0, 0.0),
        timestamps=pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"),
        terminal_soc_policy="none",
    )
    assert result.schedule.hydro_remaining_mwh.iloc[0] >= 5.0 - 1e-8
    with pytest.raises(ValueError, match="below minimum"):
        optimize_dispatch(
            [10.0], {"solar": [0.0], "wind": [0.0]}, assets,
            DispatchState(assets.battery.initial_soc_mwh, 4.0, 0.0),
            timestamps=pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"),
            terminal_soc_policy="none",
        )


def test_changed_solar_forecast_changes_dispatch_and_changed_inflow_changes_reservoir():
    assets, _ = load_dispatch_config("configs/assets.yaml")
    time = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
    state = DispatchState(assets.battery.initial_soc_mwh, 0.0, 0.0)
    no_solar = optimize_dispatch(
        [10.0], {"solar": [0.0], "wind": [0.0]}, assets, state,
        timestamps=time, terminal_soc_policy="none",
    )
    with_solar = optimize_dispatch(
        [10.0], {"solar": [5.0], "wind": [0.0]}, assets, state,
        timestamps=time, terminal_soc_policy="none",
    )
    assert with_solar.schedule.solar_to_demand_mw.iloc[0] > no_solar.schedule.solar_to_demand_mw.iloc[0]
    assert with_solar.schedule.conventional_mw.iloc[0] < no_solar.schedule.conventional_mw.iloc[0]

    expensive_hydro = replace(assets.hydro, marginal_cost_eur_mwh=200.0)
    assets = replace(assets, hydro=expensive_hydro)
    no_inflow = optimize_dispatch(
        [10.0], {"solar": [0.0], "wind": [0.0]}, assets,
        DispatchState(assets.battery.initial_soc_mwh, 0.0, 0.0),
        hydro_inflow_mw=[0.0], timestamps=time, terminal_soc_policy="none",
    )
    with_inflow = optimize_dispatch(
        [10.0], {"solar": [0.0], "wind": [0.0]}, assets,
        DispatchState(assets.battery.initial_soc_mwh, 0.0, 0.0),
        hydro_inflow_mw=[5.0], timestamps=time, terminal_soc_policy="none",
    )
    assert with_inflow.schedule.hydro_remaining_mwh.iloc[0] > no_inflow.schedule.hydro_remaining_mwh.iloc[0]
