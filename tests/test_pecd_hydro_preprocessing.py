import pandas as pd
import pytest

from griddrift.data.hydro import (
    HydroDataError,
    build_assumed_hourly_inflow_scenario,
    build_seasonal_reservoir_inflow_profile,
    clamp_initial_reservoir_energy,
    load_de00_reservoir_parameters,
    reservoir_balance_step,
)


def _capacity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["DE00", "Reservoir", "Total turbining capacity (MW)", 1297.0],
            ["DE00", "Reservoir", "Reservoir capacity (GWh)", 258.0],
            ["DE00", "Run-of-River and pondage", "Total turbining capacity (MW)", 4036.0],
            ["DE00", "Pump Storage - Open Loop", "Reservoir capacity (GWh)", 417.0],
        ],
        columns=["zone", "type", "variable", "value"],
    )


def _inflows() -> pd.DataFrame:
    column = "Cumulated inflow into reservoirs per week in GWh"
    return pd.DataFrame(
        [
            ["DE00", 1, 2022, 168.0],
            ["DE00", 1, 2023, 336.0],
            ["DE00", 53, 2020, 840.0],
            ["DE00", 53, 2015, 1680.0],
        ],
        columns=["zone", "week", "year", column],
    )


def test_extracts_only_de00_reservoir_and_clamps_initial_fraction():
    params = load_de00_reservoir_parameters(_capacity(), initial_filling_fraction=1.2)

    assert params.turbine_capacity_mw == 1297.0
    assert params.reservoir_capacity_gwh == 258.0
    assert params.reservoir_capacity_mwh == 258_000.0
    assert params.initial_filling_fraction == 1.0
    assert params.initial_reservoir_energy_mwh == 258_000.0


def test_minimum_reservoir_energy_is_configurable_and_cannot_exceed_initial_state():
    params = load_de00_reservoir_parameters(
        _capacity(), initial_filling_fraction=0.5, minimum_reservoir_energy_mwh=25_000.0
    )
    assert params.minimum_reservoir_energy_mwh == 25_000.0
    with pytest.raises(HydroDataError, match="minimum reservoir energy"):
        load_de00_reservoir_parameters(
            _capacity(), initial_filling_fraction=0.5, minimum_reservoir_energy_mwh=130_000.0
        )


def test_weekly_median_and_gwh_to_hourly_mwh_conversion_include_week_53():
    profile = build_seasonal_reservoir_inflow_profile(_inflows())
    week1 = profile.set_index("iso_week").loc[1]
    week53 = profile.set_index("iso_week").loc[53]

    assert week1.median_weekly_inflow_gwh == 252.0
    assert week1.assumed_hourly_inflow_mwh == 1500.0
    assert week53.median_weekly_inflow_gwh == 1260.0
    assert week53.assumed_hourly_inflow_mwh == 7500.0
    assert week53.historical_year_count == 2


def test_hourly_scenario_uses_iso_week_and_handles_leap_year_and_week_53():
    profile = pd.DataFrame(
        {
            "iso_week": range(1, 54),
            "median_weekly_inflow_gwh": [1.0] * 53,
            "assumed_hourly_inflow_mwh": [1000.0 / 168.0] * 52 + [7.5],
        }
    )

    leap = build_assumed_hourly_inflow_scenario(profile, year=2020)
    nonleap = build_assumed_hourly_inflow_scenario(profile, year=2025)
    week53 = leap.loc[leap.iso_week == 53]

    assert len(leap) == 8784
    assert len(nonleap) == 8760
    assert not week53.empty
    assert week53.assumed_inflow_mwh_per_hour.eq(7.5).all()
    assert leap.synthetic_2025_hydro_inflow_mwh.equals(leap.assumed_inflow_mwh_per_hour)
    assert leap.inflow_scenario.str.contains("not observed").all()
    assert nonleap.target_timestamp_utc.dt.tz is not None
    assert nonleap.target_timestamp_utc.is_monotonic_increasing


def test_reservoir_clamp_and_balance_conserve_energy_with_spill():
    assert clamp_initial_reservoir_energy(-10.0, 100.0) == 0.0
    assert clamp_initial_reservoir_energy(110.0, 100.0) == 100.0

    next_energy, spill = reservoir_balance_step(90.0, 20.0, 5.0, 100.0)
    assert next_energy == 100.0
    assert spill == 5.0
    assert next_energy == pytest.approx(90.0 + 20.0 - 5.0 - spill)

    with pytest.raises(HydroDataError, match="exceeds"):
        reservoir_balance_step(10.0, 2.0, 13.0, 100.0)
