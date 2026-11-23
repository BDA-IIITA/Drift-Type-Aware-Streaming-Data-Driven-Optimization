"""Data contracts, ENTSO-E preprocessing, and simulated resource scenarios."""

from griddrift.data.entsoe import EntsoeLoadError, fetch_entsoe_actual_load
from griddrift.data.hydro import (
    HydroDataError,
    ReservoirParameters,
    build_assumed_hourly_inflow_scenario,
    build_seasonal_reservoir_inflow_profile,
    clamp_initial_reservoir_energy,
    load_assumed_hourly_inflow_series,
    load_de00_reservoir_parameters,
    load_reservoir_parameters_json,
    reservoir_balance_step,
    write_de00_hydro_scenario,
)
from griddrift.data.preprocessing import normalize_demand, normalize_hourly_demand
from griddrift.data.renewables import (
    RenewableDataError,
    RenewableResourceProvider,
    preprocess_entsoe_renewables,
    preprocess_entsoe_renewables_csv,
)
from griddrift.data.synthetic import SyntheticScenarioMetadata, make_synthetic_scenario

__all__ = [
    "EntsoeLoadError",
    "HydroDataError",
    "RenewableDataError",
    "RenewableResourceProvider",
    "ReservoirParameters",
    "SyntheticScenarioMetadata",
    "build_assumed_hourly_inflow_scenario",
    "build_seasonal_reservoir_inflow_profile",
    "clamp_initial_reservoir_energy",
    "fetch_entsoe_actual_load",
    "load_assumed_hourly_inflow_series",
    "load_de00_reservoir_parameters",
    "load_reservoir_parameters_json",
    "make_synthetic_scenario",
    "normalize_demand",
    "normalize_hourly_demand",
    "preprocess_entsoe_renewables",
    "preprocess_entsoe_renewables_csv",
    "reservoir_balance_step",
    "write_de00_hydro_scenario",
]
