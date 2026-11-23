"""Physically constrained one-interval-at-a-time greedy baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from griddrift.config.schema import AssetConfig
from griddrift.dispatch.engine import DispatchState, optimize_dispatch


def terminal_soc_settlement(
    final_soc_mwh: float,
    target_soc_mwh: float,
    asset_config: AssetConfig,
    replacement_price_eur_mwh: float,
) -> float:
    """Value terminal stored-energy mismatch at a consistent replacement rate.

    A deficit is valued at the input energy required to restore it, including
    charge losses. Excess stored energy receives the corresponding delivered-
    energy value, including discharge losses.
    """
    delta = target_soc_mwh - final_soc_mwh
    if delta >= 0:
        return delta / asset_config.battery.charge_efficiency * replacement_price_eur_mwh
    return delta * asset_config.battery.discharge_efficiency * replacement_price_eur_mwh


def run_greedy_dispatch(
    demand_forecast: Sequence[float],
    renewable_forecasts: Mapping[str, Sequence[float]],
    asset_config: AssetConfig,
    initial_state: DispatchState,
    *,
    terminal_soc_target_mwh: float | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, float, float]:
    """Solve each hour myopically, carrying SOC, hydro, and generator state.

    Returns (hourly schedule, raw operating cost, terminal SOC settlement).
    The settlement values a terminal SOC deficit at one MWh's conventional
    replacement cost, so the greedy baseline does not receive free battery
    energy that the 24-hour optimizer is required to restore.
    """
    n = len(demand_forecast)
    state = initial_state
    rows = []
    cost = 0.0
    hydro_inflow = kwargs.get("hydro_inflow_mw")
    hydro_generation_limits = kwargs.get("hydro_generation_limits_mw")
    conventional_prices = kwargs.get("conventional_costs_eur_mwh")
    emergency_prices = kwargs.get("emergency_costs_eur_mwh")
    solar_prices = kwargs.get("solar_costs_eur_mwh")
    wind_prices = kwargs.get("wind_costs_eur_mwh")
    hydro_inflow = [0.0] * n if hydro_inflow is None else hydro_inflow
    hydro_generation_limits = (
        [asset_config.hydro.capacity_mw] * n
        if hydro_generation_limits is None
        else hydro_generation_limits
    )
    conventional_prices = (
        [asset_config.conventional.marginal_cost_eur_mwh] * n
        if conventional_prices is None
        else conventional_prices
    )
    emergency_prices = (
        [asset_config.emergency_grid.marginal_cost_eur_mwh] * n
        if emergency_prices is None
        else emergency_prices
    )
    for t in range(n):
        one = optimize_dispatch(
            [demand_forecast[t]],
            {key: [values[t]] for key, values in renewable_forecasts.items()},
            asset_config,
            state,
            hydro_inflow_mw=[hydro_inflow[t]],
            hydro_generation_limits_mw=[hydro_generation_limits[t]],
            solar_costs_eur_mwh=[solar_prices[t]] if solar_prices is not None else None,
            wind_costs_eur_mwh=[wind_prices[t]] if wind_prices is not None else None,
            conventional_costs_eur_mwh=[conventional_prices[t]],
            emergency_costs_eur_mwh=[emergency_prices[t]],
            conventional_min_output_mw=kwargs.get("conventional_min_output_mw", 0.0),
            battery_charging_policy=kwargs.get("battery_charging_policy", "renewable_only"),
            allow_hydro_battery_charging=kwargs.get("allow_hydro_battery_charging", False),
            terminal_soc_policy="none",
            timestamps=[kwargs["timestamps"][t]] if kwargs.get("timestamps") is not None else None,
            solver_name=kwargs.get("solver_name", "appsi_highs"),
        )
        row = one.schedule.iloc[0].to_dict()
        rows.append(row)
        cost += one.total_cost_eur
        state = one.first_hour_state
    schedule = pd.DataFrame(rows)
    target = (
        initial_state.battery_soc_mwh
        if terminal_soc_target_mwh is None
        else terminal_soc_target_mwh
    )
    settlement_price = sum(conventional_prices) / len(conventional_prices)
    settlement = terminal_soc_settlement(state.battery_soc_mwh, target, asset_config, settlement_price)
    schedule.index = (
        pd.Index(kwargs["timestamps"], name="timestamp")
        if kwargs.get("timestamps") is not None
        else pd.RangeIndex(n, name="interval")
    )
    return schedule, float(cost), float(settlement)
