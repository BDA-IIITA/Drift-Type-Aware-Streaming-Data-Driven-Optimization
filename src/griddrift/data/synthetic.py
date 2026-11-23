"""Deterministic simulated solar/wind and demand data for smoke tests.

This module intentionally creates a scenario, not measurements. Its forecast
columns are noisy issue-time proxies for later forecasting/dispatch integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from griddrift.data.schema import validate_scenario_frame


@dataclass(frozen=True)
class SyntheticScenarioMetadata:
    is_simulated: bool = True
    renewable_source: str = "deterministic seasonal profiles with seeded forecast noise"
    demand_source: str = "deterministic load shape with seeded noise"


def make_synthetic_scenario(
    *,
    start: str | pd.Timestamp = "2023-01-01T00:00:00Z",
    periods: int = 24 * 14,
    seed: int = 42,
    solar_capacity_mw: float = 12.0,
    wind_capacity_mw: float = 16.0,
) -> tuple[pd.DataFrame, SyntheticScenarioMetadata]:
    """Create a small hourly scenario with separate actual and forecast renewables."""
    if periods < 24:
        raise ValueError("periods must be at least 24 hours")
    if solar_capacity_mw < 0 or wind_capacity_mw < 0:
        raise ValueError("renewable capacities must be non-negative")

    index = pd.date_range(pd.Timestamp(start), periods=periods, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    hour = index.hour.to_numpy()
    day = index.dayofyear.to_numpy()
    weekday = index.dayofweek.to_numpy()

    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0.0, None)
    seasonal_solar = 0.65 + 0.35 * np.sin(2 * np.pi * (day - 80) / 365)
    solar_actual = np.clip(solar_capacity_mw * daylight * seasonal_solar + rng.normal(0, 0.35, periods), 0, solar_capacity_mw)

    wind_base = 0.48 + 0.14 * np.sin(2 * np.pi * (hour + 5) / 24) + 0.12 * np.sin(2 * np.pi * day / 365)
    wind_actual = np.clip(wind_capacity_mw * (wind_base + rng.normal(0, 0.10, periods)), 0, wind_capacity_mw)

    evening_peak = 9.0 * np.exp(-((hour - 19) / 3.5) ** 2)
    morning_peak = 4.5 * np.exp(-((hour - 8) / 3.0) ** 2)
    weekend_adjustment = np.where(weekday >= 5, -3.0, 0.0)
    annual_demand = 3.0 * np.cos(2 * np.pi * (day - 15) / 365)
    demand = np.clip(43.0 + evening_peak + morning_peak + weekend_adjustment + annual_demand + rng.normal(0, 0.8, periods), 1.0, None)

    frame = pd.DataFrame(
        {
            "demand_mw": demand,
            "solar_actual_mw": solar_actual,
            "solar_forecast_mw": np.clip(solar_actual + rng.normal(0, 0.75, periods), 0, solar_capacity_mw),
            "wind_actual_mw": wind_actual,
            "wind_forecast_mw": np.clip(wind_actual + rng.normal(0, 1.2, periods), 0, wind_capacity_mw),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return validate_scenario_frame(frame), SyntheticScenarioMetadata()
