"""Canonical hourly frame validation shared by later GridDRIFT phases."""

from __future__ import annotations

import pandas as pd

REQUIRED_SCENARIO_COLUMNS = (
    "demand_mw",
    "solar_actual_mw",
    "solar_forecast_mw",
    "wind_actual_mw",
    "wind_forecast_mw",
)


def validate_hourly_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("scenario index must be a DatetimeIndex")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("scenario index must be timezone-aware UTC")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("scenario index must be strictly increasing with no duplicates")
    if len(index) > 1 and not (index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("scenario index must have exactly one-hour spacing")


def validate_scenario_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a float-valued canonical hourly scenario frame."""
    validate_hourly_index(frame.index)
    missing = set(REQUIRED_SCENARIO_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"scenario is missing required columns: {sorted(missing)}")
    result = frame.loc[:, REQUIRED_SCENARIO_COLUMNS].astype(float).copy()
    if result.isna().any().any():
        raise ValueError("scenario values must not contain missing values")
    if (result < 0).any().any():
        bad = result.columns[(result < 0).any()].tolist()
        raise ValueError(f"scenario power values must be non-negative; invalid columns: {bad}")
    return result
