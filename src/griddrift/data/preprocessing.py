"""ENTSO-E demand preprocessing with explicit hourly UTC semantics."""

from __future__ import annotations

import pandas as pd


def normalize_hourly_demand(
    raw: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    demand_column: str = "demand_mw",
) -> pd.DataFrame:
    """Return contiguous hourly UTC demand in MW."""
    return normalize_demand(
        raw,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
        resolution="1h",
    )


def normalize_demand(
    raw: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    demand_column: str = "demand_mw",
    resolution: str = "1h",
) -> pd.DataFrame:
    """Return contiguous UTC demand in MW at the selected target resolution.

    Input values represent average power over each source interval. Therefore,
    resampling uses a mean, not a sum. Missing target intervals are rejected rather
    than invented because later forecast and dispatch stages require an honest
    observation history.
    """
    try:
        interval = pd.Timedelta(resolution)
    except ValueError as exc:
        raise ValueError("resolution must be a valid pandas duration") from exc
    if interval <= pd.Timedelta(0):
        raise ValueError("resolution must be positive")
    if timestamp_column not in raw or demand_column not in raw:
        raise KeyError(f"input must contain {timestamp_column!r} and {demand_column!r}")

    timestamp = pd.to_datetime(raw[timestamp_column], utc=True, errors="coerce")
    demand = pd.to_numeric(raw[demand_column], errors="coerce")
    if timestamp.isna().any() or demand.isna().any():
        raise ValueError("timestamps and demand values must be parseable and non-missing")
    if (demand < 0).any():
        raise ValueError("demand_mw must be non-negative")

    frame = pd.DataFrame({"demand_mw": demand.to_numpy()}, index=pd.DatetimeIndex(timestamp))
    normalized = frame.groupby(level=0).mean().sort_index().resample(resolution).mean()
    if normalized["demand_mw"].isna().any():
        missing = normalized.index[normalized["demand_mw"].isna()]
        raise ValueError(
            f"input has {len(missing)} missing {resolution} interval(s); first={missing[0]}"
        )
    normalized.index.name = "timestamp"
    return normalized
