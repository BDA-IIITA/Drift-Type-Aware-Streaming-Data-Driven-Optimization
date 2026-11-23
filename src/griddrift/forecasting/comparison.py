"""Timestamp-safe alignment helpers for offline forecasting-model comparisons."""

from __future__ import annotations

import pandas as pd

FORECAST_KEYS = ["issue_timestamp", "target_timestamp", "horizon_hours"]


def normalize_forecasts(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return a compact, validated frame with common prediction column names."""
    required = {*FORECAST_KEYS, "predicted_demand_mw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{model_name} forecast is missing columns: {sorted(missing)}")
    result = frame.loc[:, FORECAST_KEYS + ["predicted_demand_mw"]].copy()
    for column in ("issue_timestamp", "target_timestamp"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    result["horizon_hours"] = pd.to_numeric(result["horizon_hours"], errors="raise").astype(int)
    result["predicted_demand_mw"] = pd.to_numeric(
        result["predicted_demand_mw"], errors="raise"
    ).astype(float)
    if result[FORECAST_KEYS].isna().any().any() or result.predicted_demand_mw.isna().any():
        raise ValueError(f"{model_name} forecast has missing timestamps or predictions")
    if not result.horizon_hours.between(1, 24).all():
        raise ValueError(f"{model_name} horizon must be between 1 and 24 hours")
    lead = (result.target_timestamp - result.issue_timestamp).dt.total_seconds() / 3600
    if not (lead == result.horizon_hours).all():
        raise ValueError(f"{model_name} issue/target timestamps do not match forecast horizon")
    if result.duplicated(FORECAST_KEYS).any():
        raise ValueError(f"{model_name} has duplicate issue/target/horizon forecast keys")
    result = result.rename(columns={"predicted_demand_mw": f"{model_name}_predicted_mw"})
    return result


def align_forecasts(
    forecast_frames: dict[str, pd.DataFrame], actual_demand: pd.Series
) -> pd.DataFrame:
    """Inner-join models on issue/target/lead and attach source actuals once."""
    if len(forecast_frames) < 2:
        raise ValueError("at least two forecast models are required for comparison")
    actual_index = pd.DatetimeIndex(actual_demand.index)
    if actual_index.tz is None:
        raise ValueError("actual-demand index must be timezone-aware")
    actual_index = actual_index.tz_convert("UTC")
    actual = pd.Series(
        pd.to_numeric(actual_demand, errors="raise").to_numpy(dtype=float),
        index=actual_index,
        name="actual_demand_mw",
    )
    if actual.index.has_duplicates:
        raise ValueError("actual-demand index must not contain duplicates")

    names = list(forecast_frames)
    aligned = normalize_forecasts(forecast_frames[names[0]], names[0])
    for name in names[1:]:
        aligned = aligned.merge(
            normalize_forecasts(forecast_frames[name], name),
            on=FORECAST_KEYS,
            how="inner",
            validate="one_to_one",
        )
    if aligned.empty:
        raise ValueError("forecast models have no common issue/target/horizon keys")
    observed = actual.reindex(pd.DatetimeIndex(aligned.target_timestamp))
    if observed.isna().any():
        raise ValueError("one or more common forecast targets have no actual observation")
    aligned["actual_demand_mw"] = observed.to_numpy()
    return aligned.sort_values(FORECAST_KEYS, kind="stable").reset_index(drop=True)
