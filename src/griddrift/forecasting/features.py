"""Past-only features for direct 24-hour demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LAGS_HOURS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
ROLLING_WINDOWS_HOURS = (24, 48, 168)


@dataclass(frozen=True)
class DirectFeatureSpec:
    lookback_hours: int = 168
    horizon_hours: int = 24
    lags_hours: tuple[int, ...] = LAGS_HOURS
    rolling_windows_hours: tuple[int, ...] = ROLLING_WINDOWS_HOURS

    def __post_init__(self) -> None:
        if self.lookback_hours < max((*self.lags_hours, *self.rolling_windows_hours)):
            raise ValueError("lookback_hours must cover every configured lag and rolling window")
        if self.horizon_hours < 1:
            raise ValueError("horizon_hours must be >= 1")


def _validate_demand(demand: pd.Series) -> pd.Series:
    values = pd.to_numeric(demand, errors="coerce").astype(float)
    index = pd.DatetimeIndex(demand.index)
    if index.tz is None:
        raise ValueError("demand index must be timezone-aware UTC")
    index = index.tz_convert("UTC")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("demand index must be strictly increasing without duplicates")
    if values.isna().any() or (values < 0).any():
        raise ValueError("demand values must be non-negative and non-missing")
    if len(index) > 1 and not (index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("direct 24-hour MLP demand must have one-hour spacing")
    return pd.Series(values.to_numpy(), index=index, name="demand_mw")


def build_feature_row(history: pd.Series, issue_time: pd.Timestamp, spec: DirectFeatureSpec) -> dict[str, float]:
    """Build one feature row using observations available at ``issue_time`` only."""
    history = _validate_demand(history)
    return _build_feature_row_validated(history, issue_time, spec)


def _build_feature_row_validated(
    history: pd.Series, issue_time: pd.Timestamp, spec: DirectFeatureSpec
) -> dict[str, float]:
    """Internal fast path for callers that have already validated the full series."""
    issue = pd.Timestamp(issue_time)
    issue = issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
    past = history.loc[history.index <= issue]
    if len(past) < spec.lookback_hours:
        raise ValueError("insufficient observed history before issue_time")
    past = past.iloc[-spec.lookback_hours:]
    features: dict[str, float] = {}
    for lag in spec.lags_hours:
        features[f"lag_{lag}h"] = float(past.iloc[-lag])
    for window in spec.rolling_windows_hours:
        recent = past.iloc[-window:]
        features[f"roll_mean_{window}h"] = float(recent.mean())
        features[f"roll_std_{window}h"] = float(recent.std(ddof=0))
    for step in range(1, spec.horizon_hours + 1):
        target = issue + pd.Timedelta(hours=step)
        prefix = f"target_{step:02d}h"
        features[f"{prefix}_hour_sin"] = float(np.sin(2 * np.pi * target.hour / 24))
        features[f"{prefix}_hour_cos"] = float(np.cos(2 * np.pi * target.hour / 24))
        features[f"{prefix}_dow_sin"] = float(np.sin(2 * np.pi * target.dayofweek / 7))
        features[f"{prefix}_dow_cos"] = float(np.cos(2 * np.pi * target.dayofweek / 7))
        features[f"{prefix}_month_sin"] = float(np.sin(2 * np.pi * target.month / 12))
        features[f"{prefix}_month_cos"] = float(np.cos(2 * np.pi * target.month / 12))
    return features


def build_supervised_dataset(
    demand: pd.Series, spec: DirectFeatureSpec = DirectFeatureSpec()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create direct-horizon samples; each feature row is past-only at its issue time."""
    demand = _validate_demand(demand)
    first_issue = spec.lookback_hours - 1
    last_issue_exclusive = len(demand) - spec.horizon_hours
    if last_issue_exclusive <= first_issue:
        raise ValueError("not enough demand history for lookback plus forecast horizon")
    features: list[dict[str, float]] = []
    labels: list[np.ndarray] = []
    issues: list[pd.Timestamp] = []
    for position in range(first_issue, last_issue_exclusive):
        issue = demand.index[position]
        features.append(_build_feature_row_validated(demand.iloc[: position + 1], issue, spec))
        labels.append(
            demand.iloc[position + 1 : position + 1 + spec.horizon_hours].to_numpy(dtype=float)
        )
        issues.append(issue)
    X = pd.DataFrame(features, index=pd.DatetimeIndex(issues, name="issue_timestamp"))
    y = pd.DataFrame(
        labels,
        index=X.index,
        columns=[f"target_{step:02d}h" for step in range(1, spec.horizon_hours + 1)],
    )
    return X, y
