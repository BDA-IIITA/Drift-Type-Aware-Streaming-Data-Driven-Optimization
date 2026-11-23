"""Utilities for paired, post-update XGBoost snapshot evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from griddrift.forecasting.features import DirectFeatureSpec, _validate_demand, build_feature_row


def build_feature_matrix_for_issues(
    demand: pd.Series,
    issue_times: pd.DatetimeIndex,
    spec: DirectFeatureSpec,
) -> pd.DataFrame:
    """Vectorized equivalent of the existing feature builder, using observations <= issue."""
    demand = _validate_demand(demand)
    issues = pd.DatetimeIndex(issue_times)
    if issues.tz is None:
        issues = issues.tz_localize("UTC")
    else:
        issues = issues.tz_convert("UTC")
    if not issues.isin(demand.index).all():
        raise ValueError("every issue timestamp must exist in the demand index")

    columns: dict[str, np.ndarray] = {}
    for lag in spec.lags_hours:
        # GridDRIFT's lag_1h is the latest observation at the issue time.
        columns[f"lag_{lag}h"] = demand.shift(lag - 1).to_numpy()
    for window in spec.rolling_windows_hours:
        rolling = demand.rolling(window=window, min_periods=window)
        columns[f"roll_mean_{window}h"] = rolling.mean().to_numpy()
        columns[f"roll_std_{window}h"] = rolling.std(ddof=0).to_numpy()

    for step in range(1, spec.horizon_hours + 1):
        target_times = demand.index + pd.Timedelta(hours=step)
        prefix = f"target_{step:02d}h"
        columns[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * target_times.hour / 24)
        columns[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * target_times.hour / 24)
        columns[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * target_times.dayofweek / 7)
        columns[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * target_times.dayofweek / 7)
        columns[f"{prefix}_month_sin"] = np.sin(2 * np.pi * target_times.month / 12)
        columns[f"{prefix}_month_cos"] = np.cos(2 * np.pi * target_times.month / 12)

    features = pd.DataFrame(columns, index=demand.index)
    selected = features.reindex(issues)
    if selected.isna().any().any():
        raise ValueError("issue timestamps do not have the required observed lookback history")

    # Guard against accidental feature drift if the canonical builder changes later.
    for issue in (issues[0], issues[-1]) if len(issues) else ():
        expected = build_feature_row(demand, issue, spec)
        actual = selected.loc[issue]
        if not np.allclose(
            actual.loc[list(expected)].to_numpy(dtype=float),
            np.fromiter(expected.values(), dtype=float),
            rtol=1e-10,
            atol=1e-8,
        ):
            raise AssertionError(f"vectorized features differ from canonical builder at {issue}")
    return selected


def block_bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    block_size: int = 24,
    replicates: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% moving-block bootstrap CI for updated-minus-old absolute-error differences."""
    values = np.asarray(deltas, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < block_size or replicates < 1:
        return float("nan"), float("nan")
    block_size = min(block_size, len(values))
    starts = np.arange(len(values) - block_size + 1)
    blocks_per_sample = int(np.ceil(len(values) / block_size))
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for sample in range(replicates):
        chosen = rng.choice(starts, size=blocks_per_sample, replace=True)
        indices = np.concatenate([np.arange(start, start + block_size) for start in chosen])[: len(values)]
        means[sample] = values[indices].mean()
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def compare_absolute_errors(
    actual: np.ndarray,
    previous_prediction: np.ndarray,
    updated_prediction: np.ndarray,
    *,
    block_size: int = 24,
    bootstrap_replicates: int = 2_000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Paired metrics; a 24-hour block-bootstrap CI classifies fixed-lead change."""
    actual = np.asarray(actual, dtype=float)
    previous_prediction = np.asarray(previous_prediction, dtype=float)
    updated_prediction = np.asarray(updated_prediction, dtype=float)
    if not (actual.shape == previous_prediction.shape == updated_prediction.shape):
        raise ValueError("paired actual and prediction arrays must have identical shapes")
    if not np.isfinite(actual).all() or not np.isfinite(previous_prediction).all() or not np.isfinite(updated_prediction).all():
        raise ValueError("paired values must be finite")
    previous_errors = np.abs(actual - previous_prediction)
    updated_errors = np.abs(actual - updated_prediction)
    delta = updated_errors - previous_errors
    lower, upper = block_bootstrap_mean_ci(
        delta, block_size=block_size, replicates=bootstrap_replicates, seed=seed
    )
    if not np.isfinite(lower):
        status = "inconclusive"
    elif upper < 0:
        status = "improved"
    elif lower > 0:
        status = "worsened"
    else:
        status = "inconclusive"
    old_mae = float(previous_errors.mean()) if len(previous_errors) else float("nan")
    new_mae = float(updated_errors.mean()) if len(updated_errors) else float("nan")
    return {
        "sample_count": len(actual),
        "previous_mae_mw": old_mae,
        "updated_mae_mw": new_mae,
        "previous_rmse_mw": float(np.sqrt(np.mean((actual - previous_prediction) ** 2))) if len(actual) else float("nan"),
        "updated_rmse_mw": float(np.sqrt(np.mean((actual - updated_prediction) ** 2))) if len(actual) else float("nan"),
        "mae_change_mw": new_mae - old_mae,
        "mae_change_percent": (new_mae - old_mae) / old_mae * 100 if old_mae else float("nan"),
        "paired_delta_ci_low_mw": lower,
        "paired_delta_ci_high_mw": upper,
        "effect": status,
    }


def season_name(timestamp: pd.Timestamp) -> str:
    """Meteorological seasons in UTC: DJF, MAM, JJA, SON."""
    month = pd.Timestamp(timestamp).month
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}[month]
