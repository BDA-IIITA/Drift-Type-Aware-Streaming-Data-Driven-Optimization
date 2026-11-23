"""Experiment-only forecast input validation for GridDRIFT Phase 4.5.

The adapter intentionally contains no actual-demand values. Actual observations
belong to a separate reveal/execution path so a day-ahead plan can only receive
the two saved prediction vectors and their model-version metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

KEYS = ["issue_timestamp", "target_timestamp", "horizon_hours"]
HORIZON = 24


@dataclass(frozen=True)
class ForecastVector:
    """One matched 24-hour forecast pair, with no actual-demand field."""

    issue_timestamp: pd.Timestamp
    target_timestamps: tuple[pd.Timestamp, ...]
    mlp_predicted_demand_mw: tuple[float, ...]
    xgboost_predicted_demand_mw: tuple[float, ...]
    mlp_model_version: str
    xgboost_model_version: str

    def __post_init__(self) -> None:
        expected_targets = tuple(
            self.issue_timestamp + pd.Timedelta(hours=step)
            for step in range(1, HORIZON + 1)
        )
        if len(self.target_timestamps) != HORIZON or self.target_timestamps != expected_targets:
            raise ValueError("forecast vector must contain exact targets issue+1 through issue+24")
        for name, values in (
            ("MLP", self.mlp_predicted_demand_mw),
            ("XGBoost", self.xgboost_predicted_demand_mw),
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != (HORIZON,) or not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"{name} forecast must be 24 finite, non-negative MW values")
        if not self.mlp_model_version or not self.xgboost_model_version:
            raise ValueError("both forecast model versions must be present")


@dataclass(frozen=True)
class ForecastVectorSet:
    """Validated full vectors plus counts for intentionally excluded boundaries."""

    vectors: tuple[ForecastVector, ...]
    aligned_prediction_keys: int
    partial_issue_count: int


@dataclass(frozen=True)
class Phase45InputSettings:
    """Shared input/scenario settings copied from the integrated configuration."""

    horizon_hours: int = HORIZON
    demand_scale: float = 0.001
    resource_seed: int = 42
    resource_scenario: str = "normal"
    conventional_min_output_mw: float = 0.0
    battery_charging_policy: str = "renewable_only"
    terminal_soc_policy: str = "preserve_initial"
    terminal_soc_target_mwh: float | None = None
    solver_name: str = "appsi_highs"

    def __post_init__(self) -> None:
        if self.horizon_hours != HORIZON:
            raise ValueError("Phase 4.5 requires the existing 24-hour forecast/dispatch horizon")
        if self.demand_scale <= 0:
            raise ValueError("demand_scale must be positive")
        if self.conventional_min_output_mw < 0:
            raise ValueError("conventional minimum output must be non-negative")
        if self.resource_scenario not in {"normal", "low_renewables", "wind_scarcity"}:
            raise ValueError("unsupported resource scenario")
        if self.battery_charging_policy != "renewable_only":
            raise ValueError("Phase 4.5 must preserve the renewable-only battery charging policy")
        if self.terminal_soc_policy not in {"preserve_initial", "minimum", "target", "none"}:
            raise ValueError("unsupported terminal SOC policy")
        if self.terminal_soc_policy == "target" and self.terminal_soc_target_mwh is None:
            raise ValueError("terminal_soc_target_mwh is required for target policy")


def _normalized_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required key columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("issue_timestamp", "target_timestamp"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    numeric_horizon = pd.to_numeric(result["horizon_hours"], errors="raise")
    if not np.equal(numeric_horizon, np.floor(numeric_horizon)).all():
        raise ValueError(f"{label} has non-integer horizon values")
    result["horizon_hours"] = numeric_horizon.astype(int)
    if result.duplicated(KEYS).any():
        raise ValueError(f"{label} contains duplicate issue/target/horizon keys")
    expected = result["issue_timestamp"] + pd.to_timedelta(result["horizon_hours"], unit="h")
    if not expected.equals(result["target_timestamp"]):
        raise ValueError(f"{label} violates target_timestamp = issue_timestamp + horizon_hours")
    return result


def _registry_columns(registry: pd.DataFrame, label: str) -> pd.DataFrame:
    if "predicted_demand_mw" not in registry or "model_version" not in registry:
        raise ValueError(f"{label} registry must contain predicted_demand_mw and model_version")
    result = _normalized_keys(registry, label)
    result["predicted_demand_mw"] = pd.to_numeric(result["predicted_demand_mw"], errors="raise")
    if not np.isfinite(result["predicted_demand_mw"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} registry has non-finite predictions")
    if (result["predicted_demand_mw"] < 0).any():
        raise ValueError(f"{label} registry has negative demand predictions")
    if result["model_version"].isna().any() or (result["model_version"].astype(str).str.len() == 0).any():
        raise ValueError(f"{label} registry has missing model versions")
    return result[KEYS + ["predicted_demand_mw", "model_version"]]


def load_phase45_forecast_vectors(
    aligned_predictions: pd.DataFrame,
    adaptive_mlp_registry: pd.DataFrame,
    adaptive_xgboost_registry: pd.DataFrame,
    *,
    prediction_tolerance_mw: float = 1e-6,
) -> ForecastVectorSet:
    """Join exact saved forecast keys/versions and return only complete vectors.

    `actual_demand_mw` in the common comparison file is deliberately ignored.
    It is not copied into a returned vector and cannot be passed to a planning
    call through this interface.
    """

    if prediction_tolerance_mw < 0:
        raise ValueError("prediction_tolerance_mw must be non-negative")
    aligned = _normalized_keys(aligned_predictions, "Phase 3.8 aligned predictions")
    for column in ("adaptive_mlp_predicted_mw", "adaptive_xgboost_predicted_mw"):
        if column not in aligned:
            raise ValueError(f"Phase 3.8 aligned predictions are missing {column}")
        aligned[column] = pd.to_numeric(aligned[column], errors="raise")
        if not np.isfinite(aligned[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Phase 3.8 aligned predictions contain non-finite {column}")

    mlp = _registry_columns(adaptive_mlp_registry, "adaptive MLP")
    xgb = _registry_columns(adaptive_xgboost_registry, "adaptive XGBoost")

    # Left joins ensure every comparison key has its original forecast record.
    joined = aligned[KEYS + ["adaptive_mlp_predicted_mw", "adaptive_xgboost_predicted_mw"]]
    joined = joined.merge(mlp, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_mlp"))
    if joined["predicted_demand_mw"].isna().any():
        raise ValueError("adaptive MLP registry is missing Phase 3.8 common forecast keys")
    if not np.allclose(
        joined["adaptive_mlp_predicted_mw"], joined["predicted_demand_mw"],
        rtol=0.0, atol=prediction_tolerance_mw,
    ):
        raise ValueError("adaptive MLP registry predictions differ from Phase 3.8 aligned predictions")
    joined = joined.rename(columns={"predicted_demand_mw": "mlp_predicted_demand_mw", "model_version": "mlp_model_version"})
    joined = joined.merge(xgb, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_xgb"))
    if joined["predicted_demand_mw"].isna().any():
        raise ValueError("adaptive XGBoost registry is missing Phase 3.8 common forecast keys")
    if not np.allclose(
        joined["adaptive_xgboost_predicted_mw"], joined["predicted_demand_mw"],
        rtol=0.0, atol=prediction_tolerance_mw,
    ):
        raise ValueError("adaptive XGBoost registry predictions differ from Phase 3.8 aligned predictions")
    joined = joined.rename(columns={"predicted_demand_mw": "xgboost_predicted_demand_mw", "model_version": "xgboost_model_version"})

    vectors: list[ForecastVector] = []
    partial_issue_count = 0
    for issue, group in joined.groupby("issue_timestamp", sort=True):
        group = group.sort_values("horizon_hours")
        if group["horizon_hours"].tolist() != list(range(1, HORIZON + 1)):
            partial_issue_count += 1
            continue
        mlp_versions = group["mlp_model_version"].astype(str).unique()
        xgb_versions = group["xgboost_model_version"].astype(str).unique()
        if len(mlp_versions) != 1 or len(xgb_versions) != 1:
            raise ValueError(f"model version changed within issue-time horizon at {issue}")
        vectors.append(ForecastVector(
            issue_timestamp=issue,
            target_timestamps=tuple(group["target_timestamp"].tolist()),
            mlp_predicted_demand_mw=tuple(group["mlp_predicted_demand_mw"].astype(float).tolist()),
            xgboost_predicted_demand_mw=tuple(group["xgboost_predicted_demand_mw"].astype(float).tolist()),
            mlp_model_version=mlp_versions[0],
            xgboost_model_version=xgb_versions[0],
        ))
    return ForecastVectorSet(
        vectors=tuple(vectors),
        aligned_prediction_keys=len(joined),
        partial_issue_count=partial_issue_count,
    )


def paired_initial_states(battery_soc_mwh: float, hydro_remaining_mwh: float, conventional_output_mw: float):
    """Return equal, independent DispatchState values for the paired runs."""
    from griddrift.dispatch import DispatchState

    return (
        DispatchState(battery_soc_mwh, hydro_remaining_mwh, conventional_output_mw),
        DispatchState(battery_soc_mwh, hydro_remaining_mwh, conventional_output_mw),
    )
