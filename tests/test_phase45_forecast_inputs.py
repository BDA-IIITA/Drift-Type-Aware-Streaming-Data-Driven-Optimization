from __future__ import annotations

import pandas as pd
import pytest

from griddrift.simulation.phase45_inputs import (
    ForecastVector,
    Phase45InputSettings,
    load_phase45_forecast_vectors,
    paired_initial_states,
)


def _fixture_frames(*, partial_mlp: bool = False):
    issue = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    rows = []
    mlp = []
    xgb = []
    for horizon in range(1, 25):
        target = issue + pd.Timedelta(hours=horizon)
        rows.append({
            "issue_timestamp": issue,
            "target_timestamp": target,
            "horizon_hours": horizon,
            "adaptive_mlp_predicted_mw": 50_000 + horizon,
            "adaptive_xgboost_predicted_mw": 49_000 + horizon,
            # Deliberately extreme truth verifies that it is not copied into plans.
            "actual_demand_mw": 999_999,
        })
        mlp.append({"issue_timestamp": issue, "target_timestamp": target,
                    "horizon_hours": horizon, "predicted_demand_mw": 50_000 + horizon,
                    "model_version": "mlp-version-7"})
        xgb.append({"issue_timestamp": issue, "target_timestamp": target,
                    "horizon_hours": horizon, "predicted_demand_mw": 49_000 + horizon,
                    "model_version": "xgb-version-12"})
    aligned = pd.DataFrame(rows)
    mlp_registry = pd.DataFrame(mlp)
    xgb_registry = pd.DataFrame(xgb)
    if partial_mlp:
        mlp_registry = mlp_registry.iloc[:-1].copy()
    return aligned, mlp_registry, xgb_registry


def test_builds_full_vector_and_preserves_each_model_version_without_actuals():
    aligned, mlp, xgb = _fixture_frames()
    result = load_phase45_forecast_vectors(aligned, mlp, xgb)

    assert len(result.vectors) == 1
    vector = result.vectors[0]
    assert vector.target_timestamps[0] == vector.issue_timestamp + pd.Timedelta(hours=1)
    assert vector.target_timestamps[-1] == vector.issue_timestamp + pd.Timedelta(hours=24)
    assert vector.mlp_model_version == "mlp-version-7"
    assert vector.xgboost_model_version == "xgb-version-12"
    assert vector.mlp_predicted_demand_mw[0] == 50_001
    assert vector.xgboost_predicted_demand_mw[-1] == 49_024
    assert not hasattr(vector, "actual_demand_mw")


def test_excludes_partial_boundary_issue_time_and_reports_it():
    aligned, mlp, xgb = _fixture_frames(partial_mlp=True)
    with pytest.raises(ValueError, match="missing Phase 3.8 common forecast keys"):
        load_phase45_forecast_vectors(aligned, mlp, xgb)

    # A partial row set from the common source is a boundary, not a valid plan.
    aligned, mlp, xgb = _fixture_frames()
    aligned = aligned.iloc[:-1].copy()
    mlp = mlp.iloc[:-1].copy()
    xgb = xgb.iloc[:-1].copy()
    result = load_phase45_forecast_vectors(aligned, mlp, xgb)
    assert result.vectors == ()
    assert result.partial_issue_count == 1


def test_rejects_issue_target_horizon_mismatch():
    aligned, mlp, xgb = _fixture_frames()
    aligned.loc[0, "target_timestamp"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="target_timestamp = issue_timestamp"):
        load_phase45_forecast_vectors(aligned, mlp, xgb)


def test_rejects_prediction_values_that_do_not_match_saved_registry():
    aligned, mlp, xgb = _fixture_frames()
    xgb.loc[0, "predicted_demand_mw"] += 10
    with pytest.raises(ValueError, match="XGBoost registry predictions differ"):
        load_phase45_forecast_vectors(aligned, mlp, xgb)


def test_settings_lock_common_scenario_and_renewable_only_policy():
    settings = Phase45InputSettings()
    assert settings.horizon_hours == 24
    assert settings.demand_scale == 0.001
    assert settings.battery_charging_policy == "renewable_only"
    with pytest.raises(ValueError, match="renewable-only"):
        Phase45InputSettings(battery_charging_policy="unrestricted")


def test_paired_initial_states_are_equal_but_independent_values():
    mlp_state, xgb_state = paired_initial_states(15.0, 120.0, 0.0)
    assert mlp_state == xgb_state
    assert mlp_state is not xgb_state


def test_forecast_vector_rejects_wrong_horizon_or_missing_version():
    issue = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    targets = tuple(issue + pd.Timedelta(hours=h) for h in range(1, 24))
    with pytest.raises(ValueError, match=r"issue\+1 through issue\+24"):
        ForecastVector(issue, targets, tuple([1.0] * 23), tuple([2.0] * 23), "mlp", "xgb")
