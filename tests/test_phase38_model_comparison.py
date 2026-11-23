import pandas as pd
import pytest

from griddrift.forecasting.comparison import align_forecasts, normalize_forecasts


def test_alignment_uses_common_issue_target_horizon_keys_and_source_actuals():
    issue = pd.Timestamp("2025-01-01T00:00:00Z")
    target = issue + pd.Timedelta(hours=24)
    actual = pd.Series([50_000.0], index=pd.DatetimeIndex([target]))
    first = pd.DataFrame({
        "issue_timestamp": [issue, issue + pd.Timedelta(hours=1)],
        "target_timestamp": [target, target + pd.Timedelta(hours=1)],
        "horizon_hours": [24, 24],
        "predicted_demand_mw": [49_000.0, 48_000.0],
    })
    second = first.iloc[:1].copy()
    second["predicted_demand_mw"] = 50_200.0
    joined = align_forecasts({"first": first, "second": second}, actual)
    assert len(joined) == 1
    assert joined.loc[0, "actual_demand_mw"] == 50_000.0
    assert joined.loc[0, "first_predicted_mw"] == 49_000.0
    assert joined.loc[0, "second_predicted_mw"] == 50_200.0


def test_normalize_rejects_issue_target_horizon_mismatch():
    frame = pd.DataFrame({
        "issue_timestamp": ["2025-01-01T00:00:00Z"],
        "target_timestamp": ["2025-01-02T00:00:00Z"],
        "horizon_hours": [23],
        "predicted_demand_mw": [50_000.0],
    })
    with pytest.raises(ValueError, match="do not match"):
        normalize_forecasts(frame, "sample")


def test_normalize_rejects_duplicate_forecast_keys():
    row = {
        "issue_timestamp": "2025-01-01T00:00:00Z",
        "target_timestamp": "2025-01-02T00:00:00Z",
        "horizon_hours": 24,
        "predicted_demand_mw": 50_000.0,
    }
    with pytest.raises(ValueError, match="duplicate"):
        normalize_forecasts(pd.DataFrame([row, row]), "sample")
