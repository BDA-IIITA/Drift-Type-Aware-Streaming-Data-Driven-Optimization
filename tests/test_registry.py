import pandas as pd

from griddrift.forecasting.registry import ForecastRegistry


def test_registry_aligns_target_actual_and_fixed_24_hour_error():
    issue = pd.Timestamp("2024-03-01T00:00:00Z")
    prediction = pd.Series(
        range(1, 25), index=pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="1h"), dtype=float
    )
    registry = ForecastRegistry()
    registry.record_horizon(issue, prediction, "mlp-test")

    matches = registry.match_actual(issue + pd.Timedelta(hours=24), actual_demand_mw=30.0)
    assert len(matches) == 1
    assert matches.iloc[0]["horizon_hours"] == 24
    assert matches.iloc[0]["absolute_error_mw"] == 6.0
    fixed = registry.fixed_horizon_errors(issue + pd.Timedelta(hours=24), 30.0)
    assert len(fixed) == 1
