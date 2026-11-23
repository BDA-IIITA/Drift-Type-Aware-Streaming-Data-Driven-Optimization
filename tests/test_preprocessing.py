import pandas as pd
import pytest

from griddrift.data.preprocessing import normalize_demand, normalize_hourly_demand


def test_normalize_uses_utc_hourly_mean_for_power():
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:30:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T01:30:00Z",
            ],
            "demand_mw": [10.0, 14.0, 20.0, 24.0],
        }
    )
    result = normalize_hourly_demand(raw)
    assert list(result["demand_mw"]) == [12.0, 22.0]
    assert str(result.index.tz) == "UTC"


def test_normalize_rejects_missing_hour():
    raw = pd.DataFrame(
        {"timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z"], "demand_mw": [1, 2]}
    )
    with pytest.raises(ValueError, match="missing 1h"):
        normalize_hourly_demand(raw)


def test_normalize_accepts_a_configurable_target_resolution():
    raw = pd.DataFrame(
        {"timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:30:00Z"], "demand_mw": [1, 2]}
    )
    result = normalize_demand(raw, resolution="30min")
    assert result["demand_mw"].tolist() == [1.0, 2.0]
