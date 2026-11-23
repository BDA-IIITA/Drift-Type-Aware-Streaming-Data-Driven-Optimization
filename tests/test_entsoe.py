import pandas as pd
import pytest
from pathlib import Path

from griddrift.data.entsoe import EntsoeLoadError, fetch_entsoe_actual_load


class FakeLoadClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def query_load(self, country_code, start, end):
        self.calls.append((country_code, start, end))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_entsoe_fetch_normalizes_and_reuses_cache():
    index = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="30min")
    client = FakeLoadClient(pd.Series([10.0, 14.0, 20.0, 24.0], index=index))
    fetched, path, cached = fetch_entsoe_actual_load(
        bidding_zone="DE-LU",
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T02:00:00Z",
        cache_dir=Path("data/processed/test_entsoe_cache"),
        force_refresh=True,
        api_token="not-a-real-token",
        client=client,
    )
    assert not cached
    assert path.exists()
    assert fetched["demand_mw"].tolist() == [12.0, 22.0]
    assert client.calls[0][0] == "DE_LU"

    cached_frame, _, from_cache = fetch_entsoe_actual_load(
        bidding_zone="DE-LU",
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T02:00:00Z",
        cache_dir=Path("data/processed/test_entsoe_cache"),
        client=FakeLoadClient(RuntimeError("network must not be called")),
    )
    assert from_cache
    pd.testing.assert_frame_equal(fetched, cached_frame)


def test_entsoe_failure_never_becomes_synthetic_data():
    with pytest.raises(EntsoeLoadError, match="no synthetic fallback") as error:
        fetch_entsoe_actual_load(
            bidding_zone="FR",
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T02:00:00Z",
            cache_dir=Path("data/processed/test_entsoe_failure_cache"),
            force_refresh=True,
            api_token="not-a-real-token",
            client=FakeLoadClient(RuntimeError("request failed: securityToken=do-not-leak")),
        )
    assert error.value.__cause__ is None
    assert "do-not-leak" not in str(error.value)
