import numpy as np
import pandas as pd
import pytest

from griddrift.data.renewables import (
    RenewableDataError,
    RenewableResourceProvider,
    preprocess_entsoe_renewables,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T01:00:00+01:00",
                "2025-01-01T02:00:00+01:00",
            ],
            "demand_mw": [100.0, 110.0],
            "solar_actual_mw": [1.0, 2.0],
            "wind_onshore_actual_mw": [20.0, 21.0],
            "wind_offshore_actual_mw": [5.0, 6.0],
            "wind_actual_mw": [25.0, 27.0],
            "solar_day_ahead_forecast_mw": [3.0, 4.0],
            "wind_onshore_day_ahead_forecast_mw": [30.0, 31.0],
            "wind_offshore_day_ahead_forecast_mw": [7.0, 8.0],
            "day_ahead_price_eur_mwh": [50.0, -2.0],
        }
    )


def test_parses_timezone_and_aligns_values_to_hourly_utc_targets():
    clean = preprocess_entsoe_renewables(_source())

    assert clean.target_timestamp_utc.tolist() == list(
        pd.date_range("2025-01-01T00:00:00Z", periods=2, freq="h")
    )
    assert clean.demand_actual_mw.tolist() == [100.0, 110.0]
    assert clean.source_row_present.tolist() == [True, True]


def test_preserves_mw_and_eur_per_mwh_and_sums_wind_components():
    clean = preprocess_entsoe_renewables(_source())

    assert clean.wind_day_ahead_forecast_mw.tolist() == [37.0, 39.0]
    assert clean.wind_actual_mw.tolist() == [25.0, 27.0]
    assert clean.solar_day_ahead_forecast_mw.tolist() == [3.0, 4.0]
    assert clean.day_ahead_market_price_eur_mwh.tolist() == [50.0, -2.0]


def test_keeps_reservoir_hydro_generation_separate_from_inflow():
    source = _source()
    source["hydro_water_reservoir_actual_mw"] = [8.0, 9.0]
    source["hydro_run_of_river_and_pondage_actual_mw"] = [20.0, 21.0]
    clean = preprocess_entsoe_renewables(source)
    assert clean.hydro_water_reservoir_actual_mw.tolist() == [8.0, 9.0]
    assert clean.hydro_run_of_river_and_pondage_actual_mw.tolist() == [20.0, 21.0]
    assert "synthetic_2025_hydro_inflow_mwh" not in clean


def test_missing_forecasts_are_not_imputed_and_missing_hours_are_flagged():
    source = _source()
    source.loc[1, "timestamp"] = "2025-01-01T03:00:00+01:00"
    source.loc[0, "wind_onshore_day_ahead_forecast_mw"] = np.nan
    source.loc[0, "solar_day_ahead_forecast_mw"] = np.nan
    clean = preprocess_entsoe_renewables(source)

    assert len(clean) == 3
    assert clean.source_row_present.tolist() == [True, False, True]
    assert clean.any_renewable_forecast_missing.tolist() == [True, True, False]
    assert pd.isna(clean.solar_day_ahead_forecast_mw.iloc[0])
    assert pd.isna(clean.wind_day_ahead_forecast_mw.iloc[0])
    assert clean.wind_forecast_missing.tolist() == [True, True, False]
    assert clean.wind_forecast_missing.iloc[1]
    assert np.isfinite(clean.wind_actual_mw.iloc[0])
    assert pd.isna(clean.wind_day_ahead_forecast_mw.iloc[0])
    assert clean.solar_actual_mw.iloc[0] > 0
    assert pd.isna(clean.solar_day_ahead_forecast_mw.iloc[0])


def test_issue_times_are_preserved_when_present_and_never_inferred():
    source = _source()
    source["solar_forecast_issue_timestamp"] = [
        "2024-12-31T11:00:00-05:00",
        "2024-12-31T12:00:00-05:00",
    ]
    clean = preprocess_entsoe_renewables(source)
    assert clean.solar_forecast_issue_timestamp_utc.iloc[0] == pd.Timestamp(
        "2024-12-31T16:00:00Z"
    )
    assert clean.wind_forecast_issue_timestamp_utc.isna().all()

    without_issue_times = preprocess_entsoe_renewables(_source())
    assert without_issue_times.solar_forecast_issue_timestamp_utc.isna().all()
    assert without_issue_times.wind_forecast_issue_timestamp_utc.isna().all()


def test_rejects_negative_generation_and_duplicate_timestamps():
    negative = _source()
    negative.loc[0, "solar_actual_mw"] = -0.1
    with pytest.raises(RenewableDataError, match="negative physical"):
        preprocess_entsoe_renewables(negative)

    duplicate = _source()
    duplicate.loc[1, "timestamp"] = duplicate.loc[0, "timestamp"]
    with pytest.raises(RenewableDataError, match="duplicate"):
        preprocess_entsoe_renewables(duplicate)


def test_resource_provider_uses_only_forecasts_and_applies_asof_assumption():
    clean = preprocess_entsoe_renewables(_source())
    provider = RenewableResourceProvider(clean)
    targets = pd.DatetimeIndex(clean.target_timestamp_utc)

    before_release = provider.forecast_at(
        pd.Timestamp("2024-12-30T23:00:00Z"), targets, scale=0.001
    )
    at_release = provider.forecast_at(
        pd.Timestamp("2024-12-31T00:00:00Z"), targets, scale=0.001
    )
    assert before_release.solar_available_mw.eq(0.0).all()
    assert before_release.solar_forecast_available_asof.eq(False).all()
    assert at_release.solar_available_mw.iloc[0] == pytest.approx(0.003)
    assert at_release.wind_available_mw.iloc[0] == pytest.approx(0.037)

    changed = clean.copy()
    changed["solar_actual_mw"] = 999_999.0
    changed["wind_actual_mw"] = 999_999.0
    changed_provider = RenewableResourceProvider(changed)
    changed_forecast = changed_provider.forecast_at(
        pd.Timestamp("2024-12-31T00:00:00Z"), targets, scale=0.001
    )
    pd.testing.assert_series_equal(at_release.solar_available_mw, changed_forecast.solar_available_mw)
    pd.testing.assert_series_equal(at_release.wind_available_mw, changed_forecast.wind_available_mw)

    actual = provider.actual_at(targets[0], scale=0.001)
    assert actual["solar_actual_mw"] == pytest.approx(0.001)
    assert actual["wind_actual_mw"] == pytest.approx(0.025)
