import numpy as np

from griddrift.data.synthetic import make_synthetic_scenario


def test_synthetic_scenario_is_deterministic_and_explicitly_simulated():
    first, metadata = make_synthetic_scenario(periods=48, seed=7)
    second, _ = make_synthetic_scenario(periods=48, seed=7)
    assert metadata.is_simulated
    assert first.equals(second)
    assert (first >= 0).all().all()
    assert first.index.freq == "h"


def test_synthetic_forecasts_are_separate_from_actuals_and_capacity_bounded():
    frame, _ = make_synthetic_scenario(periods=48, seed=3, solar_capacity_mw=12, wind_capacity_mw=16)
    assert not np.array_equal(frame["solar_actual_mw"], frame["solar_forecast_mw"])
    assert not np.array_equal(frame["wind_actual_mw"], frame["wind_forecast_mw"])
    assert frame[["solar_actual_mw", "solar_forecast_mw"]].to_numpy().max() <= 12
    assert frame[["wind_actual_mw", "wind_forecast_mw"]].to_numpy().max() <= 16
