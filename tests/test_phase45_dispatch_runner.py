import numpy as np
import pandas as pd
import pytest

from griddrift.dispatch import load_dispatch_config
from griddrift.simulation.phase45_dispatch import run_phase45_dispatch_comparison
from griddrift.simulation.phase45_inputs import ForecastVector, Phase45InputSettings


@pytest.fixture
def assets():
    return load_dispatch_config("configs/assets.yaml")[0]


def two_hour_fixture():
    issues = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    vectors = []
    for number, issue in enumerate(issues):
        targets = tuple(issue + pd.Timedelta(hours=step) for step in range(1, 25))
        base = np.asarray([48_000 + 100 * step + 100 * number for step in range(24)])
        vectors.append(ForecastVector(
            issue_timestamp=issue,
            target_timestamps=targets,
            mlp_predicted_demand_mw=tuple(base),
            xgboost_predicted_demand_mw=tuple(base + 1_250),
            mlp_model_version=f"mlp-v{number}",
            xgboost_model_version=f"xgb-v{number}",
        ))
    actual_index = pd.date_range("2025-01-01 00:00", periods=4, freq="h", tz="UTC")
    actuals = pd.Series([64_000, 35_000, 69_000, 51_000], index=actual_index, name="actual_demand_mw")
    return vectors, actuals


def test_runner_plans_both_models_before_next_hour_reveal_and_keeps_state(assets, monkeypatch):
    from griddrift.simulation import phase45_dispatch as runner_module

    vectors, actuals = two_hour_fixture()
    call_sequence = []
    real_optimizer = runner_module.optimize_dispatch

    def observe_solver_calls(*args, **kwargs):
        call_sequence.append("plan" if kwargs.get("commitment_schedule") is None else "recourse")
        return real_optimizer(*args, **kwargs)

    monkeypatch.setattr(runner_module, "optimize_dispatch", observe_solver_calls)
    output = run_phase45_dispatch_comparison(
        vectors,
        actuals,
        assets,
        settings=Phase45InputSettings(resource_seed=42),
    )

    assert len(output.plan_rows) == 2 * 2 * 24
    assert len(output.execution_rows) == 2 * 2
    first_issue = vectors[0].issue_timestamp
    first_target = vectors[0].target_timestamps[0]
    plans = output.plan_rows.query("issue_timestamp == @first_issue and horizon_hours == 1")
    assert set(plans.model) == {"adaptive_mlp", "adaptive_xgboost"}
    assert plans.loc[plans.model == "adaptive_mlp", "demand_mw"].iloc[0] == pytest.approx(48.0)
    assert plans.loc[plans.model == "adaptive_xgboost", "demand_mw"].iloc[0] == pytest.approx(49.25)
    # Actual at the first target is 35 MW scaled; neither plan was built from it.
    assert actuals.loc[first_target] == 35_000
    assert (plans.demand_mw != actuals.loc[first_target] * output.settings.demand_scale).all()

    execution = output.execution_rows.sort_values(["model", "issue_timestamp"])
    assert (execution.groupby("issue_timestamp").actual_demand_mw.nunique() == 1).all()
    assert (execution.groupby("issue_timestamp").actual_solar_available_mw.nunique() == 1).all()
    assert (execution.groupby("issue_timestamp").actual_wind_available_mw.nunique() == 1).all()
    for _model, rows in execution.groupby("model"):
        rows = rows.sort_values("issue_timestamp")
        assert rows.iloc[1].initial_battery_soc_mwh == pytest.approx(rows.iloc[0].ending_battery_soc_mwh)
        assert rows.iloc[1].initial_hydro_remaining_mwh == pytest.approx(rows.iloc[0].ending_hydro_remaining_mwh)
        assert rows.iloc[1].initial_conventional_output_mw == pytest.approx(rows.iloc[0].ending_conventional_output_mw)
    first_execution = output.execution_rows.query("issue_timestamp == @first_issue").set_index("model")
    first_plans = output.plan_rows.query("issue_timestamp == @first_issue")
    for model, plan_rows in first_plans.groupby("model"):
        terminal_soc = plan_rows.sort_values("horizon_hours").soc_mwh.iloc[-1]
        assert terminal_soc == pytest.approx(first_execution.loc[model, "initial_battery_soc_mwh"])
    assert set(output.execution_rows.model_version) == {"mlp-v0", "mlp-v1", "xgb-v0", "xgb-v1"}
    assert (output.execution_rows.termination_condition == "optimal").all()
    assert call_sequence == ["plan", "plan", "recourse", "recourse"] * 2
    actual_recourse_supply = output.execution_rows[[
        "recourse_solar_to_demand_mw", "recourse_wind_to_demand_mw", "recourse_hydro_mw",
        "recourse_conventional_mw", "recourse_battery_discharge_mw", "recourse_emergency_mw",
        "recourse_unserved_mw",
    ]].sum(axis=1)
    actual_recourse_need = (
        output.execution_rows.recourse_demand_mw
        + output.execution_rows.recourse_battery_charge_mw
    )
    assert np.allclose(actual_recourse_supply, actual_recourse_need, atol=1e-6)


def test_plan_objectives_are_not_double_counted_as_realized_cost(assets):
    vectors, actuals = two_hour_fixture()
    output = run_phase45_dispatch_comparison(vectors, actuals, assets)
    breakdown_columns = [
        column for column in output.execution_rows.columns
        if column.startswith("realized_cost_") and column != "realized_cost_eur"
    ]
    assert len(breakdown_columns) == 7
    assert np.allclose(
        output.execution_rows[breakdown_columns].sum(axis=1),
        output.execution_rows.realized_cost_eur,
        atol=1e-5,
    )
    for _model, rows in output.execution_rows.groupby("model"):
        assert (rows.plan_24h_objective_eur > 0).all()
        assert len(rows) == 2  # exactly one executed interval per issue, no 24h objective added
