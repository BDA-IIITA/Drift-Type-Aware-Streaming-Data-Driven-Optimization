"""Interactive GridDRIFT rolling forecast and dispatch demonstration."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from griddrift.dashboard.model_selection import (
    ADAPTIVE_MLP,
    ADAPTIVE_XGBOOST,
    FORECASTING_MODEL_LABELS,
    initialize_forecasting_model,
    load_selected_adaptation_config,
    load_selected_forecaster,
    model_checkpoint_version,
    model_switch_pending,
)
from griddrift.dashboard.views import (
    daily_planned_cost_eur,
    dispatch_download_frame,
    streaming_forecast_frame,
    validate_forecast_plan_alignment,
)
from griddrift.data import (
    RenewableResourceProvider,
    load_assumed_hourly_inflow_series,
    load_reservoir_parameters_json,
)
from griddrift.dispatch import load_dispatch_config
from griddrift.dispatch.baselines import run_greedy_dispatch
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.simulation import (
    IntegratedConfig,
    IntegratedSimulation,
    ReplayConfig,
    load_integrated_settings,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATA = ROOT / "data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20250101T0000Z_1h.csv"
DEFAULT_DATA = ROOT / "data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv"
DEFAULT_REPLAY_START = pd.Timestamp("2025-01-01T00:00:00Z")
DEFAULT_MODEL = ROOT / "artifacts/models/de-lu-mlp"
DEFAULT_XGBOOST_MODEL = (
    ROOT / "artifacts/experiments/phase37/static/static-multioutput-20261023/static_xgboost_model"
)
FINAL_RESOURCE_DIR = ROOT / "artifacts/experiments/phase55/final_integration"
DEFAULT_RENEWABLE_DATA = FINAL_RESOURCE_DIR / "entsoe_de_lu_2025_renewables_hourly.csv"
DEFAULT_HYDRO_PARAMETERS = FINAL_RESOURCE_DIR / "pecd_de00_reservoir_parameters.json"
DEFAULT_HYDRO_INFLOW = FINAL_RESOURCE_DIR / "pecd_de00_synthetic_hourly_hydro_2025.csv"
SIMULATED_RESOURCE_MODE = "simulated"
HISTORICAL_RESOURCE_MODE = "entsoe_pecd"
HYDRO_CHARGING_DEFAULT_SESSION_KEY = "hydro_charging_default_version"


def _read_demand(path: str) -> pd.Series:
    frame = pd.read_csv(path, low_memory=False)
    if {"timestamp", "demand_mw"}.issubset(frame.columns):
        timestamp_column, demand_column = "timestamp", "demand_mw"
    elif {"target_timestamp_utc", "demand_actual_mw"}.issubset(frame.columns):
        timestamp_column, demand_column = "target_timestamp_utc", "demand_actual_mw"
    else:
        raise ValueError(
            "demand CSV must contain timestamp/demand_mw or "
            "target_timestamp_utc/demand_actual_mw"
        )
    index = pd.DatetimeIndex(pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce"))
    values = pd.to_numeric(frame[demand_column], errors="coerce")
    if index.isna().any() or values.isna().any():
        raise ValueError("demand CSV contains invalid timestamps or missing demand values")
    return pd.Series(values.to_numpy(dtype=float), index=index, name="demand_mw")


def _read_demand_bounds(path: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    columns = set(pd.read_csv(path, nrows=0).columns)
    timestamp_column = (
        "timestamp" if "timestamp" in columns else
        "target_timestamp_utc" if "target_timestamp_utc" in columns else None
    )
    if timestamp_column is None:
        raise ValueError("demand CSV must contain timestamp or target_timestamp_utc")
    timestamps = pd.read_csv(path, usecols=[timestamp_column])[timestamp_column]
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce"))
    if index.isna().any():
        raise ValueError("demand CSV contains invalid timestamps")
    if index.empty or not index.is_monotonic_increasing:
        raise ValueError("demand CSV timestamps must be non-empty and chronological")
    return index[0], index[-1]


def _load_resource_inputs(resource_mode: str, defaults: dict):
    """Load one resource mode's inputs; return no providers for simulated mode."""
    if resource_mode == SIMULATED_RESOURCE_MODE:
        return None, None, None
    if resource_mode != HISTORICAL_RESOURCE_MODE:
        raise ValueError(f"unsupported resource data mode: {resource_mode}")
    required = (DEFAULT_RENEWABLE_DATA, DEFAULT_HYDRO_PARAMETERS, DEFAULT_HYDRO_INFLOW)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("historical renewable/hydro inputs are missing: " + ", ".join(missing))
    provider = RenewableResourceProvider(
        DEFAULT_RENEWABLE_DATA,
        issue_time_policy=defaults["renewable_forecast_issue_policy"],
    )
    return (
        provider,
        load_reservoir_parameters_json(DEFAULT_HYDRO_PARAMETERS),
        load_assumed_hourly_inflow_series(DEFAULT_HYDRO_INFLOW),
    )


def _limit_demand_to_resource_horizon(
    demand: pd.Series, provider: RenewableResourceProvider | None
) -> pd.Series:
    """Keep enough historical resource coverage for the last complete 24h plan."""
    if provider is None:
        return demand
    last_issue = provider.target_timestamps[-1] - pd.Timedelta(hours=24)
    limited = demand.loc[demand.index <= last_issue]
    if limited.empty:
        raise ValueError("demand data does not overlap the historical renewable resource coverage")
    return limited


def _replay_config_for_data(path: str) -> ReplayConfig:
    """Keep normal validation; demo data skips only the acceptance gate."""
    config = load_selected_adaptation_config(ADAPTIVE_MLP, ROOT)
    if "demo_drift_3events" in Path(path).stem.lower():
        return replace(config, adaptation_policy="increase_only")
    return config


def _build_simulation(setup: tuple, defaults: dict) -> IntegratedSimulation:
    demand = _read_demand(setup[0])
    model_family = setup[7] if len(setup) > 7 else ADAPTIVE_MLP
    resource_mode = setup[8] if len(setup) > 8 else SIMULATED_RESOURCE_MODE
    resource_provider, hydro_parameters, hydro_inflow = _load_resource_inputs(resource_mode, defaults)
    demand = _limit_demand_to_resource_horizon(demand, resource_provider)
    model = load_selected_forecaster(
        model_family,
        mlp_checkpoint=setup[1],
        xgboost_checkpoint=DEFAULT_XGBOOST_MODEL,
    )
    assets, _ = load_dispatch_config(ROOT / "configs/assets.yaml")
    start_timestamp = setup[6] if len(setup) > 6 else None
    demo_adaptation = "demo_drift_3events" in Path(setup[0]).stem.lower()
    mlp_adaptation = _replay_config_for_data(setup[0])
    xgboost_adaptation = load_selected_adaptation_config(ADAPTIVE_XGBOOST, ROOT)
    simulation = IntegratedSimulation(
        demand, model, assets,
        config=IntegratedConfig(
            demand_scale=setup[2], policy=setup[3], resource_seed=defaults["resource_seed"],
            solar_cost_blocks_eur_mwh=defaults.get("solar_cost_blocks_eur_mwh", (8, 9, 12, 14, 11, 8)),
            wind_cost_blocks_eur_mwh=defaults.get("wind_cost_blocks_eur_mwh", (12, 11, 10, 9, 11, 13)),
            resource_scenario=setup[4], battery_charging_policy=setup[5],
            allow_hydro_battery_charging=defaults["allow_hydro_battery_charging"],
            terminal_soc_policy=defaults["terminal_soc_policy"],
            adaptation=mlp_adaptation,
            xgboost_adaptation=xgboost_adaptation,
        ),
        start_timestamp=start_timestamp,
        model_family=model_family,
        renewable_provider=resource_provider,
        hydro_parameters=hydro_parameters,
        hydro_inflow_profile=hydro_inflow,
    )
    simulation.resource_data_label = (
        "ENTSO-E 2025 day-ahead solar/wind forecasts + hourly revealed generation supply proxy + "
        "synthetic PECD seasonal reservoir inflow"
        if resource_provider is not None else "Simulated renewable forecasts and actuals"
    )
    simulation.demand_label = (
        "SIMULATED demand (three demo shifts)" if demo_adaptation else
        "SIMULATED demand (demo level shift)"
        if "demo_drift" in Path(setup[0]).stem.lower()
        else "Actual demand (observed)"
    )
    simulation.demo_adaptation = demo_adaptation
    return simulation


def _sync_hydro_charging_default(session_state, simulation, enabled: bool) -> bool:
    """Migrate a cached session plan once to the configured hydro-charging default.

    The migration marker prevents a user's subsequent manual toggle from being
    reset on Streamlit reruns. A changed flag triggers IntegratedSimulation's
    in-place 24-hour re-optimization; the forecast and simulation time remain.
    """
    marker = f"phase55-hydro-charging-default-v1:{int(bool(enabled))}"
    if session_state.get(HYDRO_CHARGING_DEFAULT_SESSION_KEY) == marker:
        return False
    changed = False
    if simulation.config.battery_charging_policy == "renewable_only":
        changed = simulation.set_hydro_battery_charging(bool(enabled))
    session_state[HYDRO_CHARGING_DEFAULT_SESSION_KEY] = marker
    widget_key = f"hydro_battery_enabled_{id(simulation)}"
    if widget_key in session_state:
        session_state[widget_key] = simulation.config.allow_hydro_battery_charging
    return changed


def _on_forecasting_model_change() -> None:
    """Pause the current session as soon as a different model is selected."""
    import streamlit as st

    selected = st.session_state.selected_forecasting_model
    active = st.session_state.get("active_forecasting_model")
    st.session_state.running = False
    st.session_state.model_switch_pending = selected != active


def _figure(sim):
    import plotly.graph_objects as go

    window_hours = 24 * 14
    frame = streaming_forecast_frame(sim, window_hours=window_hours)
    chart_start = sim.timestamp - pd.Timedelta(hours=window_hours - 1)
    chart_end = sim.forecast.index[-1]
    fig = go.Figure()
    history = frame.loc[frame.index <= sim.timestamp].dropna(subset=["actual_demand_mw"])
    future = frame.loc[frame.index > sim.timestamp].dropna(subset=["predicted_demand_mw"])
    fig.add_trace(go.Scatter(
        x=history.index, y=history.actual_demand_mw,
        name=getattr(sim, "demand_label", "Actual demand (observed)"),
        line={"color": "#16824B", "width": 2.5},
    ))
    errors = pd.DataFrame(sim.errors)
    if not errors.empty:
        errors["target_timestamp"] = pd.to_datetime(errors.target_timestamp, utc=True)
        recent_errors = errors.loc[errors.target_timestamp >= chart_start]
        # Draw the frozen static baseline last so it remains visible when the
        # two prediction series overlap; the adaptive trace remains visible
        # wherever an accepted update makes its predictions diverge.
        for policy_name, color, dash in (
            ("adaptive", "#38BDF8", "solid"),
            ("static", "#F59E0B", "dot"),
        ):
            scored = recent_errors.loc[recent_errors.model_policy == policy_name].sort_values("target_timestamp")
            if not scored.empty:
                model_versions = (
                    scored.model_version.astype(str).to_numpy()
                    if "model_version" in scored
                    else [""] * len(scored)
                )
                fig.add_trace(go.Scatter(
                    x=scored.target_timestamp, y=scored.predicted_demand_mw,
                    name=("Adaptive 24h-ahead" if policy_name == "adaptive"
                          else "Static 24h-ahead (frozen)"),
                    mode="lines+markers", line={
                        "color": color, "width": 3.0 if policy_name == "adaptive" else 2.0,
                        "dash": dash,
                    },
                    marker={
                        "size": 5,
                        "symbol": "circle" if policy_name == "adaptive" else "x",
                    },
                    customdata=model_versions,
                    hovertemplate=(
                        "%{x}<br>Prediction: %{y:,.0f} MW<br>Model: %{customdata}<extra>%{fullData.name}</extra>"
                    ),
                ))
    fig.add_trace(go.Scatter(
        x=future.index, y=future.predicted_demand_mw, name="Next 24h forecast",
        mode="lines+markers", line={"color": "#2878D0", "width": 2.8, "dash": "dash"},
        marker={"size": 4},
    ))
    drift_times, drift_values = [], []
    for event in sim.drift_events:
        drift_time = pd.Timestamp(event["drift_timestamp"])
        if drift_time in sim.actual_history.index and drift_time >= chart_start:
            drift_times.append(drift_time)
            drift_values.append(float(sim.actual_history.loc[drift_time]))
    if drift_times:
        fig.add_trace(go.Scatter(
            x=drift_times, y=drift_values, name="ADWIN drift",
            mode="markers", marker={"color": "#E11D48", "size": 12, "symbol": "x"},
        ))
    accepted_updates = [event for event in sim.adaptation_events if event.outcome == "adapted"]
    accepted_updates = [
        event for event in accepted_updates if event.drift_timestamp in sim.actual_history.index
    ]
    if accepted_updates:
        update_times = [event.drift_timestamp for event in accepted_updates]
        update_values = [float(sim.actual_history.loc[timestamp]) for timestamp in update_times]
        update_labels = [
            f"{event.previous_model_version} → {event.new_model_version}"
            for event in accepted_updates
        ]
        fig.add_trace(go.Scatter(
            x=update_times, y=update_values, name="Accepted adaptive update",
            mode="markers", marker={"color": "#C084FC", "size": 13, "symbol": "diamond"},
            customdata=update_labels,
            hovertemplate="%{x}<br>Accepted model update<br>%{customdata}<extra></extra>",
        ))
    fig.add_vline(
        x=sim.timestamp, line_color="#6B7280", line_dash="dot",
        annotation_text=f"Simulation now · {sim.timestamp:%d %b %H:%M UTC}",
        annotation_position="top left",
    )
    fig.update_layout(
        height=480, margin={"l": 10, "r": 10, "t": 35, "b": 10},
        xaxis_title="UTC time", yaxis_title="Demand (MW)", legend_orientation="h",
    )
    fig.update_xaxes(range=[chart_start, chart_end])
    return fig


def _render_live_forecasting(st, sim) -> None:
    if getattr(sim, "demand_label", "").startswith("SIMULATED"):
        if getattr(sim, "demo_adaptation", False):
            st.warning(
                "Demo only: this file contains three artificial demand shifts. ADWIN and the 7-day cooldown "
                "remain active. On a meaningful error increase, the MLP fine-tunes on recent and representative "
                "historical samples without the validation gate, so accuracy may worsen. Original ENTSO-E data is unchanged."
            )
        else:
            st.warning(
                "Demo only: the demand series has an artificial level increase to demonstrate drift. "
                "The original ENTSO-E data file is unchanged."
            )
    st.subheader("Streaming demand, scored forecasts, and the next 24 hours")
    st.plotly_chart(_figure(sim), width="stretch", key="live_demand_forecast")
    metrics = sim.metrics()
    policy = sim.config.policy
    mae, rmse = metrics[f"{policy}_mae_mw"], metrics[f"{policy}_rmse_mw"]
    c1, c2 = st.columns(2)
    c1.metric("Current simulation time", f"{sim.timestamp:%d %b %Y, %H:%M UTC}")
    c2.metric(f"{policy.title()} forecast MAE / RMSE", f"{mae:,.1f} / {rmse:,.1f} MW" if pd.notna(mae) else "Not scored yet")
    st.caption(
        f"Selected model: {FORECASTING_MODEL_LABELS.get(sim.model_family, sim.model_family)} "
        f"({sim.active_model.model_version}); issue time {sim.timestamp:%d %b %H:%M UTC}; "
        f"24 target hours ending {sim.forecast.index[-1]:%d %b %H:%M UTC}."
    )
    st.caption(
        f"Forecast issued {sim.timestamp:%d %b %H:%M UTC} · 24-hour horizon "
        f"({sim.forecast.index[0]:%d %b %H:%M}–{sim.forecast.index[-1]:%d %b %H:%M} UTC)."
    )
    st.caption(
        "Static is the frozen baseline; adaptive starts from the same model and changes only "
        "after an accepted ADWIN-triggered update. Red × marks a drift alarm; purple diamonds "
        "mark accepted model updates. Hover over a scored forecast to see its model version."
    )
    st.info(
        "We predict the entire next 24 hours, plan energy allocation for all 24 hours, "
        "execute the first hour, observe actual demand, and replan."
    )
    forecast_details = pd.DataFrame({
        "issue_timestamp": sim.timestamp,
        "target_timestamp": sim.forecast.index,
        "predicted_demand_mw": sim.forecast.to_numpy(),
        "model_version": sim.active_model.model_version,
    })
    with st.expander("Show all 24 forecast values", expanded=False):
        st.dataframe(forecast_details, width="stretch", hide_index=True)


def _render_drift_monitoring(st, px, go, sim) -> None:
    st.subheader("Fixed 24-hour-ahead forecast errors")
    if sim.model_family == ADAPTIVE_XGBOOST:
        st.info("XGBoost ADWIN observes only matched fixed 24-hour-ahead adaptive errors. Updates require an error-increase alarm, an eligible validation window, and the configured cooldown.")
    if sim.errors:
        errors = pd.DataFrame(sim.errors)
        error_chart = px.line(
            errors, x="target_timestamp", y="absolute_error_mw", color="model_policy",
            markers=True, title="Absolute forecast error (MW)",
            labels={"target_timestamp": "Actual target timestamp (UTC)", "absolute_error_mw": "Absolute error (MW)"},
        )
        for event in sim.drift_events:
            direction = event.get("direction")
            color = "#C2410C" if direction in {"increase", "error_increased"} else "#16824B"
            error_chart.add_vline(
                x=event["drift_timestamp"], line_dash="dot", line_color=color,
                annotation_text=str(event.get("direction", "change")),
            )
        st.plotly_chart(error_chart, width="stretch", key="drift_error_chart")
        actual = errors.drop_duplicates("target_timestamp").sort_values("target_timestamp")
        comparison = go.Figure()
        comparison.add_trace(go.Scatter(
            x=actual.target_timestamp, y=actual.actual_demand_mw,
            name="Actual (observed)", line={"color": "#16824B", "width": 2.5},
        ))
        for policy, color, dash in (("adaptive", "#38BDF8", "solid"), ("static", "#F59E0B", "dot")):
            subset = errors.loc[errors.model_policy == policy].sort_values("target_timestamp")
            comparison.add_trace(go.Scatter(
                x=subset.target_timestamp, y=subset.predicted_demand_mw,
                name=("Adaptive 24h prediction" if policy == "adaptive"
                      else "Static 24h prediction (frozen)"),
                mode="lines+markers", line={"color": color, "dash": dash, "width": 2},
                marker={"size": 4},
            ))
        comparison.update_layout(
            height=330, xaxis_title="Actual target timestamp (UTC)",
            yaxis_title="Demand (MW)", margin={"l": 10, "r": 10, "t": 25, "b": 10},
        )
        st.plotly_chart(comparison, width="stretch", key="drift_forecast_comparison")
    else:
        st.info("No fixed-horizon errors are available yet; errors are recorded when their actual timestamps arrive.")

    metrics = sim.metrics()
    accepted_count = sum(event.outcome == "adapted" for event in sim.adaptation_events)
    attempt_count = sum(
        event.outcome in {"adapted", "candidate_rejected"}
        or event.outcome.startswith("candidate_rejected:")
        for event in sim.adaptation_events
    )
    rejected_count = max(0, attempt_count - accepted_count)
    alarm_metric, attempt_metric, accepted_metric, rejected_metric = st.columns(4)
    alarm_metric.metric("ADWIN changes", len(sim.drift_events))
    attempt_metric.metric("Candidate attempts", attempt_count)
    accepted_metric.metric("Accepted updates", accepted_count)
    rejected_metric.metric("Rejected candidates", rejected_count)
    if sim.drift_events:
        drift = pd.DataFrame(sim.drift_events).copy()
        drift["interpretation"] = drift.direction.map({
            "increase": "Increased error", "error_increased": "Increased error",
            "decrease": "Decreased error", "error_decreased": "Decreased error",
        }).fillna("Change detected")
        st.markdown("**ADWIN changes and error direction**")
        st.dataframe(drift, width="stretch", hide_index=True)
    if sim.adaptation_events:
        events = pd.DataFrame([event.__dict__ for event in sim.adaptation_events])
        events["decision"] = events.outcome.map(
            lambda outcome: "Accepted" if outcome == "adapted" else "Rejected / skipped"
        )
        # Demo-only increase-triggered retraining does not run candidate
        # validation; omit those wholly empty fields instead of showing None.
        events = events.dropna(axis=1, how="all")
        st.markdown("**Accepted and rejected adaptation events / model versions**")
        st.dataframe(events, width="stretch", hide_index=True)
    else:
        st.caption(
            f"No candidate has been deployed. Frozen static model: {sim.static_model.model_version}; "
            f"current adaptive model: {sim.replay.adaptive_model.model_version}. Candidates are evaluated "
            "only after an eligible ADWIN error-increase alarm."
        )

    st.markdown("**Static vs adaptive forecasting performance (MW)**")
    st.dataframe(pd.DataFrame([
        {"Model": "Frozen static", "MAE": metrics["static_mae_mw"], "RMSE": metrics["static_rmse_mw"]},
        {"Model": "Adaptive", "MAE": metrics["adaptive_mae_mw"], "RMSE": metrics["adaptive_rmse_mw"]},
    ]).round(2), width="stretch", hide_index=True)
    if sim.errors:
        st.markdown("**Recent scored forecasts**")
        scored = pd.DataFrame(sim.errors).tail(100).drop(columns=["horizon_hours"], errors="ignore")
        scored = scored.rename(columns={
            "issue_timestamp": "Forecast issued (UTC)",
            "target_timestamp": "Actual hour (UTC)",
            "predicted_demand_mw": "Predicted demand (MW)",
            "actual_demand_mw": "Actual demand (MW)",
            "absolute_error_mw": "Absolute error (MW)",
            "model_policy": "Model",
            "model_version": "Model version",
        })
        st.dataframe(scored, width="stretch", hide_index=True)


def _dispatch_table(sim) -> pd.DataFrame:
    schedule = dispatch_download_frame(sim)
    # A Streamlit process can keep an older imported dispatch module alive
    # until it is restarted. Keep the table usable for such an in-memory plan;
    # newly computed plans carry the exact (possibly time-varying) solver rates.
    rate_defaults = {
        "solar_cost_rate_eur_mwh": sim.assets.solar.marginal_cost_eur_mwh,
        "wind_cost_rate_eur_mwh": sim.assets.wind.marginal_cost_eur_mwh,
        "hydro_cost_rate_eur_mwh": sim.assets.hydro.marginal_cost_eur_mwh,
        "conventional_cost_rate_eur_mwh": sim.assets.conventional.marginal_cost_eur_mwh,
        "battery_degradation_rate_eur_mwh": sim.assets.battery.degradation_cost_eur_mwh,
    }
    for column, default in rate_defaults.items():
        if column not in schedule:
            schedule[column] = default
    if "hydro_to_demand_mw" not in schedule:
        schedule["hydro_to_demand_mw"] = schedule.get("hydro_mw", 0.0)
    if "hydro_to_battery_mw" not in schedule:
        schedule["hydro_to_battery_mw"] = 0.0
    if "hydro_remaining_mwh" not in schedule:
        schedule["hydro_remaining_mwh"] = 0.0
    if "hydro_inflow_mw" not in schedule:
        schedule["hydro_inflow_mw"] = 0.0
    schedule.insert(1, "predicted_demand_original_mw", sim.forecast.to_numpy())
    schedule["solar_utilized_mw"] = schedule.solar_to_demand_mw + schedule.solar_to_battery_mw
    schedule["wind_utilized_mw"] = schedule.wind_to_demand_mw + schedule.wind_to_battery_mw
    renamed = schedule.rename(columns={
        "demand_mw": "Dispatch demand (scaled MW)",
        "predicted_demand_original_mw": "Predicted demand (MW)",
        "solar_available_mw": "Solar forecast input (MW)", "solar_utilized_mw": "Solar utilized (MW)",
        "solar_to_demand_mw": "Solar to demand (MW)", "solar_to_battery_mw": "Solar to battery (MW)",
        "solar_curtailment_mw": "Solar curtailed (MW)",
        "wind_available_mw": "Wind forecast input (MW)", "wind_utilized_mw": "Wind utilized (MW)",
        "wind_to_demand_mw": "Wind to demand (MW)", "wind_to_battery_mw": "Wind to battery (MW)",
        "wind_curtailment_mw": "Wind curtailed (MW)", "hydro_mw": "Hydro generation (MW)",
        "hydro_to_demand_mw": "Hydro to demand (MW)",
        "hydro_to_battery_mw": "Hydro to battery (MW)",
        "hydro_inflow_mw": "Hydro inflow (MW)",
        "hydro_remaining_mwh": "Hydro remaining (MWh)",
        "conventional_mw": "Conventional generation (MW)",
        "battery_charge_mw": "Battery charging (MW)", "battery_discharge_mw": "Battery discharging (MW)",
        "soc_mwh": "Battery SOC (MWh)", "emergency_mw": "Emergency balancing (MW)",
        "unserved_mw": "Unmet demand (MW)", "cost_total_eur": "Hourly planned cost (EUR)",
        "solar_cost_rate_eur_mwh": "Solar €/MWh",
        "wind_cost_rate_eur_mwh": "Wind €/MWh",
        "hydro_cost_rate_eur_mwh": "Hydro €/MWh",
        "conventional_cost_rate_eur_mwh": "Conventional €/MWh",
        "battery_degradation_rate_eur_mwh": "Battery wear €/MWh",
    })
    columns = [
        "timestamp", "Predicted demand (MW)", "Dispatch demand (scaled MW)",
        "Solar forecast input (MW)", "Solar utilized (MW)", "Solar to demand (MW)", "Solar to battery (MW)",
        "Solar curtailed (MW)", "Wind forecast input (MW)", "Wind utilized (MW)", "Wind to demand (MW)",
        "Wind to battery (MW)", "Wind curtailed (MW)", "Hydro generation (MW)",
        "Hydro to demand (MW)", "Hydro to battery (MW)", "Hydro inflow (MW)",
        "Hydro remaining (MWh)",
        "Conventional generation (MW)", "Battery charging (MW)", "Battery discharging (MW)",
        "Battery SOC (MWh)", "Solar €/MWh", "Wind €/MWh", "Hydro €/MWh",
        "Conventional €/MWh", "Battery wear €/MWh", "Emergency balancing (MW)", "Unmet demand (MW)",
        "Hourly planned cost (EUR)",
    ]
    numeric = renamed[columns].select_dtypes(include="number").columns
    renamed.loc[:, numeric] = renamed.loc[:, numeric].round(3)
    return renamed[columns]


def _plan_cost_rate(sim, column: str, fallback: float) -> pd.Series:
    schedule = sim.plan.schedule
    values = schedule[column] if column in schedule else pd.Series(fallback, index=schedule.index)
    return pd.to_numeric(values, errors="coerce").fillna(fallback)


def _active_plan_comparison_key(sim) -> tuple:
    validate_forecast_plan_alignment(sim)
    plan = sim.plan.schedule
    state = sim.plan.initial_state
    conventional_rate = _plan_cost_rate(
        sim, "conventional_cost_rate_eur_mwh", sim.assets.conventional.marginal_cost_eur_mwh
    )
    emergency_rate = _plan_cost_rate(
        sim, "emergency_cost_rate_eur_mwh", sim.assets.emergency_grid.marginal_cost_eur_mwh
    )
    hydro_limits = plan.get(
        "hydro_generation_limit_mw",
        pd.Series(sim.assets.hydro.capacity_mw, index=plan.index),
    )
    return (
        id(sim), getattr(sim, "model_family", ADAPTIVE_MLP), str(sim.timestamp),
        sim.active_model.model_version, sim.config.policy,
        sim.config.resource_scenario, sim.config.battery_charging_policy,
        getattr(sim.config, "allow_hydro_battery_charging", False),
        tuple(map(float, plan.demand_mw)),
        tuple(map(float, sim.resource_forecast["solar"])),
        tuple(map(float, sim.resource_forecast["wind"])),
        tuple(map(float, _plan_cost_rate(sim, "solar_cost_rate_eur_mwh", sim.assets.solar.marginal_cost_eur_mwh))),
        tuple(map(float, _plan_cost_rate(sim, "wind_cost_rate_eur_mwh", sim.assets.wind.marginal_cost_eur_mwh))),
        tuple(map(float, conventional_rate)), tuple(map(float, emergency_rate)),
        tuple(map(float, hydro_limits)),
        sim.config.terminal_soc_policy,
        state.battery_soc_mwh, state.hydro_remaining_mwh, state.conventional_output_mw,
    )


def _compare_active_plan_with_greedy(sim) -> dict[str, float]:
    """Compare the active Pyomo plan with myopic dispatch under matching inputs."""
    validate_forecast_plan_alignment(sim)
    if sim.config.terminal_soc_policy != "preserve_initial":
        raise ValueError("a common initial-SOC terminal target is not configured")
    planned = sim.plan.schedule
    assets = _comparison_assets(sim)
    conventional_rate = _plan_cost_rate(
        sim, "conventional_cost_rate_eur_mwh", assets.conventional.marginal_cost_eur_mwh
    )
    emergency_rate = _plan_cost_rate(
        sim, "emergency_cost_rate_eur_mwh", assets.emergency_grid.marginal_cost_eur_mwh
    )
    _schedule, greedy_operating_cost, greedy_soc_settlement = run_greedy_dispatch(
        planned.demand_mw.to_numpy(), sim.resource_forecast, assets, sim.plan.initial_state,
        terminal_soc_target_mwh=sim.plan.initial_state.battery_soc_mwh,
        timestamps=planned.index,
        battery_charging_policy=sim.config.battery_charging_policy,
        allow_hydro_battery_charging=getattr(sim.config, "allow_hydro_battery_charging", False),
        hydro_generation_limits_mw=planned.get(
            "hydro_generation_limit_mw",
            pd.Series(assets.hydro.capacity_mw, index=planned.index),
        ).to_numpy(),
        conventional_costs_eur_mwh=conventional_rate.to_numpy(),
        emergency_costs_eur_mwh=emergency_rate.to_numpy(),
        solar_costs_eur_mwh=_plan_cost_rate(sim, "solar_cost_rate_eur_mwh", assets.solar.marginal_cost_eur_mwh).to_numpy(),
        wind_costs_eur_mwh=_plan_cost_rate(sim, "wind_cost_rate_eur_mwh", assets.wind.marginal_cost_eur_mwh).to_numpy(),
        hydro_inflow_mw=planned.get("hydro_inflow_mw", pd.Series(0.0, index=planned.index)).to_numpy(),
    )
    greedy_total = float(greedy_operating_cost + greedy_soc_settlement)
    optimized_total = float(sim.plan.total_cost_eur)
    savings = greedy_total - optimized_total
    return {
        "greedy_total_eur": greedy_total,
        "multi_period_total_eur": optimized_total,
        "savings_eur": savings,
        "savings_percent": 100 * savings / max(greedy_total, 1e-9),
    }


def _comparison_assets(sim):
    """Use the same availability-adjusted renewable limits as the live plan."""
    adjust_assets = getattr(sim, "_assets_for_resource_availability", None)
    if callable(adjust_assets):
        return adjust_assets(sim.resource_forecast)
    return sim.assets


def _render_energy_planning(st, px, go, sim) -> None:
    st.subheader("Synchronized 24-hour forecast and dispatch plan")
    validate_forecast_plan_alignment(sim)
    hydro_toggle_key = f"hydro_battery_enabled_{id(sim)}"
    with st.form(key=f"hydro_battery_form_{id(sim)}"):
        hydro_enabled = st.toggle(
            "Allow hydro-to-battery charging",
            value=sim.config.allow_hydro_battery_charging,
            disabled=sim.config.battery_charging_policy != "renewable_only",
            key=hydro_toggle_key,
            help=("When enabled, hydro may charge the battery. Conventional and emergency sources "
                  "remain excluded from battery charging."),
        )
        if sim.config.battery_charging_policy != "renewable_only":
            st.caption("Select renewable-only battery charging in Replay controls to enable hydro charging.")
        apply_hydro_setting = st.form_submit_button("Apply and re-optimize 24-hour plan")
    if apply_hydro_setting and hydro_enabled != sim.config.allow_hydro_battery_charging:
        try:
            sim.set_hydro_battery_charging(hydro_enabled)
        except (RuntimeError, ValueError) as exc:
            st.error(f"Could not re-optimize with the selected hydro charging setting: {exc}")
        else:
            st.rerun()
    st.caption(
        "Hydro-to-battery charging is "
        f"{'enabled' if sim.config.allow_hydro_battery_charging else 'disabled'} for this plan."
    )
    st.caption(
        f"Plan targets {sim.plan.schedule.index[0]:%Y-%m-%d %H:%M} through "
        f"{sim.plan.schedule.index[-1]:%Y-%m-%d %H:%M UTC}; these are the same 24 timestamps as the forecast."
    )
    st.caption(
        f"Demand plan based on {FORECASTING_MODEL_LABELS.get(sim.model_family, sim.model_family)} "
        f"{sim.active_model.model_version}, issued {sim.timestamp:%Y-%m-%d %H:%M UTC}."
    )
    table = _dispatch_table(sim)
    st.caption("Compact cost-rate columns are in EUR/MWh; the final hourly cost uses the solver's complete objective, including emergency balancing and unmet-demand penalties.")
    st.download_button(
        "Download current 24-hour plan (CSV)", table.to_csv(index=False),
        file_name="griddrift_24h_plan.csv", mime="text/csv", key="dispatch_csv",
    )
    st.dataframe(table, width="stretch", hide_index=True)

    plan = sim.plan.schedule
    left, right = st.columns(2)
    with left:
        st.markdown("**Planned supply to demand (MW)**")
        contribution = pd.DataFrame({
            "Solar direct": plan.solar_to_demand_mw,
            "Wind direct": plan.wind_to_demand_mw,
            "Hydro direct": plan.hydro_to_demand_mw,
            "Conventional": plan.conventional_mw,
            "Battery discharge": plan.battery_discharge_mw,
            "Emergency": plan.emergency_mw,
            "Unmet": plan.unserved_mw,
        }, index=plan.index)
        st.plotly_chart(px.bar(contribution, barmode="stack", height=350), width="stretch", key="dispatch_supply_chart")
    with right:
        st.markdown("**Battery state-of-charge plan (MWh)**")
        battery_fig = go.Figure()
        battery_fig.add_trace(go.Scatter(
            x=plan.index, y=plan.soc_mwh, name="Battery SOC",
            line={"color": "#7C3AED", "width": 2.5},
        ))
        battery_fig.update_layout(
            height=270, yaxis_title="Battery SOC (MWh)",
            margin={"l": 10, "r": 10, "t": 20, "b": 10}, showlegend=False,
        )
        st.plotly_chart(battery_fig, width="stretch", key="dispatch_battery_soc_chart")

        st.markdown("**Hydro energy remaining (MWh)**")
        hydro_fig = go.Figure()
        hydro_fig.add_trace(go.Scatter(
            x=plan.index, y=plan.hydro_remaining_mwh, name="Hydro remaining",
            line={"color": "#D97706", "width": 2.5},
        ))
        hydro_fig.update_layout(
            height=270, yaxis_title="Hydro remaining (MWh)",
            margin={"l": 10, "r": 10, "t": 20, "b": 10}, showlegend=False,
        )
        st.plotly_chart(hydro_fig, width="stretch", key="dispatch_hydro_remaining_chart")


def _render_cost_analysis(st, px, sim) -> None:
    validate_forecast_plan_alignment(sim)
    planned = sim.plan.schedule
    realized_frame = pd.DataFrame(sim.realized)
    cost_columns = {
        "Solar": "cost_solar_eur", "Wind": "cost_wind_eur", "Hydro": "cost_hydro_eur",
        "Conventional": "cost_conventional_eur", "Battery degradation": "cost_battery_degradation_eur",
        "Emergency supply": "cost_emergency_eur", "Unserved-demand penalty": "cost_unserved_voll_eur",
    }
    source_costs = {label: float(planned[column].sum()) for label, column in cost_columns.items()}
    plan_total = float(planned.cost_total_eur.sum())
    if not abs(sum(source_costs.values()) - plan_total) <= max(1e-5, abs(plan_total) * 1e-7):
        st.error("The displayed source costs do not reconcile with the dispatch objective.")

    st.subheader("A. Overall 24-hour plan cost")
    st.caption(
        f"This cost breakdown uses the active 24-hour plan from "
        f"{FORECASTING_MODEL_LABELS.get(sim.model_family, sim.model_family)} "
        f"{sim.active_model.model_version}, issued {sim.timestamp:%Y-%m-%d %H:%M UTC}."
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total optimized dispatch cost", f"€{plan_total:,.2f}")
    metric_cols[1].metric("Average planned hourly cost", f"€{plan_total / max(len(planned), 1):,.2f}")
    metric_cols[2].metric("Lifetime realized cost", f"€{sim.accumulated_realized_cost_eur:,.2f}")
    cost_plot = pd.DataFrame({"Cost source": list(source_costs), "Cost (EUR)": list(source_costs.values())})
    st.plotly_chart(px.bar(
        cost_plot, x="Cost source", y="Cost (EUR)", height=330,
        color="Cost source", color_discrete_sequence=px.colors.qualitative.Safe,
    ), width="stretch", key="cost_breakdown_chart")
    if sim.assets.solar.marginal_cost_eur_mwh == 0 and sim.assets.wind.marginal_cost_eur_mwh == 0:
        st.caption("Solar and wind use zero marginal operating costs in this simulation. These are configurable assumptions, not real-world total generation costs.")
    else:
        st.caption("Solar and wind rates change in four-hour UTC blocks. They are configurable simulation assumptions, not ENTSO-E market prices or total generation costs.")
    st.caption(
        f"Hydro generation cost: {planned.cost_hydro_eur.sum():,.2f} EUR; battery degradation cost: "
        f"{planned.cost_battery_degradation_eur.sum():,.2f} EUR. Generation is charged once at its source, "
        "not again when stored energy is discharged."
    )

    st.subheader("B. Why was this allocation selected?")
    dt = sim.assets.interval_hours

    solar_rate = _plan_cost_rate(sim, "solar_cost_rate_eur_mwh", sim.assets.solar.marginal_cost_eur_mwh)
    wind_rate = _plan_cost_rate(sim, "wind_cost_rate_eur_mwh", sim.assets.wind.marginal_cost_eur_mwh)
    hydro_rate = _plan_cost_rate(sim, "hydro_cost_rate_eur_mwh", sim.assets.hydro.marginal_cost_eur_mwh)
    conv_rate = _plan_cost_rate(sim, "conventional_cost_rate_eur_mwh", sim.assets.conventional.marginal_cost_eur_mwh)
    emergency_rate = _plan_cost_rate(sim, "emergency_cost_rate_eur_mwh", sim.assets.emergency_grid.marginal_cost_eur_mwh)
    wear_rate = _plan_cost_rate(sim, "battery_degradation_rate_eur_mwh", sim.assets.battery.degradation_cost_eur_mwh)
    direct_renewable = float((planned.solar_to_demand_mw + planned.wind_to_demand_mw).sum()) * dt
    solar_dispatch = planned.solar_mw > 1e-6
    wind_dispatch = planned.wind_mw > 1e-6
    solar_wind_stored = float((planned.solar_to_battery_mw + planned.wind_to_battery_mw).sum()) * dt
    hydro_stored = float(planned.hydro_to_battery_mw.sum()) * dt
    total_stored = solar_wind_stored + hydro_stored
    discharged = float(planned.battery_discharge_mw.sum()) * dt
    hydro_used = float(planned.hydro_mw.sum()) * dt
    hydro_direct = float(planned.hydro_to_demand_mw.sum()) * dt
    hydro_left = float(planned.hydro_remaining_mwh.iloc[-1])
    conventional = float(planned.conventional_mw.sum()) * dt
    emergency = float(planned.emergency_mw.sum()) * dt
    unmet = float(planned.unserved_mw.sum()) * dt
    explanations = []
    if direct_renewable > 1e-6:
        solar_lower_cost_hours = int((solar_dispatch & (solar_rate < conv_rate)).sum())
        wind_lower_cost_hours = int((wind_dispatch & (wind_rate < conv_rate)).sum())
        explanations.append(
            f"Solar and wind supply {direct_renewable:.1f} MWh directly where available. Their configured rates "
            f"are below conventional's in {solar_lower_cost_hours}/{int(solar_dispatch.sum())} solar-use hours and "
            f"{wind_lower_cost_hours}/{int(wind_dispatch.sum())} wind-use hours, respectively."
        )
    if total_stored > 1e-6:
        charge_mask = planned.battery_charge_mw > 1e-6
        charge_hours = int(charge_mask.sum())
        charge_times = planned.index[charge_mask]
        charge_sources = []
        if solar_wind_stored > 1e-6:
            charge_sources.append(f"{solar_wind_stored:.1f} MWh from solar/wind")
        if hydro_stored > 1e-6:
            charge_sources.append(f"{hydro_stored:.1f} MWh from hydro")
        battery_explanation = (
            f"The optimizer scheduled {', '.join(charge_sources)} for storage over {charge_hours} hours "
            f"(first at {charge_times[0]:%Y-%m-%d %H:%M}, last at "
            f"{charge_times[-1]:%Y-%m-%d %H:%M} UTC), according to the source-specific flows selected by the optimizer."
        )
        if discharged > 1e-6:
            discharge_mask = planned.battery_discharge_mw > 1e-6
            higher_conventional_rate = int((discharge_mask & (conv_rate > wear_rate)).sum())
            discharge_times = planned.index[discharge_mask]
            battery_explanation += (
                f" It discharges {discharged:.1f} MWh over {int(discharge_mask.sum())} hours "
                f"(first at {discharge_times[0]:%Y-%m-%d %H:%M}, last at {discharge_times[-1]:%Y-%m-%d %H:%M} UTC)"
            )
            if higher_conventional_rate:
                battery_explanation += f"; conventional's configured rate exceeds battery wear in {higher_conventional_rate} discharge hours."
            else:
                battery_explanation += "; the schedule alone does not establish a cost saving from discharge."
        explanations.append(battery_explanation)
    if discharged > 1e-6 and total_stored <= 1e-6:
        explanations.append(f"The schedule uses {discharged:.1f} MWh of battery discharge from its starting energy; the plan alone does not establish a cost saving.")
    if hydro_used > 1e-6:
        lower_cost_hydro_hours = int(((planned.hydro_mw > 1e-6) & (hydro_rate < conv_rate)).sum())
        first_hydro = planned.index[planned.hydro_mw > 1e-6][0]
        explanations.append(
            f"Hydro generated {hydro_used:.1f} MWh: {hydro_direct:.1f} MWh went directly to demand and "
            f"{hydro_stored:.1f} MWh went to the battery. The reservoir plan moves from "
            f"{sim.plan.initial_state.hydro_remaining_mwh:.1f} to {hydro_left:.1f} MWh; "
            f"hydro generation cost is {float(planned.cost_hydro_eur.sum()):,.2f} EUR. "
            f"Dispatch begins at {first_hydro:%Y-%m-%d %H:%M} UTC; hydro's configured rate is below "
            f"conventional's in {lower_cost_hydro_hours} hydro-use hours."
        )
    elif sim.plan.initial_state.hydro_remaining_mwh > 1e-6:
        explanations.append(
            f"The optimizer scheduled no hydro generation and kept the reservoir at {hydro_left:.1f} MWh "
            "for this 24-hour plan."
        )
    if conventional > 1e-6:
        conv_hours = int((planned.conventional_mw > 1e-6).sum())
        explanations.append(
            f"Conventional generation supplies {conventional:.1f} MWh across {conv_hours} hours to complete the "
            f"balance after other scheduled flows; its configured rate is {conv_rate.min():.0f}–{conv_rate.max():.0f} EUR/MWh."
        )
    if emergency > 1e-6:
        explanations = explanations[:4]
        emergency_hours = int((planned.emergency_mw > 1e-6).sum())
        emergency_mask = planned.emergency_mw > 1e-6
        previous_conv = planned.conventional_mw.shift(1)
        previous_conv.iloc[0] = sim.plan.initial_state.conventional_output_mw
        binding = (
            (planned.conventional_mw >= sim.assets.conventional.capacity_mw - 1e-5)
            | ((planned.conventional_mw - previous_conv) >= sim.assets.conventional_ramp_up_mw_per_interval - 1e-5)
            | ((previous_conv - planned.conventional_mw) >= sim.assets.conventional_ramp_down_mw_per_interval - 1e-5)
        )
        binding_emergency_hours = int((emergency_mask & binding).sum())
        constraint_note = (
            f" Conventional capacity or ramp limits are active in {binding_emergency_hours}/{emergency_hours} emergency hours."
            if binding_emergency_hours else " The schedule does not show a conventional capacity or ramp limit binding in those hours."
        )
        emergency_text = (
            f"Emergency supply covers {emergency:.1f} MWh in {emergency_hours} hours when scheduled resources cannot "
            f"meet the balance; its configured rate is {emergency_rate.min():.0f}–{emergency_rate.max():.0f} EUR/MWh."
            f"{constraint_note}"
        )
        if unmet > 1e-6:
            emergency_text += f" A further {unmet:.1f} MWh remains unmet under the configured VOLL penalty."
        explanations.append(emergency_text)
    if unmet > 1e-6 and emergency <= 1e-6:
        explanations = explanations[:4]
        explanations.append(f"The plan leaves {unmet:.1f} MWh unmet, so the configured VOLL penalty applies.")
    explanations = explanations[:5]
    if explanations:
        for explanation in explanations:
            st.write(f"- {explanation}")
    else:
        st.write("No positive resource allocation was recorded in the current 24-hour solver output.")

    st.subheader("C. Greedy vs multi-period optimization")
    st.caption(
        "Comparison uses this exact forecast and renewable plan, the same assets, initial SOC/hydro/generator "
        f"state, ramp limits, {sim.config.battery_charging_policy} charging policy, hydro charging "
        f"{'enabled' if sim.config.allow_hydro_battery_charging else 'disabled'}, and per-hour conventional/emergency rates."
    )
    comparison_key = _active_plan_comparison_key(sim)
    if st.button("Compare this plan with hourly greedy dispatch", key="run_active_greedy_comparison"):
        st.session_state.active_greedy_comparison_key = comparison_key
        st.session_state.active_greedy_comparison = None
        try:
            st.session_state.active_greedy_comparison = _compare_active_plan_with_greedy(sim)
            st.session_state.pop("active_greedy_comparison_error", None)
        except (RuntimeError, ValueError) as exc:
            st.session_state.active_greedy_comparison_error = f"Comparison not available for this plan: {exc}"
    if st.session_state.get("active_greedy_comparison_key") == comparison_key:
        result = st.session_state.get("active_greedy_comparison")
        if result:
            comp_cols = st.columns(3)
            comp_cols[0].metric("Greedy total cost", f"€{result['greedy_total_eur']:,.2f}")
            comp_cols[1].metric("Multi-period cost", f"€{result['multi_period_total_eur']:,.2f}")
            comp_cols[2].metric("Savings", f"€{result['savings_eur']:,.2f}", f"{result['savings_percent']:.1f}%")
            st.plotly_chart(px.bar(
                pd.DataFrame({"Strategy": ["Greedy", "Multi-period"], "Cost (EUR)": [result["greedy_total_eur"], result["multi_period_total_eur"]]}),
                x="Strategy", y="Cost (EUR)", height=300,
            ), width="stretch", key="active_plan_greedy_chart")
            st.caption(
                f"Plan issued {sim.timestamp:%Y-%m-%d %H:%M} UTC; targets "
                f"{planned.index[0]:%Y-%m-%d %H:%M} to {planned.index[-1]:%Y-%m-%d %H:%M} UTC; "
                f"resource scenario: {sim.config.resource_scenario}."
            )
            st.caption("Greedy terminal SOC mismatch is valued at the mean conventional replacement rate; the multi-period plan ends at its configured initial-SOC target. Savings may be negative when greedy costs less.")
        elif st.session_state.get("active_greedy_comparison_error"):
            st.info(st.session_state.active_greedy_comparison_error)
    else:
        st.info("Comparison not available for this plan yet. Run the comparison when you want to evaluate the current 24-hour plan.")

    st.subheader("D. Planned vs realized operating cost")
    if realized_frame.empty:
        st.info("No replay hours have been executed yet, so planned and realized costs cannot be compared.")
    else:
        aligned = realized_frame.copy()
        aligned["timestamp"] = pd.to_datetime(aligned.timestamp, utc=True)
        aligned = aligned.sort_values("timestamp")
        aligned["planned_cost_eur"] = aligned.planned_first_hour_cost_eur
        aligned["planned_cumulative_eur"] = aligned.planned_cost_eur.cumsum()
        aligned["realized_cumulative_eur"] = aligned.realized_cost_eur.cumsum()
        delta = float(aligned.realized_cost_eur.sum() - aligned.planned_cost_eur.sum())
        cols = st.columns(3)
        cols[0].metric("Matched executed hours", len(aligned))
        cols[1].metric("Committed-plan cost", f"€{aligned.planned_cost_eur.sum():,.2f}")
        cols[2].metric("Cost difference (Actual − Planned)", f"€{delta:,.2f}")
        st.caption("Negative: actual operating cost was lower than planned. Positive: actual cost exceeded planned cost. Zero: the costs matched.")
        cumulative = aligned.rename(columns={
            "planned_cumulative_eur": "Planned cumulative cost",
            "realized_cumulative_eur": "Actual cumulative cost",
        })
        st.plotly_chart(px.line(
            cumulative, x="timestamp", y=["Planned cumulative cost", "Actual cumulative cost"],
            labels={"value": "Cumulative cost (EUR)", "timestamp": "Executed hour (UTC)", "variable": "Cost series"},
            height=330,
        ), width="stretch", key="planned_vs_realized_chart")
        st.caption("Each executed hour is compared with the first-hour cost from the plan committed immediately before that hour was revealed; later rolling plans do not replace it.")
        with st.expander("Technical Details — realized operations"):
            st.dataframe(aligned.tail(48), width="stretch", hide_index=True)
    st.info("We predicted demand for the next 24 hours, optimized how energy resources should be allocated, and evaluated why that allocation is more economical than planning each hour independently.")


def _legacy_main() -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="GridDRIFT", page_icon="⚡", layout="wide")
    defaults = load_integrated_settings(str(ROOT / "configs/integration.yaml"))

    with st.sidebar:
        st.header("Replay controls")
        data_path = st.text_input("Historical demand CSV", str(DEFAULT_DATA))
        model_path = st.text_input("Frozen MLP model folder", str(DEFAULT_MODEL))
        scale = st.number_input(
            "Representative-grid scale (dispatch only)", min_value=0.00001, max_value=1.0,
            value=defaults["demand_scale"], format="%.5f",
        )
        policy = st.selectbox(
            "Forecasting policy", ["static", "adaptive"],
            index=0 if defaults["policy"] == "static" else 1,
        )
        scenario = st.selectbox("Simulated renewable scenario", ["normal", "low_renewables", "wind_scarcity"])
        charging = st.selectbox(
            "Battery charging source", ["renewable_only", "unrestricted"],
            index=0 if defaults["battery_charging_policy"] == "renewable_only" else 1,
        )
        speed = st.slider(
            "Seconds per simulated hour", 0.5, 10.0,
            defaults["replay_speed_seconds_per_hour"], 0.5, key="replay_speed",
        )
        batch = int(st.number_input("Step-N hours", min_value=1, max_value=168, value=6))
        if st.button("Load / apply setup", type="primary", key="apply_setup"):
            st.session_state.simulation = None
            st.session_state.running = False
            st.session_state.has_started = False
            st.session_state.setup = (data_path, model_path, scale, policy, scenario, charging)

    setup = st.session_state.get("setup", (data_path, model_path, scale, policy, scenario, charging))
    if "simulation" not in st.session_state or st.session_state.simulation is None:
        try:
            demand = _read_demand(setup[0])
            model = DirectMLPForecaster.load(setup[1])
            assets, _ = load_dispatch_config(ROOT / "configs/assets.yaml")
            replay_config = ReplayConfig(adaptation_policy="validated")
            st.session_state.simulation = IntegratedSimulation(
                demand, model, assets,
                config=IntegratedConfig(
                    demand_scale=setup[2], policy=setup[3], resource_seed=defaults["resource_seed"],
                    resource_scenario=setup[4], battery_charging_policy=setup[5],
                    allow_hydro_battery_charging=defaults["allow_hydro_battery_charging"],
                    terminal_soc_policy=defaults["terminal_soc_policy"], adaptation=replay_config,
                ),
            )
            st.session_state.running = False
            st.session_state.has_started = False
            st.session_state.next_tick = 0.0
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            st.error(f"Could not initialize replay: {exc}")
            st.stop()

    sim = st.session_state.simulation
    running = st.session_state.get("running", False)
    if st.sidebar.button("Start", disabled=running or st.session_state.get("has_started", False) or sim.status == "complete"):
        st.session_state.running = True
        st.session_state.has_started = True
        st.session_state.next_tick = time.monotonic()
    if st.sidebar.button("Pause", disabled=not running):
        st.session_state.running = False
    if st.sidebar.button("Resume", disabled=running or not st.session_state.get("has_started", False) or sim.status == "complete"):
        st.session_state.running = True
        st.session_state.next_tick = time.monotonic()
    step_col, step_n_col = st.sidebar.columns(2)
    if step_col.button("Step", disabled=sim.status == "complete"):
        st.session_state.running = False
        st.session_state.has_started = True
        sim.step()
    if step_n_col.button(f"Step {batch}", disabled=sim.status == "complete"):
        st.session_state.running = False
        st.session_state.has_started = True
        sim.step_n(batch)
    if st.sidebar.button("Reset"):
        st.session_state.running = False
        st.session_state.has_started = False
        sim.reset()

    @st.fragment(run_every=speed if st.session_state.get("running", False) else None)
    def dashboard_panel() -> None:
        current = st.session_state.simulation
        if st.session_state.get("running") and time.monotonic() >= st.session_state.get("next_tick", 0):
            current.step()
            st.session_state.next_tick = time.monotonic() + speed
            if current.status == "complete":
                st.session_state.running = False

        status = "Running" if st.session_state.get("running") else current.status.title()
        st.title("GridDRIFT")
        header = st.columns([1.4, 1, 1, 1, 1])
        header[0].metric("Simulation time (UTC)", f"{current.timestamp:%d %b %Y, %H:%M}")
        header[1].metric("Replay status", status)
        header[2].metric("Battery SOC", f"{current.state.battery_soc_mwh:.1f} MWh")
        header[3].metric("Hydro remaining", f"{current.state.hydro_remaining_mwh:.1f} MWh")
        header[4].metric(
            "Today's planned cost",
            f"€{daily_planned_cost_eur(current):,.2f}",
            help=("Cost from the current forecast-based dispatch plan for target hours "
                  "on this UTC date; it excludes realized costs and other dates."),
        )

        live, drift, planning, costs = st.tabs([
            "Live Forecasting", "Drift Monitoring", "24-Hour Energy Planning", "Cost & Analysis",
        ])
        with live:
            _render_live_forecasting(st, current)
        with drift:
            _render_drift_monitoring(st, px, go, current)
        with planning:
            _render_energy_planning(st, px, go, current)
        with costs:
            _render_cost_analysis(st, px, current)

    dashboard_panel()


def main() -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="GridDRIFT", page_icon="⚡", layout="wide")
    defaults = load_integrated_settings(str(ROOT / "configs/integration.yaml"))
    existing_simulation = st.session_state.get("simulation")
    if existing_simulation is not None:
        _sync_hydro_charging_default(
            st.session_state, existing_simulation, defaults["allow_hydro_battery_charging"]
        )
    selected_model = initialize_forecasting_model(
        st.session_state,
        has_existing_mlp_session=existing_simulation is not None,
    )
    with st.sidebar:
        st.header("Forecasting model")
        selected_model = st.selectbox(
            "Model",
            options=[ADAPTIVE_XGBOOST, ADAPTIVE_MLP],
            format_func=lambda family: FORECASTING_MODEL_LABELS[family]
            + (" (default)" if family == ADAPTIVE_XGBOOST else ""),
            key="selected_forecasting_model",
            on_change=_on_forecasting_model_change,
        )
        selected_version = model_checkpoint_version(selected_model, ROOT)
        st.caption(f"Selected checkpoint version: {selected_version}")

    default_setup = (
        str(DEFAULT_DATA), str(DEFAULT_MODEL), defaults["demand_scale"], "adaptive",
        defaults["resource_scenario"], defaults["battery_charging_policy"],
        DEFAULT_REPLAY_START if DEFAULT_DATA.exists() else None, selected_model,
        defaults.get("renewable_data_mode", SIMULATED_RESOURCE_MODE),
    )
    setup = st.session_state.get("setup", default_setup)
    if len(setup) == 6:
        setup = (*setup, None)
    if len(setup) == 7:
        legacy_family = ADAPTIVE_MLP if existing_simulation is not None else selected_model
        setup = (*setup, legacy_family)
    if len(setup) == 8:
        setup = (*setup, SIMULATED_RESOURCE_MODE)
    migrate_static_replay = (
        existing_simulation is not None
        and getattr(existing_simulation.config, "policy", "static") != "adaptive"
    )
    if setup[3] != "adaptive":
        setup = (*setup[:3], "adaptive", *setup[4:])
        if migrate_static_replay:
            st.session_state.simulation = None
            st.session_state.running = False
            st.session_state.has_started = False
            existing_simulation = None
    # Upgrade sessions that were initialized before the extended ENTSO-E range
    # was downloaded, so the combined actuals are selected automatically.
    if setup[0] == str(LEGACY_DATA) and DEFAULT_DATA.exists():
        setup = (str(DEFAULT_DATA), *setup[1:6], DEFAULT_REPLAY_START, *setup[7:])
        st.session_state.simulation = None
        st.session_state.has_started = False
    st.session_state.setup = setup

    if "simulation" not in st.session_state or st.session_state.simulation is None:
        setup = (*setup[:7], selected_model, setup[8])
        st.session_state.setup = setup
        try:
            st.session_state.simulation = _build_simulation(setup, defaults)
            _sync_hydro_charging_default(
                st.session_state, st.session_state.simulation,
                defaults["allow_hydro_battery_charging"],
            )
            st.session_state.active_forecasting_model = selected_model
            st.session_state.model_switch_pending = False
            if migrate_static_replay:
                st.sidebar.info(
                    "The previous static MLP replay was reset to Adaptive MLP at its selected start time."
                )
            if setup[6] is None:
                setup = (
                    *setup[:6], st.session_state.simulation.timestamp,
                    selected_model, setup[8],
                )
                st.session_state.setup = setup
            st.session_state.running = False
            st.session_state.has_started = False
            st.session_state.next_tick = 0.0
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            st.error(f"Could not initialize replay: {exc}")
            st.stop()

    sim = st.session_state.simulation
    active_model = st.session_state.get("active_forecasting_model", ADAPTIVE_MLP)
    selected_model_pending = model_switch_pending(selected_model, active_model)
    if st.session_state.setup[6] is None:
        st.session_state.setup = (
            *tuple(st.session_state.setup[:6]), sim.timestamp, active_model,
            st.session_state.setup[8],
        )
    minimum_start = sim.demand.index[sim.minimum_replay_position]
    final_actual = sim.demand.index[-1]
    default_start = pd.Timestamp(setup[6]) if setup[6] is not None else sim.timestamp
    if default_start.tz is None:
        default_start = default_start.tz_localize("UTC")
    else:
        default_start = default_start.tz_convert("UTC")
    with st.sidebar:
        st.header("Replay controls")
        data_path = st.text_input("Historical demand CSV", setup[0])
        model_path = st.text_input("Frozen MLP model folder", setup[1])
        resource_mode = st.selectbox(
            "Resource data",
            options=[SIMULATED_RESOURCE_MODE, HISTORICAL_RESOURCE_MODE],
            index=[SIMULATED_RESOURCE_MODE, HISTORICAL_RESOURCE_MODE].index(setup[8]),
            format_func=lambda mode: {
                SIMULATED_RESOURCE_MODE: "Simulated solar/wind + configured hydro",
                HISTORICAL_RESOURCE_MODE: "ENTSO-E 2025 renewables + synthetic PECD hydro",
            }[mode],
        )
        try:
            candidate_first, candidate_last = _read_demand_bounds(data_path)
        except (OSError, ValueError, KeyError) as exc:
            candidate_first, candidate_last = sim.demand.index[0], final_actual
            st.warning(f"Cannot inspect selected demand timestamps yet: {exc}")
        if resource_mode == HISTORICAL_RESOURCE_MODE:
            if not DEFAULT_RENEWABLE_DATA.exists():
                st.error(f"Historical renewable input not found: {DEFAULT_RENEWABLE_DATA}")
                st.stop()
            resource_times = pd.to_datetime(
                pd.read_csv(DEFAULT_RENEWABLE_DATA, usecols=["target_timestamp_utc"])
                ["target_timestamp_utc"],
                utc=True,
            )
            candidate_first = max(candidate_first, resource_times.min())
            candidate_last = min(
                candidate_last, resource_times.max() - pd.Timedelta(hours=24)
            )
        valid_first = max(minimum_start, candidate_first)
        if valid_first > candidate_last:
            st.error("The selected demand data does not overlap the model's eligible replay period.")
            st.stop()
        start_date_default = min(max(default_start.date(), valid_first.date()), candidate_last.date())
        if "replay_start_date" not in st.session_state:
            st.session_state.replay_start_date = start_date_default
        else:
            st.session_state.replay_start_date = min(
                max(st.session_state.replay_start_date, valid_first.date()), candidate_last.date()
            )
        scale = st.number_input(
            "Representative-grid scale (dispatch only)", min_value=0.00001, max_value=1.0,
            value=float(setup[2]), format="%.5f",
        )
        policy = "adaptive"
        if resource_mode == SIMULATED_RESOURCE_MODE:
            scenario = st.selectbox(
                "Simulated renewable scenario", ["normal", "low_renewables", "wind_scarcity"],
                index=["normal", "low_renewables", "wind_scarcity"].index(setup[4]),
            )
        else:
            scenario = "normal"
            st.caption(
                "Historical-resource replay uses 2025 day-ahead forecasts and revealed actuals. "
                "Forecast publication time is assumed to be 00:00 UTC the previous day; "
                "PECD hydro inflow is a modeled seasonal profile. Actual renewable generation "
                "is revealed hourly and used as a conservative dispatch supply proxy; this source "
                "does not provide curtailment-adjusted physical renewable potential."
            )
        charging_options = ["renewable_only", "unrestricted"]
        charging = st.selectbox(
            "Battery charging source", charging_options, index=charging_options.index(setup[5]),
        )
        start_date = st.date_input(
            "Replay start date (UTC)", min_value=valid_first.date(), max_value=candidate_last.date(),
            key="replay_start_date",
        )
        st.caption(
            f"Selected file has actuals through {candidate_last:%Y-%m-%d %H:%M UTC}. "
            "To replay from 2025-01-01, select a CSV that includes actual observations on and after that date."
        )
        speed = st.slider(
            "Seconds per simulated hour", 0.5, 10.0,
            defaults["replay_speed_seconds_per_hour"], 0.5, key="replay_speed",
        )
        apply_setup = st.button(
            "Load / apply setup",
            type="primary",
            key="apply_setup",
        )
        if selected_model == ADAPTIVE_XGBOOST:
            st.caption(f"Live forecasts use the saved XGBoost checkpoint {selected_version}.")

    selected_start = pd.Timestamp(start_date).tz_localize("UTC")
    if apply_setup:
        selected_setup = (
            data_path, model_path, scale, policy, scenario, charging, selected_start, selected_model,
            resource_mode,
        )
        try:
            st.session_state.simulation = _build_simulation(selected_setup, defaults)
            _sync_hydro_charging_default(
                st.session_state, st.session_state.simulation,
                defaults["allow_hydro_battery_charging"],
            )
            st.session_state.setup = selected_setup
            st.session_state.active_forecasting_model = selected_model
            st.session_state.model_switch_pending = False
            st.session_state.running = False
            st.session_state.has_started = False
            st.session_state.next_tick = 0.0
            sim = st.session_state.simulation
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            st.error(f"Could not apply replay setup: {exc}")

    sim = st.session_state.simulation
    loaded_start = pd.Timestamp(st.session_state.setup[6])
    setup_pending = (
        data_path != st.session_state.setup[0]
        or model_path != st.session_state.setup[1]
        or scale != st.session_state.setup[2]
        or policy != st.session_state.setup[3]
        or scenario != st.session_state.setup[4]
        or charging != st.session_state.setup[5]
        or resource_mode != st.session_state.setup[8]
        or selected_start.date() != loaded_start.date()
        or selected_model_pending
    )
    if setup_pending:
        if selected_model_pending:
            st.sidebar.warning(
                f"Selected {FORECASTING_MODEL_LABELS[selected_model]}, but the visible replay is still "
                f"{FORECASTING_MODEL_LABELS[active_model]} ({sim.active_model.model_version}). "
                "Playback is paused to prevent mixing model forecasts and dispatch. Applying this setup "
                "restarts the replay and resets its model-specific drift and physical dispatch state."
            )
        else:
            st.sidebar.info("Apply the selected replay setup before stepping or starting playback.")
    running = st.session_state.get("running", False)

    @st.fragment(run_every=speed if st.session_state.get("running", False) else None)
    def dashboard_panel() -> None:
        current = st.session_state.simulation
        if st.session_state.get("running") and time.monotonic() >= st.session_state.get("next_tick", 0):
            current.step()
            st.session_state.next_tick = time.monotonic() + speed
            if current.status == "complete":
                st.session_state.running = False

        status = "Running" if st.session_state.get("running") else current.status.title()
        st.title("GridDRIFT")
        header = st.columns([1.4, 1, 1, 1, 1])
        header[0].metric("Simulation time (UTC)", f"{current.timestamp:%d %b %Y, %H:%M}")
        header[1].metric("Replay status", status)
        header[2].metric("Battery SOC", f"{current.state.battery_soc_mwh:.1f} MWh")
        header[3].metric("Hydro remaining", f"{current.state.hydro_remaining_mwh:.1f} MWh")
        header[4].metric(
            "Today's planned cost",
            f"€{daily_planned_cost_eur(current):,.2f}",
            help=("Cost from the current forecast-based dispatch plan for target hours "
                  "on this UTC date; it excludes realized costs and other dates."),
        )
        st.caption(f"Resource inputs: {getattr(current, 'resource_data_label', 'Simulated renewable resources')}")
        if getattr(current, "resource_data_mode", SIMULATED_RESOURCE_MODE) == HISTORICAL_RESOURCE_MODE:
            st.caption(
                "ENTSO-E day-ahead market prices are not used in the dispatch objective; "
                "configured scenario generation costs remain active."
            )
        if model_switch_pending(selected_model, active_model):
            st.warning(
                f"Selected {FORECASTING_MODEL_LABELS[selected_model]} "
                f"({selected_version}), while the visible forecast, drift history, and dispatch "
                f"still belong to {FORECASTING_MODEL_LABELS[active_model]} "
                f"({current.active_model.model_version}). Replay controls are disabled until the "
                "selected model is connected and its plan is rebuilt."
            )

        live, drift, planning, costs = st.tabs([
            "Live Forecasting", "Drift Monitoring", "24-Hour Energy Planning", "Cost & Analysis",
        ])
        with live:
            st.subheader("Replay controls")
            batch = int(st.number_input("Step-N hours", min_value=1, max_value=168, value=6, key="live_step_n_hours"))
            controls = st.columns(6)
            if controls[0].button(
                "Start", disabled=setup_pending or running or st.session_state.get("has_started", False)
                or current.status == "complete", key="live_start",
            ):
                st.session_state.running = True
                st.session_state.has_started = True
                st.session_state.next_tick = time.monotonic()
                st.rerun()
            if controls[1].button("Pause", disabled=not running, key="live_pause"):
                st.session_state.running = False
                st.rerun()
            if controls[2].button(
                "Resume", disabled=setup_pending or running
                or not st.session_state.get("has_started", False) or current.status == "complete",
                key="live_resume",
            ):
                st.session_state.running = True
                st.session_state.next_tick = time.monotonic()
                st.rerun()
            if controls[3].button(
                "Step", disabled=setup_pending or current.status == "complete", key="live_step",
            ):
                st.session_state.running = False
                st.session_state.has_started = True
                current.step()
                st.rerun()
            if controls[4].button(
                f"Step {batch}", disabled=setup_pending or current.status == "complete", key="live_step_n",
            ):
                st.session_state.running = False
                st.session_state.has_started = True
                current.step_n(batch)
                st.rerun()
            if controls[5].button("Reset", disabled=setup_pending, key="live_reset"):
                st.session_state.running = False
                st.session_state.has_started = False
                current.reset(start_timestamp=selected_start)
                st.session_state.setup = (
                    *tuple(st.session_state.setup[:6]), selected_start,
                    st.session_state.setup[7], st.session_state.setup[8],
                )
                st.rerun()
            _render_live_forecasting(st, current)
        with drift:
            _render_drift_monitoring(st, px, go, current)
        with planning:
            _render_energy_planning(st, px, go, current)
        with costs:
            _render_cost_analysis(st, px, current)

    dashboard_panel()


if __name__ == "__main__":
    main()
