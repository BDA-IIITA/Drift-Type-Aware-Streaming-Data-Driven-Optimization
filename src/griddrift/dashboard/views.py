"""Pure dataframe helpers shared by the dashboard and offline smoke tests."""

from __future__ import annotations

import pandas as pd


def validate_forecast_plan_alignment(simulation) -> None:
    """Fail clearly if the forecast and dispatch plan no longer share target hours."""
    forecast = getattr(simulation, "forecast", None)
    if forecast is None:
        return
    forecast_index = pd.DatetimeIndex(forecast.index)
    plan_index = pd.DatetimeIndex(simulation.plan.schedule.index)
    if len(forecast_index) != len(plan_index) or not forecast_index.equals(plan_index):
        raise ValueError("selected model forecast and dispatch plan must share identical target timestamps")


def demand_chart_frame(simulation) -> pd.DataFrame:
    """Combine observed-only demand history and the complete future forecast."""
    history = simulation.history_frame().rename(columns={"demand_mw": "actual_demand_mw"})
    forecast = simulation.forecast_frame()
    validate_forecast_plan_alignment(simulation)
    return history.join(forecast, how="outer")


def streaming_forecast_frame(simulation, window_hours: int = 24 * 14) -> pd.DataFrame:
    """Return a readable rolling window of observed history plus all 24 forecasts."""
    if window_hours < 1:
        raise ValueError("window_hours must be positive")
    frame = demand_chart_frame(simulation)
    start = simulation.timestamp - pd.Timedelta(hours=window_hours - 1)
    end = simulation.forecast.index[-1]
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def dispatch_download_frame(simulation) -> pd.DataFrame:
    """Return the complete 24-hour schedule with timestamp as an explicit column."""
    validate_forecast_plan_alignment(simulation)
    schedule = simulation.plan.schedule.copy()
    schedule.index.name = "timestamp"
    return schedule.reset_index()


def daily_planned_cost_eur(simulation) -> float:
    """Return the current plan's cost for forecast targets on today's UTC date."""
    validate_forecast_plan_alignment(simulation)
    day = pd.Timestamp(simulation.timestamp).tz_convert("UTC").date()
    schedule = simulation.plan.schedule
    plan_timestamps = pd.DatetimeIndex(schedule.index).tz_convert("UTC")
    return float(schedule.loc[plan_timestamps.date == day, "cost_total_eur"].sum())


def dashboard_snapshot_figure(simulation):
    """Build an offline HTML-ready snapshot of the synchronized forecast/plan."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                        subplot_titles=("Observed demand and full 24-hour forecast",
                                        "Planned electricity supplied to demand",
                                        "Planned battery SOC and hydro budget"),
                        vertical_spacing=0.12)
    history = simulation.history_frame()
    fig.add_trace(go.Scatter(x=history.index, y=history.demand_mw, name="Actual demand (observed)",
                             line={"color": "#16824B", "width": 2.5}), row=1, col=1)
    future = simulation.forecast
    fig.add_trace(go.Scatter(x=future.index, y=future, name="24-hour forecast",
                             line={"color": "#2878D0", "width": 2.5, "dash": "dash"}), row=1, col=1)
    fig.add_vline(x=simulation.timestamp, line_color="#6B7280", line_dash="dot", row=1, col=1)
    plan = simulation.plan.schedule
    hydro_direct = (
        plan["hydro_to_demand_mw"]
        if "hydro_to_demand_mw" in plan
        else plan["hydro_mw"]
    )
    for column, label, color in (
        ("solar_to_demand_mw", "Solar direct", "#F4C542"),
        ("wind_to_demand_mw", "Wind direct", "#38A3A5"),
        ("__hydro_direct__", "Hydro direct", "#4F86C6"),
        ("conventional_mw", "Conventional", "#6B7280"),
        ("battery_discharge_mw", "Battery discharge", "#8B5CF6"),
        ("emergency_mw", "Emergency", "#E76F51"),
        ("unserved_mw", "Unserved", "#A4161A"),
    ):
        series = hydro_direct if column == "__hydro_direct__" else plan[column]
        fig.add_trace(go.Bar(x=plan.index, y=series, name=label, marker_color=color), row=2, col=1)
    fig.update_layout(barmode="stack")
    fig.add_trace(go.Scatter(x=plan.index, y=plan.soc_mwh, name="Battery SOC MWh",
                             line={"color": "#7C3AED"}), row=3, col=1)
    fig.add_trace(go.Scatter(x=plan.index, y=plan.hydro_remaining_mwh, name="Hydro remaining MWh",
                             line={"color": "#D97706", "dash": "dot"}), row=3, col=1)
    fig.update_layout(height=1000, title="GridDRIFT integrated replay snapshot",
                      legend={"orientation": "h"}, margin={"l": 40, "r": 40, "t": 90, "b": 40})
    fig.update_yaxes(title_text="Demand MW (original scale)", row=1, col=1)
    fig.update_yaxes(title_text="Power MW", row=2, col=1)
    fig.update_yaxes(title_text="Energy MWh", row=3, col=1)
    return fig
