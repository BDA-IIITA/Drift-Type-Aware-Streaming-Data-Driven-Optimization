"""Smoke-render the dashboard using its configured real-resource defaults."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    app = AppTest.from_file(str(ROOT / "scripts/streamlit_app.py"), default_timeout=120)
    app.run()
    exceptions = [str(item.value) for item in app.exception]
    if exceptions:
        raise RuntimeError("Streamlit app rendered with exceptions: " + " | ".join(exceptions))
    simulation = app.session_state["simulation"]
    setup = app.session_state["setup"]
    result = {
        "verification": "passed",
        "initial_render_exceptions": len(exceptions),
        "configured_resource_mode": setup[8],
        "selected_forecasting_model": app.session_state["selected_forecasting_model"],
        "active_forecasting_model": app.session_state["active_forecasting_model"],
        "hydro_to_battery_enabled_by_default": simulation.config.allow_hydro_battery_charging,
        "simulation_resource_mode": simulation.resource_data_mode,
        "resource_label": simulation.resource_data_label,
        "simulation_timestamp_utc": simulation.timestamp.isoformat(),
        "forecast_target_count": len(simulation.forecast),
        "dispatch_plan_target_count": len(simulation.plan.schedule),
        "forecast_plan_timestamps_match": simulation.forecast.index.equals(simulation.plan.schedule.index),
        "hydro_inflow_schedule_present": "hydro_inflow_mw" in simulation.plan.schedule,
        "hydro_to_battery_schedule_present": "hydro_to_battery_mw" in simulation.plan.schedule,
        "tabs": ["Live Forecasting", "Drift Monitoring", "24-Hour Energy Planning", "Cost & Analysis"],
        "note": "Rendered with Streamlit AppTest; no screenshot capture is available in AppTest.",
    }
    if result["configured_resource_mode"] != "entsoe_pecd":
        raise AssertionError("Dashboard default is not the configured historical resource mode")
    if result["selected_forecasting_model"] != "xgboost":
        raise AssertionError("Dashboard default forecasting model is not adaptive XGBoost")
    if not result["hydro_to_battery_enabled_by_default"]:
        raise AssertionError("New dashboard session did not enable hydro-to-battery charging by default")
    if not result["forecast_plan_timestamps_match"] or result["forecast_target_count"] != 24:
        raise AssertionError("Dashboard's forecast and Pyomo plan are not a synchronized 24-hour horizon")
    destination = ROOT / "artifacts/experiments/phase55/final_integration/dashboard_verification.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
