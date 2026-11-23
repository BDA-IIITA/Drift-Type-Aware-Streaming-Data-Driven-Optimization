"""Validate saved Phase 4.7 Stage 4 hourly scenario schedules."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/experiments/phase47/stage4")
    folder = Path(parser.parse_args().input)
    checks = []
    for path in sorted(folder.glob("*_hydro_charge_*.csv")):
        frame = pd.read_csv(path)
        delivered = frame[[
            "solar_to_demand_mw", "wind_to_demand_mw", "hydro_to_demand_mw",
            "conventional_mw", "battery_discharge_mw", "emergency_mw", "unserved_mw",
        ]].sum(axis=1)
        balance_error = (delivered - frame.demand_mw - frame.battery_charge_mw).abs()
        solar_error = (
            frame.solar_available_mw - frame.solar_mw - frame.solar_curtailment_mw
        ).abs()
        wind_error = (
            frame.wind_available_mw - frame.wind_mw - frame.wind_curtailment_mw
        ).abs()
        source_error = (
            frame.hydro_mw - frame.hydro_to_demand_mw - frame.hydro_to_battery_mw
        ).abs()
        cost_columns = [column for column in frame if column.startswith("cost_") and column != "cost_total_eur"]
        cost_error = (frame.cost_total_eur - frame[cost_columns].sum(axis=1)).abs()
        result = {
            "schedule_file": path.name,
            "hours": len(frame),
            "max_demand_balance_error_mw": float(balance_error.max()),
            "max_solar_allocation_error_mw": float(solar_error.max()),
            "max_wind_allocation_error_mw": float(wind_error.max()),
            "max_hydro_split_error_mw": float(source_error.max()),
            "max_hourly_cost_reconciliation_error_eur": float(cost_error.max()),
        }
        result["valid"] = bool(
            len(frame) == 24
            and result["max_demand_balance_error_mw"] < 1e-5
            and result["max_solar_allocation_error_mw"] < 1e-5
            and result["max_wind_allocation_error_mw"] < 1e-5
            and result["max_hydro_split_error_mw"] < 1e-5
            and result["max_hourly_cost_reconciliation_error_eur"] < 1e-5
        )
        checks.append(result)

    if len(checks) != 4 or not all(row["valid"] for row in checks):
        raise RuntimeError(f"Expected four valid 24-hour schedules; checks={checks}")
    payload = {"schedule_count": len(checks), "all_valid": True, "checks": checks}
    (folder / "validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
