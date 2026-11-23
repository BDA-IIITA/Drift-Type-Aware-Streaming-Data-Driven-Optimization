"""Analyze saved Phase 4.5 Stage 4 dispatch outputs without running solves."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experiments/phase45"
OUT = BASE / "analysis/stage5"
RUNS = [
    ("spring-2025w14", BASE / "runs/phase45-spring-2025w14-normal-seed42"),
    ("summer-2025w27", BASE / "runs/phase45-summer-2025w27-normal-seed42"),
    ("autumn-2025w40", BASE / "runs/phase45-autumn-2025w40-normal-seed42"),
]
DEMAND_SCALE = 0.001
INTERVAL_HOURS = 1.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without the optional tabulate dependency."""
    cols = [str(c) for c in frame.columns]
    values = [["" if pd.isna(v) else str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(cols[i]), *(len(row[i]) for row in values)) for i in range(len(cols))]
    line = "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in values]
    return "\n".join([line, sep, *body])


def forecast_metrics(plans: pd.DataFrame, actuals: pd.DataFrame, window: str) -> list[dict]:
    actual = actuals[["target_timestamp", "actual_demand_mw"]].drop_duplicates("target_timestamp")
    if actual["target_timestamp"].duplicated().any():
        raise ValueError("Actual target timestamps are not unique")
    scored = plans.merge(actual, on="target_timestamp", how="inner", validate="many_to_one")
    scored["predicted_demand_mw"] = scored["demand_mw"] / DEMAND_SCALE
    scored["absolute_error_mw"] = (scored["predicted_demand_mw"] - scored["actual_demand_mw"]).abs()
    scored["squared_error_mw2"] = (scored["predicted_demand_mw"] - scored["actual_demand_mw"]) ** 2
    output = []
    for (model, horizon), group in scored.groupby(["model", "horizon_hours"], sort=True):
        errors = group["predicted_demand_mw"] - group["actual_demand_mw"]
        output.append({
            "window": window, "model": model,
            "evaluation": "fixed_24h" if int(horizon) == 24 else f"lead_{int(horizon)}h",
            "horizon_hours": int(horizon),
            "matched_predictions": int(len(group)),
            "unique_actual_targets": int(group["target_timestamp"].nunique()),
            "target_start_utc": group["target_timestamp"].min().isoformat(),
            "target_end_utc": group["target_timestamp"].max().isoformat(),
            "mae_mw": float(errors.abs().mean()), "rmse_mw": rmse(errors),
            "mape_pct": float((errors.abs() / group["actual_demand_mw"].abs().replace(0, np.nan)).mean() * 100),
        })
    # Pool all leads per model, separate from the fixed-24h metric.
    for model, group in scored.groupby("model", sort=True):
        errors = group["predicted_demand_mw"] - group["actual_demand_mw"]
        output.append({
            "window": window, "model": model, "evaluation": "all_leads",
            "horizon_hours": "1-24", "matched_predictions": int(len(group)),
            "unique_actual_targets": int(group["target_timestamp"].nunique()),
            "target_start_utc": group["target_timestamp"].min().isoformat(),
            "target_end_utc": group["target_timestamp"].max().isoformat(),
            "mae_mw": float(errors.abs().mean()), "rmse_mw": rmse(errors),
            "mape_pct": float((errors.abs() / group["actual_demand_mw"].abs().replace(0, np.nan)).mean() * 100),
        })
    return output


def peak_metrics(plans: pd.DataFrame, actuals: pd.DataFrame, window: str) -> list[dict]:
    fixed = plans.loc[plans["horizon_hours"] == 24].merge(
        actuals[["target_timestamp", "actual_demand_mw"]].drop_duplicates("target_timestamp"),
        on="target_timestamp", how="inner", validate="many_to_one")
    fixed["predicted_demand_mw"] = fixed["demand_mw"] / DEMAND_SCALE
    fixed["target_day"] = fixed["target_timestamp"].dt.floor("D")
    rows = []
    for model, model_data in fixed.groupby("model"):
        days = []
        for day, group in model_data.groupby("target_day"):
            if len(group) != 24:
                continue
            pred_idx = group["predicted_demand_mw"].to_numpy().argmax()
            actual_idx = group["actual_demand_mw"].to_numpy().argmax()
            days.append({"day": day, "predicted_peak_mw": group["predicted_demand_mw"].max(),
                         "actual_peak_mw": group["actual_demand_mw"].max(),
                         "peak_magnitude_abs_error_mw": abs(group["predicted_demand_mw"].max() - group["actual_demand_mw"].max()),
                         "peak_timing_abs_error_hours": abs(pred_idx - actual_idx)})
        d = pd.DataFrame(days)
        rows.append({"window": window, "model": model, "complete_target_days": len(d),
                     "mean_peak_magnitude_abs_error_mw": float(d["peak_magnitude_abs_error_mw"].mean()) if len(d) else None,
                     "mean_peak_timing_abs_error_hours": float(d["peak_timing_abs_error_hours"].mean()) if len(d) else None})
    return rows


def dispatch_metrics(execution: pd.DataFrame, window: str) -> list[dict]:
    rows = []
    for model, g in execution.groupby("model", sort=True):
        served = ((g["recourse_demand_mw"] - g["recourse_unserved_mw"]).clip(lower=0) * INTERVAL_HOURS).sum()
        rows.append({
            "window": window, "model": model, "executed_hours": int(len(g)),
            "solver_failures": int((g["termination_condition"].str.lower() != "optimal").sum()),
            "realized_recourse_cost_eur": float(g["realized_cost_eur"].sum()),
            "cost_per_served_mwh_eur": float(g["realized_cost_eur"].sum() / served) if served else None,
            "served_energy_mwh": float(served),
            "emergency_energy_mwh": float(g["recourse_emergency_mw"].sum() * INTERVAL_HOURS),
            "unserved_energy_mwh": float(g["recourse_unserved_mw"].sum() * INTERVAL_HOURS),
            "battery_degradation_cost_eur": float(g["realized_cost_battery_degradation_eur"].sum()),
            "battery_throughput_mwh": float(((g["recourse_battery_charge_mw"] + g["recourse_battery_discharge_mw"]).sum()) * INTERVAL_HOURS),
            "solar_curtailment_mwh": float(g["recourse_solar_curtailment_mw"].sum() * INTERVAL_HOURS),
            "wind_curtailment_mwh": float(g["recourse_wind_curtailment_mw"].sum() * INTERVAL_HOURS),
            "hydro_used_mwh": float(g["recourse_hydro_mw"].sum() * INTERVAL_HOURS),
            "terminal_battery_soc_mwh": float(g["ending_battery_soc_mwh"].iloc[-1]),
            "terminal_hydro_remaining_mwh": float(g["ending_hydro_remaining_mwh"].iloc[-1]),
            "terminal_conventional_output_mw": float(g["ending_conventional_output_mw"].iloc[-1]),
            "terminal_state_mismatch_vs_paired_model": False,
            "mean_plan_24h_objective_eur_descriptive_only": float(g["plan_24h_objective_eur"].mean()),
            "solver_seconds": float(g["recourse_solve_time_s"].sum()),
        })
    # Mark pairwise terminal state parity, without inventing a terminal settlement.
    if len(rows) == 2:
        a, b = rows
        same = all(abs(a[k] - b[k]) < 1e-6 for k in (
            "terminal_battery_soc_mwh", "terminal_hydro_remaining_mwh", "terminal_conventional_output_mw"))
        a["terminal_state_mismatch_vs_paired_model"] = not same
        b["terminal_state_mismatch_vs_paired_model"] = not same
    return rows


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optional fresh output directory (relative to repo root or absolute).")
    args = parser.parse_args()
    if args.output_dir is not None:
        OUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if (OUT / "stage5_report.md").exists():
        raise SystemExit(f"Refusing to overwrite completed Stage 5 analysis: {OUT}")
    plans_all, execution_all = [], []
    forecast_rows, peak_rows, dispatch_rows, inputs = [], [], [], []
    for window, folder in RUNS:
        p_path, e_path = folder / "hourly_plans.csv", folder / "hourly_execution.csv"
        if not p_path.exists() or not e_path.exists():
            raise FileNotFoundError(f"Missing saved Stage 4 records in {folder}")
        p, e = pd.read_csv(p_path, parse_dates=["issue_timestamp", "target_timestamp"]), pd.read_csv(e_path, parse_dates=["issue_timestamp", "target_timestamp"])
        if len(p) != 8064 or len(e) != 336:
            raise ValueError(f"Unexpected row counts in {folder}: {len(p)} plans, {len(e)} execution")
        actuals = e[["target_timestamp", "actual_demand_mw"]].drop_duplicates()
        if actuals["target_timestamp"].duplicated().any():
            raise ValueError(f"Paired execution actuals disagree in {window}")
        forecast_rows += forecast_metrics(p, actuals, window)
        peak_rows += peak_metrics(p, actuals, window)
        dispatch_rows += dispatch_metrics(e, window)
        plans_all.append(p.assign(window=window)); execution_all.append(e.assign(window=window))
        for path in (p_path, e_path, folder / "validation.json", folder / "run_metadata.json"):
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})

    plans = pd.concat(plans_all, ignore_index=True)
    execution = pd.concat(execution_all, ignore_index=True)
    # Pooled metrics computed from the union of aligned observations, weighted by sample count.
    pooled_forecast = []
    for (model, horizon), g in plans.groupby(["model", "horizon_hours"]):
        actual = execution[["target_timestamp", "actual_demand_mw"]].drop_duplicates("target_timestamp")
        s = g.merge(actual, on="target_timestamp", how="inner", validate="many_to_one")
        err = s["demand_mw"] / DEMAND_SCALE - s["actual_demand_mw"]
        pooled_forecast.append({"window": "pooled_3_windows", "model": model,
            "evaluation": "fixed_24h" if int(horizon) == 24 else "all_leads",
            "horizon_hours": int(horizon) if int(horizon) == 24 else "1-24",
            "matched_predictions": int(len(s)), "unique_actual_targets": int(s["target_timestamp"].nunique()),
            "target_start_utc": s["target_timestamp"].min().isoformat(), "target_end_utc": s["target_timestamp"].max().isoformat(),
            "mae_mw": float(err.abs().mean()), "rmse_mw": rmse(err),
            "mape_pct": float((err.abs() / s["actual_demand_mw"].abs().replace(0, np.nan)).mean() * 100)})
    # Replace duplicate all_leads horizon=1..24 rows by a single pooled row, retained under fixed_24h separately.
    pooled_forecast = [r for r in pooled_forecast if r["evaluation"] == "fixed_24h"]
    for model, g in plans.groupby("model"):
        actual = execution[["target_timestamp", "actual_demand_mw"]].drop_duplicates("target_timestamp")
        s = g.merge(actual, on="target_timestamp", how="inner", validate="many_to_one")
        err = s["demand_mw"] / DEMAND_SCALE - s["actual_demand_mw"]
        pooled_forecast.append({"window": "pooled_3_windows", "model": model, "evaluation": "all_leads",
            "horizon_hours": "1-24", "matched_predictions": int(len(s)), "unique_actual_targets": int(s["target_timestamp"].nunique()),
            "target_start_utc": s["target_timestamp"].min().isoformat(), "target_end_utc": s["target_timestamp"].max().isoformat(),
            "mae_mw": float(err.abs().mean()), "rmse_mw": rmse(err),
            "mape_pct": float((err.abs() / s["actual_demand_mw"].abs().replace(0, np.nan)).mean() * 100)})
    dispatch_rows += dispatch_metrics(execution, "pooled_3_windows")

    OUT.mkdir(parents=True, exist_ok=True)
    fm = pd.DataFrame(forecast_rows + pooled_forecast).sort_values(["window", "model", "evaluation"])
    dm = pd.DataFrame(dispatch_rows).sort_values(["window", "model"])
    pm = pd.DataFrame(peak_rows).sort_values(["window", "model"])
    fm.to_csv(OUT / "stage5_forecast_metrics.csv", index=False)
    dm.to_csv(OUT / "stage5_dispatch_metrics.csv", index=False)
    pm.to_csv(OUT / "stage5_peak_metrics.csv", index=False)

    summary = {
        "stage": "5_economic_analysis", "source": "saved Stage 4 hourly records only",
        "dispatch_solves_run": 0, "model_training_run": False,
        "windows": [w for w, _ in RUNS], "issue_hours": 504,
        "plan_rows": int(len(plans)), "execution_rows": int(len(execution)),
        "execution_targets_utc": [execution.target_timestamp.min().isoformat(), execution.target_timestamp.max().isoformat()],
        "forecast_demand_scaling": "Dispatch forecasts are in representative-grid MW (national forecast multiplied by 0.001); converted back to MW before forecast scoring.",
        "cost_interpretation": "Configured modeled euros for the representative-grid simulation, not market bills. Recourse costs are summed once per executed hour; overlapping 24-hour objectives are not summed.",
        "terminal_accounting": "Terminal SOC, hydro, and conventional state are reported. No terminal energy value was invented; raw cost differences are not a fair winner where terminal states differ.",
        "input_files": inputs,
        "outputs": [p.name for p in OUT.iterdir()],
    }
    (OUT / "stage5_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Phase 4.5 Stage 5 — Economic analysis", "",
        "Analysis uses the three saved Stage 4 windows only. No dispatch solves or model training were run.", "",
        "## Method", "",
        "Actual demand is taken from each executed target record. Planned demand is matched by exact `target_timestamp`; dispatch-scaled demand is converted to national MW using the recorded 0.001 scale. All-lead metrics cover the saved forecasts for executed targets; fixed-24h metrics use only `horizon_hours=24`. Peak metrics use complete target days with 24 fixed-24h predictions.", "",
        "Realized recourse costs are summed once for each executed interval. Rolling 24-hour objectives overlap and are reported only as a descriptive mean, never summed. Costs are configured modeled euros for the representative-grid simulation, not actual market bills. Terminal inventory is reported; no terminal energy value is fabricated.", "",
        "## Pooled forecast metrics", "",
        markdown_table(fm.loc[fm.window == "pooled_3_windows", ["model", "evaluation", "matched_predictions", "mae_mw", "rmse_mw", "mape_pct"]]), "",
        "## Recourse dispatch metrics", "",
        markdown_table(dm.loc[dm.window == "pooled_3_windows", ["model", "executed_hours", "realized_recourse_cost_eur", "cost_per_served_mwh_eur", "emergency_energy_mwh", "unserved_energy_mwh", "battery_degradation_cost_eur", "solar_curtailment_mwh", "wind_curtailment_mwh", "hydro_used_mwh", "terminal_battery_soc_mwh", "terminal_hydro_remaining_mwh", "terminal_state_mismatch_vs_paired_model", "solver_failures"]]), "",
        "A cost difference is descriptive where terminal states differ; it is not a fair total-system cost ranking without an explicit terminal valuation.", "",
        "## Peak forecast metrics", "",
        markdown_table(pm), "",
        "## Reproducibility", "",
        "Run from the repository root with a fresh destination: `conda run -n griddrift python scripts/analyze_phase45_stage5.py --output-dir artifacts/experiments/phase45/analysis/stage5-rerun`. The default destination is the preserved Stage 5 result folder and refuses overwrite. Input SHA-256 hashes and metric CSVs are saved alongside this report.", "",
    ]
    (OUT / "stage5_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved Stage 5 analysis to {OUT.relative_to(ROOT)}")
    print(fm.loc[fm.window == "pooled_3_windows", ["model", "evaluation", "matched_predictions", "mae_mw", "rmse_mw"]].to_string(index=False))
    print(dm.loc[dm.window == "pooled_3_windows", ["model", "realized_recourse_cost_eur", "terminal_battery_soc_mwh", "terminal_hydro_remaining_mwh", "terminal_state_mismatch_vs_paired_model"]].to_string(index=False))


if __name__ == "__main__":
    main()
