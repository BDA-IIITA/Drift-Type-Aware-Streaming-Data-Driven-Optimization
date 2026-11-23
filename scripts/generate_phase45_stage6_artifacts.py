"""Create Phase 4.5 presentation artifacts from saved Stage 4/5 outputs only."""
from __future__ import annotations

import hashlib
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experiments/phase45"
STAGE5 = BASE / "analysis/stage5"
OUT = BASE / "presentation/stage6_final_20261024"
WINDOWS = [
    ("spring-2025w14", "Spring 2025", BASE / "runs/phase45-spring-2025w14-normal-seed42"),
    ("summer-2025w27", "Summer 2025", BASE / "runs/phase45-summer-2025w27-normal-seed42"),
    ("autumn-2025w40", "Autumn 2025", BASE / "runs/phase45-autumn-2025w40-normal-seed42"),
]
COLORS = {"adaptive_mlp": "#2878B5", "adaptive_xgboost": "#E07A24"}
LABELS = {"adaptive_mlp": "Adaptive MLP", "adaptive_xgboost": "Adaptive XGBoost"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_figure(fig: plt.Figure, name: str) -> list[dict]:
    paths = []
    for suffix in ("png", "svg"):
        path = OUT / "plots" / f"{name}.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    plt.close(fig)
    return paths


def style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optional fresh output directory (relative to repo root or absolute).")
    args = parser.parse_args()
    if args.output_dir is not None:
        OUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    complete = OUT / "stage6_manifest.json"
    if complete.exists():
        raise SystemExit(f"Refusing to overwrite completed Stage 6 artifacts: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)

    # Gather saved records, never dispatch or forecast again.
    plans, execution = [], []
    source_files: list[Path] = []
    for window, label, folder in WINDOWS:
        p_path, e_path = folder / "hourly_plans.csv", folder / "hourly_execution.csv"
        plans.append(pd.read_csv(p_path, parse_dates=["issue_timestamp", "target_timestamp"]).assign(window=window, window_label=label))
        execution.append(pd.read_csv(e_path, parse_dates=["issue_timestamp", "target_timestamp"]).assign(window=window, window_label=label))
        source_files.extend([p_path, e_path, folder / "run_metadata.json", folder / "validation.json"])
    plans_df = pd.concat(plans, ignore_index=True)
    execution_df = pd.concat(execution, ignore_index=True)
    forecast_metrics = pd.read_csv(STAGE5 / "stage5_forecast_metrics.csv")
    dispatch_metrics = pd.read_csv(STAGE5 / "stage5_dispatch_metrics.csv")
    peak_metrics = pd.read_csv(STAGE5 / "stage5_peak_metrics.csv")
    source_files.extend([STAGE5 / "stage5_forecast_metrics.csv", STAGE5 / "stage5_dispatch_metrics.csv", STAGE5 / "stage5_peak_metrics.csv", STAGE5 / "stage5_comparison_summary.json"])

    # Keep a self-contained copy of the metric outputs and active assumptions.
    for source in (STAGE5 / "stage5_forecast_metrics.csv", STAGE5 / "stage5_dispatch_metrics.csv", STAGE5 / "stage5_peak_metrics.csv", STAGE5 / "stage5_comparison_summary.json"):
        shutil.copy2(source, OUT / "data" / source.name)
    for source in (ROOT / "configs/assets.yaml", ROOT / "configs/integration.yaml"):
        shutil.copy2(source, OUT / "configs" / source.name)
        source_files.append(source)
    # The hourly experiment records are part of the reusable presentation package.
    for _, _, folder in WINDOWS:
        target = OUT / "data" / folder.name
        target.mkdir(exist_ok=True)
        for filename in ("hourly_plans.csv", "hourly_execution.csv", "run_metadata.json", "validation.json"):
            shutil.copy2(folder / filename, target / filename)

    plot_manifest: list[dict] = []

    # 1. Forecast MAE/RMSE, separated by all-lead and fixed 24-hour evaluations.
    pooled = forecast_metrics[forecast_metrics["window"] == "pooled_3_windows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, evaluation, title in zip(axes, ("all_leads", "fixed_24h"), ("All forecast leads (1–24 h)", "Fixed 24-hour-ahead forecasts")):
        subset = pooled[pooled["evaluation"] == evaluation].set_index("model").reindex(COLORS)
        x = np.arange(2); width = 0.34
        ax.bar(x - width / 2, subset["mae_mw"], width, label="MAE", color="#48A9A6")
        ax.bar(x + width / 2, subset["rmse_mw"], width, label="RMSE", color="#E0A458")
        ax.set_xticks(x, [LABELS[m] for m in COLORS]); ax.tick_params(axis="x", rotation=12)
        style(ax, title, "Error (MW)"); ax.legend(frameon=False)
    fig.suptitle("Forecast accuracy across the three Stage 4 windows", fontsize=16, weight="bold")
    fig.tight_layout()
    plot_manifest += save_figure(fig, "01_forecast_accuracy_comparison")

    # Prepare exact fixed-24 target alignment for plots 2 and 3.
    fixed = plans_df[plans_df["horizon_hours"] == 24].copy()
    actuals = execution_df[["window", "target_timestamp", "actual_demand_mw"]].drop_duplicates()
    if actuals.duplicated(["window", "target_timestamp"]).any():
        raise ValueError("Actual demand records disagree on a target timestamp")
    fixed = fixed.merge(actuals, on=["window", "target_timestamp"], how="inner", validate="many_to_one")
    fixed["predicted_demand_mw"] = fixed["demand_mw"] / 0.001
    fixed["abs_error_mw"] = (fixed["predicted_demand_mw"] - fixed["actual_demand_mw"]).abs()

    # 2. Actual and fixed-24 predictions for a complete representative spring week.
    spring = fixed[fixed["window"] == "spring-2025w14"].copy()
    daily = spring.groupby(spring["target_timestamp"].dt.floor("D"))["target_timestamp"].nunique()
    complete_days = daily[daily == 24].index
    sample_days = complete_days[:6]
    sample = spring[spring["target_timestamp"].dt.floor("D").isin(sample_days)]
    if sample.empty:
        raise ValueError("No complete spring target days available for forecast plot")
    actual = sample[["target_timestamp", "actual_demand_mw"]].drop_duplicates("target_timestamp").sort_values("target_timestamp")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(actual["target_timestamp"], actual["actual_demand_mw"], color="#238636", lw=2, label="Actual demand")
    for model in COLORS:
        part = sample[sample["model"] == model].sort_values("target_timestamp")
        ax.plot(part["target_timestamp"], part["predicted_demand_mw"], color=COLORS[model], lw=1.45,
                alpha=0.92, label=LABELS[model])
    style(ax, "Actual demand vs fixed 24-hour-ahead forecasts — Spring 2025", "Demand (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=sample["target_timestamp"].iloc[0].tz))
    ax.legend(frameon=False, ncol=3); fig.tight_layout()
    plot_manifest += save_figure(fig, "02_actual_vs_fixed24_forecast_spring")

    # 3. Fixed-horizon absolute forecast error over the evaluated periods.
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for model in COLORS:
        for window, label, _ in WINDOWS:
            part = fixed[(fixed["model"] == model) & (fixed["window"] == window)].sort_values("target_timestamp")
            ax.plot(part["target_timestamp"], part["abs_error_mw"], color=COLORS[model], lw=1.15,
                    alpha=0.78, label=f"{LABELS[model]} — {label}")
    style(ax, "Fixed 24-hour-ahead absolute error over time", "Absolute error (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=fixed["target_timestamp"].iloc[0].tz))
    ax.legend(frameon=False, ncol=2, fontsize=8); fig.tight_layout()
    plot_manifest += save_figure(fig, "03_fixed24_error_over_time")

    # 4. Paired raw recourse cost by season; labels warn that end states differ.
    season_order = [w[0] for w in WINDOWS]
    cost_table = dispatch_metrics[dispatch_metrics["window"].isin(season_order)].pivot(index="window", columns="model", values="realized_recourse_cost_eur").reindex(season_order)
    fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(len(season_order)); width = 0.34
    for shift, model in ((-width/2, "adaptive_mlp"), (width/2, "adaptive_xgboost")):
        bars = ax.bar(x + shift, cost_table[model], width, color=COLORS[model], label=LABELS[model])
        ax.bar_label(bars, fmt="€%.0f", padding=3, fontsize=8)
    ax.set_xticks(x, ["Spring", "Summer", "Autumn"])
    style(ax, "Executed-hour recourse cost by evaluation window", "Configured modeled cost (€)")
    ax.legend(frameon=False); ax.text(0.01, -0.24, "Raw totals only: terminal battery, hydro, or generator states differ in each paired window.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    plot_manifest += save_figure(fig, "04_recourse_cost_by_window_terminal_caveat")

    # 5. Energy delivered in recourse, by source/flow, pooled across windows.
    flow_cols = {"Solar": "recourse_solar_mw", "Wind": "recourse_wind_mw", "Hydro": "recourse_hydro_mw",
                 "Conventional": "recourse_conventional_mw", "Battery discharge": "recourse_battery_discharge_mw",
                 "Emergency": "recourse_emergency_mw", "Unserved": "recourse_unserved_mw"}
    energy = execution_df.groupby("model")[list(flow_cols.values())].sum() * 1.0
    fig, ax = plt.subplots(figsize=(10, 5.5)); bottoms = np.zeros(2); x = np.arange(2)
    palette = ["#59A14F", "#76B7B2", "#B6992D", "#E15759", "#8064A2", "#F28E2B", "#9C755F"]
    for (name, col), color in zip(flow_cols.items(), palette):
        vals = np.array([energy.loc[m, col] for m in COLORS])
        ax.bar(x, vals, bottom=bottoms, color=color, label=name)
        bottoms += vals
    ax.set_xticks(x, [LABELS[m] for m in COLORS])
    style(ax, "Recourse energy flows across all evaluation windows", "Summed interval energy (MWh)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    ax.text(0.01, -0.24, "Battery discharge is stored energy, not a separate primary generation source; solar/wind/hydro generation are also shown.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    plot_manifest += save_figure(fig, "05_recourse_energy_flows")

    # 6. State trajectories: independent battery/hydro/generator state for each model/window.
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    for ax, column, title, ylabel in (
        (axes[0], "ending_battery_soc_mwh", "Battery state of charge", "SOC (MWh)"),
        (axes[1], "ending_hydro_remaining_mwh", "Hydro budget remaining", "Remaining hydro (MWh)"),
        (axes[2], "ending_conventional_output_mw", "Conventional generator output state", "Output (MW)"),
    ):
        for window, label, _ in WINDOWS:
            for model in COLORS:
                part = execution_df[(execution_df["window"] == window) & (execution_df["model"] == model)].sort_values("target_timestamp")
                ax.plot(part["target_timestamp"], part[column], color=COLORS[model], lw=1.3,
                        linestyle="-" if window == "spring-2025w14" else ("--" if window == "summer-2025w27" else ":"),
                        label=f"{LABELS[model]} — {label}")
        style(ax, title, ylabel)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Executed target timestamp (UTC)")
    fig.suptitle("Operational state carried through each seven-day window", fontsize=16, weight="bold")
    fig.tight_layout()
    plot_manifest += save_figure(fig, "06_battery_hydro_generator_states")

    # Concise final report with evidence and caveats.
    pooled_forecast = forecast_metrics[forecast_metrics["window"] == "pooled_3_windows"].copy()
    pooled_dispatch = dispatch_metrics[dispatch_metrics["window"] == "pooled_3_windows"].set_index("model")
    def frow(model: str, evaluation: str) -> pd.Series:
        return pooled_forecast[(pooled_forecast.model == model) & (pooled_forecast.evaluation == evaluation)].iloc[0]
    mlp24, xgb24 = frow("adaptive_mlp", "fixed_24h"), frow("adaptive_xgboost", "fixed_24h")
    mlp_all, xgb_all = frow("adaptive_mlp", "all_leads"), frow("adaptive_xgboost", "all_leads")
    delta_mae = (mlp24.mae_mw - xgb24.mae_mw) / mlp24.mae_mw * 100
    m_cost, x_cost = pooled_dispatch.loc["adaptive_mlp", "realized_recourse_cost_eur"], pooled_dispatch.loc["adaptive_xgboost", "realized_recourse_cost_eur"]
    cost_diff = x_cost - m_cost
    report = f"""# Phase 4.5 final report — forecast-driven dispatch comparison

## Objective and evaluation

Compare saved adaptive MLP and adaptive XGBoost forecast streams in the same forecast-driven dispatch harness. Both use the same three preregistered, disjoint seven-day windows (April 1–7, July 1–7, and October 1–7, 2025), normal simulated renewable scenario, seed 42, assets, costs, and physical constraints. Each model starts each window at the same battery SOC (15 MWh), hydro budget (120 MWh), and generator output (0 MW), then carries its own state through that window.

This report is generated only from saved Stage 4 and Stage 5 outputs. No forecasting, retraining, or dispatch optimization was run to produce these figures.

## Forecast accuracy

Predictions are matched to actual demand by exact target timestamp. There are **{int(mlp_all.matched_predictions):,} all-lead predictions** and **{int(mlp24.matched_predictions):,} fixed 24-hour-ahead predictions per model** over the three windows.

| Model | All-lead MAE (MW) | All-lead RMSE (MW) | Fixed-24h MAE (MW) | Fixed-24h RMSE (MW) | Fixed-24h MAPE |
|---|---:|---:|---:|---:|---:|
| Adaptive MLP | {mlp_all.mae_mw:,.1f} | {mlp_all.rmse_mw:,.1f} | {mlp24.mae_mw:,.1f} | {mlp24.rmse_mw:,.1f} | {mlp24.mape_pct:.2f}% |
| Adaptive XGBoost | {xgb_all.mae_mw:,.1f} | {xgb_all.rmse_mw:,.1f} | {xgb24.mae_mw:,.1f} | {xgb24.rmse_mw:,.1f} | {xgb24.mape_pct:.2f}% |

XGBoost's fixed-24h MAE is {delta_mae:.2f}% lower on this evaluation set. All-lead MAE is {mlp_all.mae_mw-xgb_all.mae_mw:,.1f} MW lower. Peak errors vary by season; see `data/stage5_peak_metrics.csv`.

## Dispatch outcomes

| Model | Recourse cost (€) | Cost / served MWh (€) | Emergency energy (MWh) | Unserved (MWh) | Battery wear (€) | Hydro used (MWh) | Terminal SOC (MWh) | Terminal hydro (MWh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive MLP | {m_cost:,.2f} | {pooled_dispatch.loc['adaptive_mlp','cost_per_served_mwh_eur']:.2f} | {pooled_dispatch.loc['adaptive_mlp','emergency_energy_mwh']:.2f} | {pooled_dispatch.loc['adaptive_mlp','unserved_energy_mwh']:.2f} | {pooled_dispatch.loc['adaptive_mlp','battery_degradation_cost_eur']:.2f} | {pooled_dispatch.loc['adaptive_mlp','hydro_used_mwh']:.2f} | {pooled_dispatch.loc['adaptive_mlp','terminal_battery_soc_mwh']:.2f} | {pooled_dispatch.loc['adaptive_mlp','terminal_hydro_remaining_mwh']:.2f} |
| Adaptive XGBoost | {x_cost:,.2f} | {pooled_dispatch.loc['adaptive_xgboost','cost_per_served_mwh_eur']:.2f} | {pooled_dispatch.loc['adaptive_xgboost','emergency_energy_mwh']:.2f} | {pooled_dispatch.loc['adaptive_xgboost','unserved_energy_mwh']:.2f} | {pooled_dispatch.loc['adaptive_xgboost','battery_degradation_cost_eur']:.2f} | {pooled_dispatch.loc['adaptive_xgboost','hydro_used_mwh']:.2f} | {pooled_dispatch.loc['adaptive_xgboost','terminal_battery_soc_mwh']:.2f} | {pooled_dispatch.loc['adaptive_xgboost','terminal_hydro_remaining_mwh']:.2f} |

XGBoost's raw pooled recourse cost is €{abs(cost_diff):,.2f} {'lower' if cost_diff < 0 else 'higher'} ({abs(cost_diff)/m_cost*100:.3f}%). **This is not a fair economic win:** within-window terminal battery, hydro, or conventional-generator states differ, and no terminal inventory valuation was configured. Per-window costs go in different directions. All 1,008 recourse solves were optimal; unserved energy was zero.

## Interpretation and recommendation

For demand forecasting alone, adaptive XGBoost is the stronger candidate on these windows because it has lower pooled all-lead and fixed-24h MAE/RMSE. For dispatch economics, the evidence is inconclusive: the raw cost gap is small, varies by window, and terminal states are not equivalent. Treat XGBoost as the provisional forecaster for further evaluation, but do not claim that it reduced real-world electricity bills. The experiment uses simulated renewable availability and configured representative-grid costs; it does not measure market settlement or real renewable forecasts. Forecast inference runtime was not compared here.

## Reproduction and contents

- Recreate Stage 5 metrics from stored records in a new folder: `conda run -n griddrift python scripts/analyze_phase45_stage5.py --output-dir artifacts/experiments/phase45/analysis/stage5-rerun`.
- Recreate these plots/report from saved Stage 4/5 outputs in a new folder: `conda run -n griddrift python scripts/generate_phase45_stage6_artifacts.py --output-dir artifacts/experiments/phase45/presentation/stage6_rerun`. The default output folder is preserved and refuses overwrite.
- `data/` contains copies of Stage 5 metrics, summary JSON, and the three paired hourly plan/execution records with metadata and validation. `configs/` contains the active `assets.yaml` and `integration.yaml` copies.
- Original run outputs remain in `artifacts/experiments/phase45/runs/`; original Stage 5 outputs remain in `artifacts/experiments/phase45/analysis/stage5/`.

All costs are modeled euros under configured assumptions for a representative grid, not market expenditure. See the Stage 5 report for detailed per-window metrics and input hashes.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")

    # Manifest is written last to mark the stage complete and make the output auditable.
    generated = [p for p in OUT.rglob("*") if p.is_file() and p != complete]
    manifest = {
        "phase": "4.5", "stage": "6_final_report_and_plots", "status": "complete",
        "source_scope": "saved Stage 4 run records and Stage 5 metrics only",
        "dispatch_solves_run": 0, "forecasting_or_training_run": False,
        "plot_families": [
            "forecast MAE/RMSE comparison", "actual vs fixed-24h predictions",
            "fixed-24h absolute errors over time", "recourse cost by window with terminal-state caveat",
            "recourse energy flows", "battery/hydro/generator state trajectories",
        ],
        "outputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size} for p in generated],
        "source_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in source_files],
    }
    complete.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stage 6 artifacts saved to {OUT.relative_to(ROOT)}")
    print(f"Plots: {len(plot_manifest)} files ({len(plot_manifest)//2} PNG/SVG pairs); final report and manifest saved.")


if __name__ == "__main__":
    main()
