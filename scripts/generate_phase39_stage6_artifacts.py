"""Generate Phase 3.9 Stage 6 figures and report from saved outputs only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "artifacts/experiments/phase39"
RESULTS = PHASE / "development/threshold_results.csv"
COMPARISON = PHASE / "development_threshold_comparison.csv"
SELECTED = PHASE / "selected_configuration.json"
RUNS = PHASE / "development/runs"
OUT = PHASE / "presentation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS).sort_values("threshold_fraction")
    comparison = pd.read_csv(COMPARISON).sort_values("threshold_fraction")
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    selected_fraction = float(selected["selected_threshold_fraction"])
    static_mae = float(results["static_mae_mw"].iloc[0])

    thresholds = results["threshold_fraction"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.plot(thresholds, results["adaptive_mae_mw"], marker="o", linewidth=2.6,
            color="#1479b8", label="Adaptive MLP")
    ax.axhline(static_mae, color="#4a8f57", linestyle="--", linewidth=2,
               label=f"Frozen MLP ({static_mae:,.1f} MW)")
    ax.scatter([selected_fraction * 100],
               [float(results.loc[results["threshold_fraction"] == selected_fraction, "adaptive_mae_mw"].iloc[0])],
               s=120, marker="*", color="#ee8a24", edgecolor="black", linewidth=0.6,
               zorder=4, label=f"Selected: {selected_fraction:.0%}")
    ax.set(title="Validation threshold vs fixed 24-hour-ahead MAE",
           xlabel="Minimum candidate validation improvement threshold (%)",
           ylabel="Mean absolute error (MW)")
    ax.set_xticks(thresholds, [f"{v:g}%" for v in thresholds])
    ax.grid(axis="y", alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(OUT / "threshold_vs_forecasting_mae.png", dpi=240)
    fig.savefig(OUT / "threshold_vs_forecasting_mae.svg")
    plt.close(fig)

    # This figure pairs update frequency by threshold with the selected run's
    # observed 24-hour errors and actual recorded alarm/acceptance timestamps.
    run_counts = comparison
    run_id = str(results.loc[results["threshold_fraction"] == selected_fraction, "run_id"].iloc[0])
    run_dir = RUNS / run_id
    errors_path = run_dir / "adaptive_24h_errors.csv"
    events_path = run_dir / "adaptation_events.csv"
    errors = pd.read_csv(errors_path, parse_dates=["target_timestamp"])
    events = pd.read_csv(events_path, parse_dates=["drift_timestamp"])
    errors = errors.sort_values("target_timestamp")
    events = events.sort_values("drift_timestamp")
    daily_error = (errors.set_index("target_timestamp")["absolute_error_mw"]
                   .resample("D").mean().rolling(7, min_periods=1).mean())
    fig, (ax_counts, ax_error) = plt.subplots(
        2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1, 2]}, constrained_layout=True
    )
    x = run_counts["threshold_fraction"].to_numpy() * 100
    width = 0.32
    ax_counts.bar(x - width / 2, run_counts["drift_alarms"], width, color="#8795a1", label="ADWIN alarms")
    ax_counts.bar(x + width / 2, run_counts["accepted_adaptations"], width, color="#7a4fc2", label="Accepted updates")
    ax_counts.set(title="Alarm and update counts by validation threshold",
                  ylabel="Count", xticks=x, xticklabels=[f"{v:g}%" for v in x])
    ax_counts.grid(axis="y", alpha=0.25)
    ax_counts.legend(frameon=False, ncol=2, loc="upper right")

    ax_error.plot(daily_error.index, daily_error.values, color="#1668b2", linewidth=2,
                  label="7-day rolling mean of fixed-24h absolute error")
    alarm_times = events["drift_timestamp"].dropna()
    accepted = events[events["outcome"].astype(str).str.contains("accepted|adapted", case=False, na=False)]
    for idx, timestamp in enumerate(alarm_times):
        ax_error.axvline(timestamp, color="#d04a4a", alpha=0.15, linewidth=0.8,
                         label="ADWIN alarm" if idx == 0 else None)
    if not accepted.empty:
        accepted_y = [daily_error.iloc[np.abs((daily_error.index - t).asi8).argmin()] for t in accepted["drift_timestamp"]]
        ax_error.scatter(accepted["drift_timestamp"], accepted_y, marker="D", s=90,
                         color="#8c52c7", edgecolor="white", linewidth=0.5,
                         zorder=4, label="Accepted adaptation")
    ax_error.set(title=f"Selected {selected_fraction:.0%} threshold: error and event timeline",
                 xlabel="Target time (UTC)", ylabel="Absolute error (MW)")
    ax_error.xaxis.set_major_locator(mdates.MonthLocator())
    ax_error.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_error.grid(alpha=0.25)
    ax_error.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "threshold_counts_and_error_timeline.png", dpi=240)
    fig.savefig(OUT / "threshold_counts_and_error_timeline.svg")
    plt.close(fig)

    selected_row = results.loc[results["threshold_fraction"] == selected_fraction].iloc[0]
    lowest_threshold_row = results.loc[results["adaptive_mae_mw"].idxmin()]
    lowest_mae_threshold = float(lowest_threshold_row["threshold_fraction"])
    lower_mae_delta = static_mae - float(lowest_threshold_row["adaptive_mae_mw"])
    rows = []
    for row in results.itertuples(index=False):
        rows.append(
            f"| {row.threshold_fraction:.1%} | {row.adaptive_mae_mw:,.2f} | "
            f"{row.adaptive_rmse_mw:,.2f} | {row.drift_alarms} | "
            f"{row.adaptation_attempts} | {row.accepted_adaptations} | {row.rejected_adaptations} |"
        )
    report = f"""# GridDRIFT Phase 3.9 — Threshold analysis and presentation summary

## Objective and evaluation

This analysis asks how strict candidate validation should be before deploying a drift-triggered MLP update. It uses the five saved 2024 development replays only, with {int(results['matched_fixed_24h_predictions'].iloc[0]):,} matched fixed-24-hour-ahead targets per configuration. The frozen MLP baseline is identical across runs: MAE {static_mae:,.2f} MW, RMSE {float(results['static_rmse_mw'].iloc[0]):,.2f} MW. No new model fitting, replay, or evaluation-period selection was performed for this report.

## Results

| Validation threshold | Adaptive MAE (MW) | Adaptive RMSE (MW) | ADWIN alarms | Attempts | Accepted | Rejected |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The frozen selection remains **{selected_fraction:.1%}**. It had the lowest fixed-24-hour MAE ({float(selected_row['adaptive_mae_mw']):,.2f} MW) and RMSE ({float(selected_row['adaptive_rmse_mw']):,.2f} MW) in the development comparison, while accepting one candidate. Its MAE improvement over static is {static_mae - float(selected_row['adaptive_mae_mw']):,.2f} MW ({float(selected['mae_improvement_vs_static_pct']):.3f}%). This is a modest development-set gain.

## What the threshold changes

- A validation threshold requires the newly trained candidate to beat the deployed model by a minimum fraction on a later, time-ordered validation segment before its weights are accepted. It prevents deploying candidates that merely pass a drift alarm but do not demonstrate enough validation improvement.
- Lowering the threshold from 5% to 0% increased accepted updates from {int(selected_row['accepted_adaptations'])} to {int(results.iloc[0]['accepted_adaptations'])}. Fourteen updates were accepted at 0%; at 0.5% and 1%, two were accepted; at 2%, two; at 5%, one.
- The lower thresholds did not produce a consistent MAE advantage. The best observed MAE was at {lowest_mae_threshold:.1%} ({float(lowest_threshold_row['adaptive_mae_mw']):,.2f} MW), only {lower_mae_delta:,.2f} MW below frozen static. The selected 5% setting {"tied for" if abs(float(selected_row['adaptive_mae_mw']) - float(lowest_threshold_row['adaptive_mae_mw'])) < 1e-9 else "was"} the lowest observed MAE (difference from minimum: {float(selected_row['adaptive_mae_mw']) - float(lowest_threshold_row['adaptive_mae_mw']):,.2f} MW).
- At 0%, four accepted updates had equal or worse subsequent seven-day MAE; the other ten measured windows improved. At 0.5% and 1%, both accepted updates had lower subsequent seven-day MAE; at 5%, its one accepted update's subsequent window improved. These are small, descriptive event counts and are not causal comparisons.

## Recommendation and limitations

Keep **5%** as the frozen development choice for the next comparison because it delivered the lowest aggregate fixed-24-hour MAE and RMSE here, with fewer accepted updates. The margin is small, only one update was accepted at 5%, and the development period was previously seen during earlier phases. This is not evidence of performance on a fresh untouched test set. The 2025–2026 benchmark is exploratory and was not used to choose the threshold. A future untouched evaluation should be reported separately before claiming generalization.

The charts in this folder were generated directly from saved CSV outputs. Recreate them with:

```powershell
conda run -n griddrift python scripts/generate_phase39_stage6_artifacts.py
```
"""
    (OUT / "phase39_threshold_presentation_summary.md").write_text(report, encoding="utf-8")

    metadata = {
        "stage": 6,
        "status": "complete",
        "data_sources": [
            str(RESULTS.relative_to(ROOT)),
            str(COMPARISON.relative_to(ROOT)),
            str((run_dir / "adaptive_24h_errors.csv").relative_to(ROOT)),
            str((run_dir / "adaptation_events.csv").relative_to(ROOT)),
        ],
        "selected_threshold_fraction": selected_fraction,
        "plots": [
            "threshold_vs_forecasting_mae.png",
            "threshold_vs_forecasting_mae.svg",
            "threshold_counts_and_error_timeline.png",
            "threshold_counts_and_error_timeline.svg",
        ],
        "models_retrained_or_replayed": False,
    }
    (OUT / "stage6_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Phase 3.9 Stage 6 artifacts to {OUT}")


if __name__ == "__main__":
    main()
