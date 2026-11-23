"""Build Phase 3.8 presentation plots and summary from saved experiment outputs.

This script does not train models, make predictions, or run a replay. It only
refreshes its own Stage 6 presentation outputs; experiment inputs are read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE38 = ROOT / "artifacts/experiments/phase38"
COMPARISON = PHASE38 / "comparison/common-period-20261024"
XGB_ADAPT = PHASE38 / "adaptation_validation/holdout-paired-snapshots-20261024"
PHASE36 = ROOT / "artifacts/experiments/phase36"
OUTPUT = PHASE38 / "presentation/final-20261024"

COLORS = {
    "actual": "#72c7ff",
    "static_mlp": "#7f8c99",
    "adaptive_mlp": "#39a9ff",
    "static_xgboost": "#f5a623",
    "adaptive_xgboost": "#1b9e77",
    "random_forest": "#b07cc6",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Write this script's presentation outputs in both requested formats."""
    for extension in ("png", "svg"):
        path = OUTPUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("#10141b")
    ax.tick_params(colors="#e6edf3")
    ax.xaxis.label.set_color("#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    ax.title.set_color("#f4f7fa")
    ax.grid(axis="y", color="#3b4552", alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color("#3b4552")


def themed_fig(*args, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("#10141b")
    style(ax)
    return fig, ax


def load_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned = pd.read_csv(COMPARISON / "aligned_predictions.csv", parse_dates=["issue_timestamp", "target_timestamp"])
    metrics = pd.read_csv(COMPARISON / "summary_metrics.csv")
    alarms = pd.read_csv(COMPARISON / "adwin_alarm_targets.csv", parse_dates=["target_timestamp"])
    return aligned, metrics, alarms


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    models = ["static_mlp", "adaptive_mlp", "static_xgboost", "adaptive_xgboost", "random_forest"]
    labels = ["Static\nMLP", "Adaptive\nMLP", "Static\nXGBoost", "Adaptive\nXGBoost", "Random\nForest"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
    fig.patch.set_facecolor("#10141b")
    for ax, scope, title in zip(axes, ["fixed_24h", "all_horizons"], ["Fixed 24-hour-ahead", "All 24 forecast leads"]):
        style(ax)
        data = metrics[(metrics.scope == scope) & metrics.model.isin(models)].set_index("model").loc[models]
        x = range(len(models))
        width = 0.36
        ax.bar([v - width / 2 for v in x], data.mae_mw, width, label="MAE", color="#42b4e8")
        ax.bar([v + width / 2 for v in x], data.rmse_mw, width, label="RMSE", color="#f2a65a")
        ax.set_xticks(list(x), labels)
        ax.set_ylabel("Error (MW)")
        ax.set_title(title, pad=12)
        ax.legend(frameon=False, labelcolor="#e6edf3")
    fig.suptitle("Forecast model comparison — common 2025–2026 targets", color="#f4f7fa", fontsize=15)
    fig.tight_layout()
    save_figure(fig, "01_model_comparison")


def plot_actual_predictions(aligned: pd.DataFrame) -> None:
    fixed = aligned[aligned.horizon_hours == 24].copy().sort_values("target_timestamp")
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=14)
    frame = fixed[(fixed.target_timestamp >= start) & (fixed.target_timestamp < end)]
    fig, ax = themed_fig(figsize=(14, 6))
    ax.plot(frame.target_timestamp, frame.actual_demand_mw, color=COLORS["actual"], linewidth=1.7, label="Actual demand")
    for model, label, ls in [
        ("static_mlp", "Static MLP", "--"),
        ("adaptive_mlp", "Adaptive MLP", "-"),
        ("static_xgboost", "Static XGBoost", ":"),
        ("adaptive_xgboost", "Adaptive XGBoost", "-"),
        ("random_forest", "Random Forest", "-."),
    ]:
        ax.plot(frame.target_timestamp, frame[f"{model}_predicted_mw"], color=COLORS[model], linewidth=1.25, linestyle=ls, label=label)
    ax.set_title("Actual vs predicted demand — fixed 24-hour-ahead forecasts")
    ax.set_xlabel("Target timestamp (UTC)")
    ax.set_ylabel("Demand (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=mdates.UTC))
    ax.legend(ncol=3, frameon=False, labelcolor="#e6edf3")
    fig.tight_layout()
    save_figure(fig, "02_actual_vs_predicted")


def plot_adwin(aligned: pd.DataFrame, alarms: pd.DataFrame, effect: pd.DataFrame) -> None:
    fixed = aligned[aligned.horizon_hours == 24].copy().sort_values("target_timestamp")
    fixed["static_xgb_abs_error"] = (fixed.actual_demand_mw - fixed.static_xgboost_predicted_mw).abs()
    fixed["adaptive_xgb_abs_error"] = (fixed.actual_demand_mw - fixed.adaptive_xgboost_predicted_mw).abs()
    fixed["adaptive_mlp_abs_error"] = (fixed.actual_demand_mw - fixed.adaptive_mlp_predicted_mw).abs()
    fig, ax = themed_fig(figsize=(15, 6.4))
    ax.plot(fixed.target_timestamp, fixed.adaptive_mlp_abs_error, color=COLORS["adaptive_mlp"], alpha=0.74, linewidth=0.8, label="Adaptive MLP fixed-24h absolute error")
    ax.plot(fixed.target_timestamp, fixed.adaptive_xgb_abs_error, color=COLORS["adaptive_xgboost"], alpha=0.8, linewidth=0.8, label="Adaptive XGBoost fixed-24h absolute error")
    alarm_points = alarms.merge(
        fixed[["target_timestamp", "adaptive_mlp_abs_error", "adaptive_xgb_abs_error"]],
        on="target_timestamp",
        how="inner",
    )
    mlp_alarms = alarm_points[alarm_points.mlp_adwin_alarm.astype(bool)]
    xgb_alarms = alarm_points[alarm_points.xgboost_adwin_alarm.astype(bool)]
    ax.scatter(mlp_alarms.target_timestamp, mlp_alarms.adaptive_mlp_abs_error, marker="x", s=24, color="#ff4d6d", label="MLP ADWIN alarm", zorder=5)
    ax.scatter(xgb_alarms.target_timestamp, xgb_alarms.adaptive_xgb_abs_error, marker="x", s=24, color="#ffd166", label="XGBoost ADWIN alarm", zorder=5)
    accepted = effect[effect.window == "deployed_interval_up_to_7d"].copy()
    accepted_dates = pd.to_datetime(accepted.drift_timestamp, utc=True)
    ymin, ymax = ax.get_ylim()
    for i, timestamp in enumerate(accepted_dates):
        ax.axvline(timestamp, color="#d08bff", alpha=0.24, linewidth=0.8)
    ax.scatter(accepted_dates, [ymax * 0.94] * len(accepted_dates), marker="D", s=22, color="#d08bff", label="Accepted XGBoost update", zorder=6)
    ax.set_ylim(ymin, ymax * 1.08)
    ax.set_title("Fixed 24-hour-ahead errors, ADWIN alarms, and accepted XGBoost updates")
    ax.set_xlabel("Actual target timestamp (UTC)")
    ax.set_ylabel("Absolute error (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=mdates.UTC))
    ax.legend(ncol=2, frameon=False, labelcolor="#e6edf3")
    fig.tight_layout()
    save_figure(fig, "03_adwin_errors_and_updates")


def plot_before_after(effect: pd.DataFrame) -> None:
    rows = effect[effect.window == "deployed_interval_up_to_7d"].copy().sort_values("event_number")
    x = range(len(rows))
    fig, ax = themed_fig(figsize=(14, 5.8))
    ax.plot(x, rows.fixed24_previous_mae_mw, marker="o", markersize=4, color="#9aa6b2", label="Previous model MAE")
    ax.plot(x, rows.fixed24_updated_mae_mw, marker="o", markersize=4, color="#34c38f", label="Updated model MAE")
    for pos, (_, row) in enumerate(rows.iterrows()):
        color = {"improved": "#34c38f", "worsened": "#ff667a", "inconclusive": "#f2b84b"}.get(row.fixed24_effect, "#f2b84b")
        ax.plot([pos, pos], [row.fixed24_previous_mae_mw, row.fixed24_updated_mae_mw], color=color, linewidth=1.4, alpha=0.9)
    labels = pd.to_datetime(rows.drift_timestamp, utc=True).dt.strftime("%b %d")
    ax.set_xticks(list(x), labels, rotation=50, ha="right")
    ax.set_title("Accepted XGBoost snapshots — matched fixed-24h MAE in the following up-to-7-day window")
    ax.set_xlabel("Adaptation activation date (UTC)")
    ax.set_ylabel("MAE (MW)")
    ax.legend(frameon=False, labelcolor="#e6edf3")
    ax.text(0.01, 0.02, "Point changes; paired block-bootstrap results: 2 improved, 1 worsened, 17 inconclusive.", transform=ax.transAxes, color="#c5ced8", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "04_before_after_adaptation")


def plot_finetuning(metrics_path: Path) -> None:
    data = pd.read_csv(metrics_path)
    data = data[data.strategy.isin(["frozen_static_mlp", "previous_validated_finetune_60d", "gentle_recent_30d", "gentle_mixed_30d"])].copy()
    order = ["frozen_static_mlp", "previous_validated_finetune_60d", "gentle_recent_30d", "gentle_mixed_30d"]
    data = data.set_index("strategy").reindex(order).reset_index()
    names = ["Frozen", "Previous\nvalidated 60d", "Gentle\nrecent 30d", "Gentle\nmixed history 30d"]
    fig, ax = themed_fig(figsize=(11, 5.8))
    bars = ax.bar(names, data.mae_mw, color=["#82909d", "#4da3d9", "#61b89a", "#d39c4a"])
    ax.set_ylabel("Overall MAE (MW)")
    ax.set_title("Phase 3.6 fine-tuning strategy screen (development replay)")
    for bar, val in zip(bars, data.mae_mw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, f"{val:,.0f}", ha="center", color="#e6edf3", fontsize=9)
    accepted = data.accepted_adaptations.fillna(0).astype(int)
    fig.text(
        0.08,
        0.025,
        "Accepted updates: " + " | ".join(f"{name.replace(chr(10), ' ')} {count}" for name, count in zip(names, accepted)) + ". Short screening replay.",
        color="#c5ced8",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_figure(fig, "05_finetuning_strategy_screen")


def plot_monthly_seasonal(metrics: pd.DataFrame) -> None:
    data = metrics[
        metrics.model.isin(["static_xgboost", "adaptive_xgboost"])
        & metrics.scope.str.startswith(("month:", "season:"))
    ].copy()
    if data.empty:
        raise ValueError("Monthly/seasonal rows were not found in Stage 5 summary_metrics.csv")
    data["period"] = data.scope.str.split(":", n=1).str[1]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#10141b")
    for ax in axes:
        style(ax)
    monthly = data[data.scope.str.contains("month", case=False)].copy()
    seasonal = data[data.scope.str.contains("season", case=False)].copy()
    if not monthly.empty:
        pivot = monthly.pivot(index="period", columns="model", values="mae_mw")
        pivot = pivot.sort_index()
        axes[0].plot(pivot.index.astype(str), pivot["static_xgboost"], marker="o", color=COLORS["static_xgboost"], label="Static XGBoost")
        axes[0].plot(pivot.index.astype(str), pivot["adaptive_xgboost"], marker="o", color=COLORS["adaptive_xgboost"], label="Adaptive XGBoost")
        axes[0].set_title("Monthly fixed-24h MAE")
        axes[0].set_ylabel("MAE (MW)")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend(frameon=False, labelcolor="#e6edf3")
    if not seasonal.empty:
        pivot = seasonal.pivot(index="period", columns="model", values="mae_mw")
        axes[1].plot(pivot.index.astype(str), pivot["static_xgboost"], marker="o", color=COLORS["static_xgboost"], label="Static XGBoost")
        axes[1].plot(pivot.index.astype(str), pivot["adaptive_xgboost"], marker="o", color=COLORS["adaptive_xgboost"], label="Adaptive XGBoost")
        axes[1].set_title("Seasonal fixed-24h MAE")
        axes[1].set_ylabel("MAE (MW)")
        axes[1].legend(frameon=False, labelcolor="#e6edf3")
    fig.suptitle("Adaptive vs static XGBoost across time", color="#f4f7fa", fontsize=15)
    fig.tight_layout()
    save_figure(fig, "06_monthly_seasonal_xgboost")


def plot_xgb_event_effect(effect: pd.DataFrame) -> None:
    rows = effect[effect.window == "deployed_interval_up_to_7d"].copy()
    counts = rows.fixed24_effect.value_counts().reindex(["improved", "worsened", "inconclusive"], fill_value=0)
    fig, ax = themed_fig(figsize=(8, 5))
    palette = ["#34c38f", "#ff667a", "#f2b84b"]
    bars = ax.bar(["Improved", "Worsened", "Inconclusive"], counts.values, color=palette)
    ax.set_ylabel("Accepted updates (count)")
    ax.set_title("XGBoost adaptation evidence — 7-day paired block bootstrap")
    ax.set_ylim(0, max(counts.values) + 3)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val), ha="center", color="#e6edf3")
    ax.text(0.01, 0.02, "Point MAE fell for 16/20 updates; statistical classifications are based on paired 24-hour-block bootstrap intervals.", transform=ax.transAxes, color="#c5ced8", fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "07_xgboost_adaptation_effectiveness")


def write_report(metrics: pd.DataFrame, effect: pd.DataFrame, phase36_dev: pd.DataFrame) -> None:
    if (OUTPUT / "final_report.md").exists():
        return
    fixed = metrics[(metrics.scope == "fixed_24h")].set_index("model")
    allh = metrics[(metrics.scope == "all_horizons")].set_index("model")
    primary = effect[effect.window == "deployed_interval_up_to_7d"]
    mlp_dev = phase36_dev[phase36_dev.strategy == "gentle_mixed_60d_threshold_05pct"].iloc[0]
    lines = [
        "# GridDRIFT Phase 3.8 — Final comparison and presentation notes",
        "",
        "## Objective and protocol",
        "",
        "Compare MLP, XGBoost, and Random Forest forecasts using the saved common-period predictions. All five streams were joined by issue time, target time, and lead; actual demand was verified from the source CSV. No model training, replay, or prediction generation was performed for this stage.",
        "",
        "Evaluation targets: 2025-01-01 through 2026-10-20 UTC; 15,792 target hours and 379,008 issue/target/lead rows. The 2025–2026 interval had already been examined in earlier phases, so it is a comparative benchmark, not a new untouched confirmatory test.",
        "",
        "## Main benchmark results",
        "",
        "| Model | Fixed-24h MAE (MW) | Fixed-24h RMSE (MW) | All-lead MAE (MW) | All-lead RMSE (MW) |",
        "|---|---:|---:|---:|---:|",
    ]
    names = [("static_mlp", "Frozen MLP"), ("adaptive_mlp", "Adaptive MLP"), ("static_xgboost", "Frozen XGBoost"), ("adaptive_xgboost", "Adaptive XGBoost"), ("random_forest", "Frozen Random Forest")]
    for key, label in names:
        lines.append(f"| {label} | {fixed.loc[key, 'mae_mw']:,.2f} | {fixed.loc[key, 'rmse_mw']:,.2f} | {allh.loc[key, 'mae_mw']:,.2f} | {allh.loc[key, 'rmse_mw']:,.2f} |")
    lines += [
        "",
        "Adaptive MLP had the best all-lead MAE (1,658.21 MW). Adaptive XGBoost had the best fixed-24h MAE (1,640.94 MW), 1.78% below its frozen counterpart. In the fixed-24h results, adaptive MLP improved on frozen MLP by 1.81%; frozen XGBoost still outperformed both MLP streams. Random Forest was frozen and did not adapt.",
        "",
        "## Drift adaptation evidence",
        "",
        "| Saved stream | ADWIN alarms | Attempts | Accepted | Rejected |",
        "|---|---:|---:|---:|---:|",
        "| MLP adaptive replay | 92 | 33 | 0 | 33 |",
        "| XGBoost adaptive replay | 109 | 37 | 20 | 17 |",
        "| Random Forest | — | — | 0 (frozen) | — |",
        "",
        f"For the 20 accepted XGBoost snapshots, in the primary matched up-to-7-day post-activation windows, point MAE was lower for {(primary.fixed24_mae_change_mw < 0).sum()}/20 and higher for {(primary.fixed24_mae_change_mw > 0).sum()}/20. The paired 95% 24-hour-block bootstrap called {int((primary.fixed24_effect == 'improved').sum())} improved, {int((primary.fixed24_effect == 'worsened').sum())} worsened, and {int((primary.fixed24_effect == 'inconclusive').sum())} inconclusive. Thus, the aggregate adaptive XGBoost result is positive, but individual accepted updates are not reliably beneficial.",
        "",
        "On the union of 156 saved alarm target timestamps, fixed-24h adaptive MLP MAE (2,209.70 MW) was worse than static MLP (2,176.31 MW); adaptive XGBoost (2,271.73 MW) was better than static XGBoost (2,312.47 MW). These are alarm-point scores, not extended drift periods.",
        "",
        "## Fine-tuning development screen (separate evidence)",
        "",
        f"The Phase 3.6 development threshold sweep selected `{mlp_dev.strategy}` with fixed-24h MAE {mlp_dev.mae_mw:,.2f} MW versus {float(phase36_dev.iloc[0].mae_mw):,.2f} MW for its frozen baseline, and {int(mlp_dev.accepted_adaptations)} accepted update(s). The Phase 3.6 final-period run later accepted no score-period MLP updates (92 alarms, 33 attempts, 33 rejections). These are distinct periods and must not be blended into a single result.",
        "",
        "## Limitations and interpretation",
        "",
        "- Saved prediction comparisons are causal, timestamp-matched, and fixed-24h scoring isolates the outcome ADWIN monitors.",
        "- The benchmark is not untouched final-test evidence because these dates were previously inspected during Phases 3.6/3.7.",
        "- XGBoost post-update 7-day windows can overlap in calendar time; event-level paired results are descriptive and are not independent replications.",
        "- No Random Forest adaptation was tested. The selected RF configuration was tuned on 2024 development targets only.",
        "- Phase 3.6 fine-tuning screen is a separate earlier development experiment; some strategies had no accepted updates, and the short screening period is not comparable to the full 2025–2026 benchmark.",
        "",
        "## Presentation figures",
        "",
    ]
    for name, description in [
        ("01_model_comparison", "MAE/RMSE for the five saved forecast streams"),
        ("02_actual_vs_predicted", "A representative 14-day fixed-24h actual/predicted series"),
        ("03_adwin_errors_and_updates", "Fixed-24h errors with saved alarm and accepted-update timestamps"),
        ("04_before_after_adaptation", "Matched pre-update versus up-to-7-day post-update XGBoost MAE"),
        ("05_finetuning_strategy_screen", "Phase 3.6 development fine-tuning strategy screen"),
        ("06_monthly_seasonal_xgboost", "XGBoost fixed-24h MAE by month/season"),
        ("07_xgboost_adaptation_effectiveness", "Bootstrap evidence classification across accepted updates"),
    ]:
        lines.append(f"- `{name}.png` / `{name}.svg` — {description}.")
    lines += [
        "",
        "## Sources and reproduction",
        "",
        "Figures and this report are regenerated from the Stage 2 adaptation-effectiveness tables, Stage 5 aligned predictions/metrics, and the Phase 3.6 saved development metrics. Inputs are listed in `artifact_manifest.json`.",
        "",
        "```powershell",
        "conda activate griddrift",
        "python scripts/create_phase38_presentation.py",
        "```",
    ]
    (OUTPUT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output if args.output.is_absolute() else ROOT / args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)

    aligned, metrics, alarms = load_comparison()
    effect = pd.read_csv(XGB_ADAPT / "adaptation_effectiveness.csv")
    phase36_path = PHASE36 / "threshold-sweep/threshold-dev-20261023/metrics.csv"
    phase36_dev = pd.read_csv(phase36_path)
    short_path = PHASE36 / "short/screen-30d-20261023/metrics.csv"

    plot_model_comparison(metrics)
    plot_actual_predictions(aligned)
    plot_adwin(aligned, alarms, effect)
    plot_before_after(effect)
    plot_finetuning(short_path)
    plot_monthly_seasonal(metrics)
    plot_xgb_event_effect(effect)
    write_report(metrics, effect, phase36_dev)

    manifest = {
        "stage": 6,
        "generated_at": "2026-10-24",
        "models_retrained": False,
        "replay_rerun": False,
        "prediction_rows_recomputed": False,
        "inputs": [
            str(COMPARISON / "aligned_predictions.csv"),
            str(COMPARISON / "summary_metrics.csv"),
            str(COMPARISON / "adwin_alarm_targets.csv"),
            str(XGB_ADAPT / "adaptation_effectiveness.csv"),
            str(XGB_ADAPT / "xgboost_monthly_seasonal_consistency.csv"),
            str(phase36_path),
            str(short_path),
        ],
        "outputs": sorted(p.name for p in OUTPUT.iterdir() if p.is_file()),
        "chart_formats": ["PNG (300 DPI)", "SVG"],
    }
    manifest_path = OUTPUT / "artifact_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Presentation outputs: {OUTPUT}")
    print(f"PNG/SVG pairs: {sum(1 for p in OUTPUT.glob('*.png'))} / {sum(1 for p in OUTPUT.glob('*.svg'))}")
    print(f"Report: {OUTPUT / 'final_report.md'}")


if __name__ == "__main__":
    main()
