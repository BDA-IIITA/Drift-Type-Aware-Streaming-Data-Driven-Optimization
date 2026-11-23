"""Solve all reproducible standalone Phase 4 scenarios and export results."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch
from griddrift.dispatch.baselines import run_greedy_dispatch, terminal_soc_settlement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default="configs/assets.yaml")
    parser.add_argument("--scenarios", default="configs/scenarios")
    parser.add_argument("--output", default="artifacts/dispatch/demo")
    parser.add_argument("--solver", default="appsi_highs")
    args = parser.parse_args()
    assets, settings = load_dispatch_config(args.assets)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(Path(args.scenarios).glob("*.yaml")):
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        overrides = scenario.get("asset_overrides", {})
        scenario_assets = replace(assets, **overrides) if overrides else assets
        demand = scenario["demand_mw"]
        solar, wind = scenario["solar_available_mw"], scenario["wind_available_mw"]
        horizon = settings["horizon_hours"]
        if not (len(demand) == len(solar) == len(wind) == horizon):
            raise ValueError(f"{path}: demand and renewable series must each have {horizon} values")
        timestamps = pd.date_range("2025-01-01", periods=horizon, freq="h", tz="UTC")
        state_data = scenario.get("initial_state", {})
        state = DispatchState(
            battery_soc_mwh=float(
                state_data.get("battery_soc_mwh", scenario_assets.battery.initial_soc_mwh)
            ),
            hydro_remaining_mwh=float(
                state_data.get(
                    "hydro_remaining_mwh", scenario_assets.hydro.initial_energy_budget_mwh
                )
            ),
            conventional_output_mw=float(state_data.get("conventional_output_mw", 0.0)),
        )
        result = optimize_dispatch(
            demand,
            {"solar": solar, "wind": wind},
            scenario_assets,
            state,
            timestamps=timestamps,
            conventional_costs_eur_mwh=scenario.get("conventional_costs_eur_mwh"),
            emergency_costs_eur_mwh=scenario.get("emergency_costs_eur_mwh"),
            hydro_inflow_mw=scenario.get("hydro_inflow_mw"),
            conventional_min_output_mw=settings["conventional_min_output_mw"],
            terminal_soc_policy=settings["terminal_soc_policy"],
            terminal_soc_target_mwh=settings["terminal_soc_target_mwh"],
            solver_name=args.solver,
        )
        result.export(out / scenario["name"], scenario["name"])
        record = {
            "scenario": scenario["name"],
            "optimizer_cost_eur": result.total_cost_eur,
            "solver_status": result.solver_status,
            "termination_condition": result.termination_condition,
            "formulation": result.formulation,
            "hydro_used_mwh": result.hydro_used_mwh,
            "hydro_remaining_mwh": result.hydro_remaining_mwh,
            "unserved_mwh": float(
                result.schedule.unserved_mw.sum() * scenario_assets.interval_hours
            ),
            "emergency_mwh": float(
                result.schedule.emergency_mw.sum() * scenario_assets.interval_hours
            ),
            "solve_time_s": result.solve_time_s,
        }
        if scenario["name"] == "ramp_flexibility":
            fair_multi = optimize_dispatch(
                demand,
                {"solar": solar, "wind": wind},
                scenario_assets,
                state,
                timestamps=timestamps,
                conventional_costs_eur_mwh=scenario.get("conventional_costs_eur_mwh"),
                emergency_costs_eur_mwh=scenario.get("emergency_costs_eur_mwh"),
                hydro_inflow_mw=scenario.get("hydro_inflow_mw"),
                conventional_min_output_mw=settings["conventional_min_output_mw"],
                terminal_soc_policy="none",
                solver_name=args.solver,
            )
            fair_multi.schedule.to_csv(out / scenario["name"] / "multi_period_comparable.csv", index_label="timestamp")
            replacement_price = sum(scenario.get("conventional_costs_eur_mwh", [scenario_assets.conventional.marginal_cost_eur_mwh] * horizon)) / horizon
            multi_settlement = terminal_soc_settlement(
                fair_multi.schedule.soc_mwh.iloc[-1], state.battery_soc_mwh, scenario_assets, replacement_price
            )
            greedy_started = time.perf_counter()
            greedy, raw_cost, settlement = run_greedy_dispatch(
                demand,
                {"solar": solar, "wind": wind},
                scenario_assets,
                state,
                timestamps=timestamps,
                conventional_costs_eur_mwh=scenario.get("conventional_costs_eur_mwh"),
                emergency_costs_eur_mwh=scenario.get("emergency_costs_eur_mwh"),
                hydro_inflow_mw=scenario.get("hydro_inflow_mw"),
                terminal_soc_target_mwh=state.battery_soc_mwh,
                conventional_min_output_mw=settings["conventional_min_output_mw"],
                solver_name=args.solver,
            )
            greedy_runtime_s = time.perf_counter() - greedy_started
            greedy.to_csv(out / scenario["name"] / "greedy.csv", index_label="timestamp")
            greedy_total = raw_cost + settlement
            multi_total = fair_multi.total_cost_eur + multi_settlement
            record.update(
                {
                    "comparison_method": "both terminal SOC unconstrained; same replacement-value settlement",
                    "multi_period_comparable_raw_cost_eur": fair_multi.total_cost_eur,
                    "multi_period_terminal_soc_settlement_eur": multi_settlement,
                    "multi_period_comparable_adjusted_cost_eur": multi_total,
                    "greedy_raw_cost_eur": raw_cost,
                    "greedy_terminal_soc_settlement_eur": settlement,
                    "greedy_adjusted_cost_eur": greedy_total,
                    "multi_period_saving_eur": greedy_total - multi_total,
                    "greedy_runtime_s": greedy_runtime_s,
                }
            )
        summary.append(record)
        print(
            f"{scenario['name']}: cost={result.total_cost_eur:.2f} EUR, "
            f"hydro={result.hydro_used_mwh:.2f} MWh used, "
            f"emergency={record['emergency_mwh']:.2f} MWh, "
            f"unserved={record['unserved_mwh']:.2f} MWh"
        )
    pd.DataFrame(summary).to_csv(out / "scenario_summary.csv", index=False)
    (out / "scenario_summary.json").write_text(
        pd.DataFrame(summary).to_json(orient="records", indent=2), encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Install the optimization extra (includes matplotlib) to create plots"
        ) from exc
    for record in summary:
        scenario_dir = out / record["scenario"]
        frame = pd.read_csv(scenario_dir / f"{record['scenario']}.csv")
        fig, (ax, soc_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for col in (
            "solar_mw",
            "wind_mw",
            "hydro_mw",
            "conventional_mw",
            "battery_charge_mw",
            "battery_discharge_mw",
            "emergency_mw",
        ):
            ax.plot(frame[col], label=col.removesuffix("_mw"))
        ax.plot(frame.demand_mw, color="black", linewidth=2, label="demand")
        ax.set(title=record["scenario"].replace("_", " "), xlabel="Hour", ylabel="MW")
        ax.legend(ncol=3, fontsize=8)
        ax.set_ylabel("Power (MW)")
        soc_ax.plot(frame.soc_mwh, color="purple", label="battery SOC")
        soc_ax.plot(frame.hydro_remaining_mwh, color="teal", label="hydro energy remaining")
        soc_ax.set(xlabel="Hour", ylabel="Energy (MWh)")
        soc_ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(scenario_dir / "dispatch.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
