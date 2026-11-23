"""Multi-period, single-area dispatch LP.

Renewable availability is split into dispatched generation and curtailment;
curtailment never appears in the power balance. Storage SOC and hydro reservoir
energy are state variables at the end of each interval. The model intentionally
does not include unit commitment (startup, on/off, or minimum up/down times).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from griddrift.config.schema import AssetConfig, BatteryConfig, HydroConfig, ResourceConfig


class DispatchSolveError(RuntimeError):
    """Raised for unavailable solvers, infeasibility, or non-optimal solves."""


@dataclass(frozen=True)
class DispatchState:
    """Realized state at the beginning of a dispatch interval."""

    battery_soc_mwh: float
    hydro_remaining_mwh: float
    conventional_output_mw: float = 0.0


@dataclass
class DispatchResult:
    """Optimized schedule, cost breakdown, and solver metadata."""

    schedule: pd.DataFrame
    total_cost_eur: float
    cost_breakdown_eur: dict[str, float]
    solver_status: str
    termination_condition: str
    formulation: str
    solve_time_s: float
    initial_state: DispatchState
    interval_hours: float

    @property
    def first_hour_state(self) -> DispatchState:
        """State to commit after executing only the first planned interval."""
        row = self.schedule.iloc[0]
        return DispatchState(
            battery_soc_mwh=float(row["soc_mwh"]),
            hydro_remaining_mwh=float(row["hydro_remaining_mwh"]),
            conventional_output_mw=float(row["conventional_mw"]),
        )

    @property
    def hydro_used_mwh(self) -> float:
        return float(self.schedule["hydro_mw"].sum() * self.interval_hours)

    @property
    def hydro_remaining_mwh(self) -> float:
        return float(self.schedule["hydro_remaining_mwh"].iloc[-1])

    @property
    def hydro_spilled_mwh(self) -> float:
        return float(self.schedule["hydro_spill_mw"].sum() * self.interval_hours)

    def to_dict(self) -> dict:
        return {
            "solver_status": self.solver_status,
            "termination_condition": self.termination_condition,
            "formulation": self.formulation,
            "total_cost_eur": self.total_cost_eur,
            "cost_breakdown_eur": self.cost_breakdown_eur,
            "hydro_used_mwh": self.hydro_used_mwh,
            "hydro_remaining_mwh": self.hydro_remaining_mwh,
            "hydro_spilled_mwh": self.hydro_spilled_mwh,
            "first_hour_state": {
                "battery_soc_mwh": self.first_hour_state.battery_soc_mwh,
                "hydro_remaining_mwh": self.first_hour_state.hydro_remaining_mwh,
                "conventional_output_mw": self.first_hour_state.conventional_output_mw,
            },
            "solve_time_s": self.solve_time_s,
            "schedule": self.schedule.reset_index().to_dict(orient="records"),
        }

    def export(self, directory: str | Path, name: str = "dispatch") -> tuple[Path, Path]:
        """Write hourly CSV and full JSON result; returns both paths."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        csv_path, json_path = out / f"{name}.csv", out / f"{name}.json"
        self.schedule.to_csv(csv_path, index_label="timestamp")
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return csv_path, json_path


def load_dispatch_config(path: str | Path) -> tuple[AssetConfig, dict]:
    """Read and validate the standalone dispatch YAML without changing base.yaml."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("currency") != "EUR":
        raise ValueError("dispatch config must be a YAML mapping with currency: EUR")
    a = raw["assets"]
    battery = BatteryConfig(**a["battery"])
    hydro = HydroConfig(**a["hydro"])
    assets = AssetConfig(
        interval_hours=float(a["interval_hours"]),
        solar=ResourceConfig(**a["solar"]),
        wind=ResourceConfig(**a["wind"]),
        hydro=hydro,
        conventional=ResourceConfig(
            capacity_mw=float(a["conventional"]["capacity_mw"]),
            marginal_cost_eur_mwh=float(a["conventional"]["marginal_cost_eur_mwh"]),
        ),
        emergency_grid=ResourceConfig(**a["emergency_grid"]),
        conventional_ramp_up_mw_per_interval=float(a["conventional_ramp_up_mw_per_interval"]),
        conventional_ramp_down_mw_per_interval=float(a["conventional_ramp_down_mw_per_interval"]),
        value_of_lost_load_eur_mwh=float(a["value_of_lost_load_eur_mwh"]),
        battery=battery,
    )
    settings = {
        "horizon_hours": int(raw.get("horizon_hours", 24)),
        "conventional_min_output_mw": float(a.get("conventional_min_output_mw", 0.0)),
        "terminal_soc_policy": raw.get("terminal_soc_policy", "preserve_initial"),
        "terminal_soc_target_mwh": raw.get("terminal_soc_target_mwh"),
        "simulated_resources": bool(raw.get("simulated_resources", True)),
        "battery_charging_policy": raw.get("battery_charging_policy", "renewable_only"),
        "allow_hydro_battery_charging": bool(raw.get("allow_hydro_battery_charging", True)),
        "notes": raw.get("notes", []),
    }
    if settings["horizon_hours"] < 1:
        raise ValueError("horizon_hours must be positive")
    if settings["battery_charging_policy"] not in {"renewable_only", "unrestricted"}:
        raise ValueError("battery_charging_policy must be renewable_only or unrestricted")
    if settings["allow_hydro_battery_charging"] and settings["battery_charging_policy"] != "renewable_only":
        raise ValueError("allow_hydro_battery_charging requires battery_charging_policy: renewable_only")
    if settings["terminal_soc_policy"] not in {"preserve_initial", "minimum", "none", "target"}:
        raise ValueError("terminal_soc_policy must be preserve_initial, minimum, target, or none")
    if settings["terminal_soc_policy"] == "target" and settings["terminal_soc_target_mwh"] is None:
        raise ValueError("terminal_soc_target_mwh is required for target policy")
    if not 0 <= settings["conventional_min_output_mw"] <= assets.conventional.capacity_mw:
        raise ValueError("conventional minimum output must be within its capacity")
    return assets, settings


def _series(name: str, values: Sequence[float], horizon: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != horizon:
        raise ValueError(f"{name} must be a one-dimensional sequence of length {horizon}")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} values must be finite and non-negative")
    return array


def optimize_dispatch(
    demand_forecast: Sequence[float],
    renewable_forecasts: Mapping[str, Sequence[float]],
    asset_config: AssetConfig,
    initial_state: DispatchState | None = None,
    *,
    timestamps: Sequence | None = None,
    hydro_inflow_mw: Sequence[float] | None = None,
    hydro_generation_limits_mw: Sequence[float] | None = None,
    solar_costs_eur_mwh: Sequence[float] | None = None,
    wind_costs_eur_mwh: Sequence[float] | None = None,
    conventional_costs_eur_mwh: Sequence[float] | None = None,
    emergency_costs_eur_mwh: Sequence[float] | None = None,
    conventional_min_output_mw: float = 0.0,
    terminal_soc_policy: str = "preserve_initial",
    terminal_soc_target_mwh: float | None = None,
    battery_charging_policy: str = "renewable_only",
    allow_hydro_battery_charging: bool = False,
    solver_name: str = "appsi_highs",
    commitment_schedule: Mapping[str, float] | None = None,
) -> DispatchResult:
    """Minimize operating cost across all periods in the supplied forecast.

    Cost arrays can represent time-varying prices. Missing prices use the
    corresponding YAML marginal costs. The correct unmet-load formulation is
    ``generation + unmet = demand + charging``: unserved energy reduces the
    amount of demand actually served and is penalized by VOLL.

    When ``commitment_schedule`` is supplied, the solve must cover exactly one
    interval. It first minimizes the total absolute MW change from the supplied
    planned dispatch setpoints, then minimizes the existing configured cost
    while holding that minimum change (within 1e-6 MW). This models a symmetric
    least-change real-time recourse rule; it adds no balancing price.
    """
    try:
        from pyomo.environ import (
            Binary,
            ConcreteModel,
            Constraint,
            ConstraintList,
            NonNegativeReals,
            Objective,
            SolverFactory,
            Var,
            minimize,
            value,
        )
        from pyomo.opt import TerminationCondition
    except ImportError as exc:
        raise DispatchSolveError(
            'Pyomo is not installed; install with `python -m pip install -e ".[optimization]"`.'
        ) from exc

    demand = np.asarray(demand_forecast, dtype=float)
    if demand.ndim != 1 or not len(demand) or not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("demand_forecast must be a non-empty finite, non-negative sequence")
    n, dt = len(demand), asset_config.interval_hours
    solar_cost = _series("solar costs", np.full(n, asset_config.solar.marginal_cost_eur_mwh) if solar_costs_eur_mwh is None else solar_costs_eur_mwh, n)
    wind_cost = _series("wind costs", np.full(n, asset_config.wind.marginal_cost_eur_mwh) if wind_costs_eur_mwh is None else wind_costs_eur_mwh, n)
    solar_av = _series("solar availability", renewable_forecasts["solar"], n)
    wind_av = _series("wind availability", renewable_forecasts["wind"], n)
    if (solar_av > asset_config.solar.capacity_mw + 1e-8).any():
        raise ValueError("solar availability exceeds configured capacity")
    if (wind_av > asset_config.wind.capacity_mw + 1e-8).any():
        raise ValueError("wind availability exceeds configured capacity")
    inflow = _series("hydro inflow", np.zeros(n) if hydro_inflow_mw is None else hydro_inflow_mw, n)
    hydro_limits = _series(
        "hydro generation limits",
        np.full(n, asset_config.hydro.capacity_mw)
        if hydro_generation_limits_mw is None else hydro_generation_limits_mw,
        n,
    )
    if (hydro_limits > asset_config.hydro.capacity_mw + 1e-8).any():
        raise ValueError("hydro generation limits cannot exceed configured turbine capacity")
    conv_cost = _series(
        "conventional costs",
        np.full(n, asset_config.conventional.marginal_cost_eur_mwh)
        if conventional_costs_eur_mwh is None
        else conventional_costs_eur_mwh,
        n,
    )
    emergency_cost = _series(
        "emergency costs",
        np.full(n, asset_config.emergency_grid.marginal_cost_eur_mwh)
        if emergency_costs_eur_mwh is None
        else emergency_costs_eur_mwh,
        n,
    )
    if (conv_cost < 0).any() or (emergency_cost < 0).any():
        raise ValueError(
            "generation prices must be non-negative for the LP exclusivity assumptions"
        )
    if terminal_soc_policy not in {"preserve_initial", "minimum", "none", "target"}:
        raise ValueError("invalid terminal SOC policy")
    if battery_charging_policy not in {"renewable_only", "unrestricted"}:
        raise ValueError("battery_charging_policy must be renewable_only or unrestricted")
    if allow_hydro_battery_charging and battery_charging_policy != "renewable_only":
        raise ValueError("allow_hydro_battery_charging requires battery_charging_policy='renewable_only'")
    if not 0 <= conventional_min_output_mw <= asset_config.conventional.capacity_mw:
        raise ValueError("conventional minimum output must be within capacity")

    commitment_variables = {
        "solar_mw": "solar",
        "wind_mw": "wind",
        "hydro_mw": "hydro",
        "conventional_mw": "conventional",
        "battery_charge_mw": "charge",
        "battery_discharge_mw": "discharge",
        "emergency_mw": "emergency",
        "unserved_mw": "unserved",
    }
    normalized_commitment: dict[str, float] | None = None
    if commitment_schedule is not None:
        if n != 1:
            raise ValueError("commitment recourse requires a one-interval solve")
        if set(commitment_schedule) != set(commitment_variables):
            missing = sorted(set(commitment_variables) - set(commitment_schedule))
            extra = sorted(set(commitment_schedule) - set(commitment_variables))
            raise ValueError(f"commitment setpoints mismatch; missing={missing}, extra={extra}")
        normalized_commitment = {key: float(commitment_schedule[key]) for key in commitment_variables}
        if not np.isfinite(list(normalized_commitment.values())).all() or any(
            val < 0 for val in normalized_commitment.values()
        ):
            raise ValueError("commitment setpoints must be finite and non-negative")

    state = initial_state or DispatchState(
        battery_soc_mwh=asset_config.battery.initial_soc_mwh,
        hydro_remaining_mwh=asset_config.hydro.initial_energy_budget_mwh,
    )
    b, h, c = asset_config.battery, asset_config.hydro, asset_config.conventional
    if not b.min_soc_mwh - 1e-8 <= state.battery_soc_mwh <= b.max_soc_mwh + 1e-8:
        raise ValueError("initial battery SOC is outside configured SOC bounds")
    if state.hydro_remaining_mwh < h.minimum_storage_mwh - 1e-8:
        raise ValueError("initial hydro remaining energy is below minimum reservoir storage")
    if (h.reservoir_capacity_mwh is not None
            and state.hydro_remaining_mwh > h.reservoir_capacity_mwh + 1e-8):
        raise ValueError("initial hydro remaining energy exceeds reservoir capacity")
    if not 0 <= state.conventional_output_mw <= c.capacity_mw:
        raise ValueError("initial conventional output must be within capacity")
    # HiGHS feasibility tolerances can return a boundary state a few ulps
    # outside its exact limit. Normalize only values already accepted by the
    # state validation tolerance so they remain valid terminal targets next hour.
    state = DispatchState(
        battery_soc_mwh=float(np.clip(state.battery_soc_mwh, b.min_soc_mwh, b.max_soc_mwh)),
        hydro_remaining_mwh=max(h.minimum_storage_mwh, float(state.hydro_remaining_mwh)),
        conventional_output_mw=float(state.conventional_output_mw),
    )

    m = ConcreteModel(name="GridDRIFT_MultiPeriodDispatch")
    m.T = range(n)
    m.solar = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, solar_av[t]))
    m.wind = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, wind_av[t]))
    m.solar_curtail = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, solar_av[t]))
    m.wind_curtail = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, wind_av[t]))
    m.solar_to_demand = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, solar_av[t]))
    m.wind_to_demand = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, wind_av[t]))
    m.solar_to_battery = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, solar_av[t]))
    m.wind_to_battery = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, wind_av[t]))
    m.hydro = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, hydro_limits[t]))
    m.hydro_to_demand = Var(m.T, domain=NonNegativeReals, bounds=(0, h.capacity_mw))
    m.hydro_to_battery = Var(m.T, domain=NonNegativeReals, bounds=(0, h.capacity_mw))
    m.hydro_remaining = Var(
        m.T, domain=NonNegativeReals,
        bounds=lambda _m, _t: (h.minimum_storage_mwh, h.reservoir_capacity_mwh),
    )
    m.hydro_spill = Var(m.T, domain=NonNegativeReals)
    m.conventional = Var(
        m.T, domain=NonNegativeReals, bounds=(conventional_min_output_mw, c.capacity_mw)
    )
    m.charge = Var(m.T, domain=NonNegativeReals, bounds=(0, b.charge_limit_mw))
    m.discharge = Var(m.T, domain=NonNegativeReals, bounds=(0, b.discharge_limit_mw))
    m.soc = Var(m.T, domain=NonNegativeReals, bounds=(b.min_soc_mwh, b.max_soc_mwh))
    m.emergency = Var(
        m.T, domain=NonNegativeReals, bounds=(0, asset_config.emergency_grid.capacity_mw)
    )
    m.unserved = Var(m.T, domain=NonNegativeReals, bounds=lambda _m, t: (0, demand[t]))

    m.renewable_solar = Constraint(
        m.T, rule=lambda _m, t: m.solar_to_demand[t] + m.solar_to_battery[t] + m.solar_curtail[t] == solar_av[t]
    )
    m.renewable_wind = Constraint(
        m.T, rule=lambda _m, t: m.wind_to_demand[t] + m.wind_to_battery[t] + m.wind_curtail[t] == wind_av[t]
    )
    m.solar_total = Constraint(m.T, rule=lambda _m, t: m.solar[t] == m.solar_to_demand[t] + m.solar_to_battery[t])
    m.wind_total = Constraint(m.T, rule=lambda _m, t: m.wind[t] == m.wind_to_demand[t] + m.wind_to_battery[t])
    m.hydro_allocation = Constraint(
        m.T, rule=lambda _m, t: m.hydro[t] == m.hydro_to_demand[t] + m.hydro_to_battery[t]
    )
    if battery_charging_policy == "renewable_only" and not allow_hydro_battery_charging:
        m.no_hydro_battery = Constraint(m.T, rule=lambda _m, t: m.hydro_to_battery[t] == 0)
    if battery_charging_policy == "renewable_only":
        m.renewable_charge = Constraint(
            m.T,
            rule=lambda _m, t: m.charge[t] == (
                m.solar_to_battery[t] + m.wind_to_battery[t]
                + (m.hydro_to_battery[t] if allow_hydro_battery_charging else 0)
            ),
        )
    else:
        m.renewable_charge = Constraint(
            m.T,
            rule=lambda _m, t: (
                m.solar_to_battery[t] + m.wind_to_battery[t] + m.hydro_to_battery[t]
                <= m.charge[t]
            ),
        )

    def hydro_state(_m, t):
        prior = state.hydro_remaining_mwh if t == 0 else m.hydro_remaining[t - 1]
        return (
            m.hydro_remaining[t]
            == prior + inflow[t] * dt - m.hydro[t] * dt - m.hydro_spill[t] * dt
        )

    m.hydro_state = Constraint(m.T, rule=hydro_state)
    if h.reservoir_capacity_mwh is None:
        m.no_hydro_spill = Constraint(m.T, rule=lambda _m, t: m.hydro_spill[t] == 0)

    def soc_state(_m, t):
        prior = state.battery_soc_mwh if t == 0 else m.soc[t - 1]
        return (
            m.soc[t]
            == prior
            + b.charge_efficiency * m.charge[t] * dt
            - m.discharge[t] * dt / b.discharge_efficiency
        )

    m.soc_state = Constraint(m.T, rule=soc_state)

    def balance(_m, t):
        # Unserved demand is a load slack on the left with demand/charging.
        return (
            m.solar_to_demand[t]
            + m.wind_to_demand[t]
            + m.hydro_to_demand[t]
            + m.conventional[t]
            + m.discharge[t]
            + m.emergency[t]
            + m.unserved[t]
            == demand[t] + m.charge[t]
        )

    m.power_balance = Constraint(m.T, rule=balance)

    def ramp_up(_m, t):
        prior = state.conventional_output_mw if t == 0 else m.conventional[t - 1]
        return m.conventional[t] - prior <= asset_config.conventional_ramp_up_mw_per_interval

    def ramp_down(_m, t):
        prior = state.conventional_output_mw if t == 0 else m.conventional[t - 1]
        return prior - m.conventional[t] <= asset_config.conventional_ramp_down_mw_per_interval

    m.ramp_up = Constraint(m.T, rule=ramp_up)
    m.ramp_down = Constraint(m.T, rule=ramp_down)

    if terminal_soc_policy != "none":
        if terminal_soc_policy == "preserve_initial":
            target_soc = state.battery_soc_mwh
        elif terminal_soc_policy == "minimum":
            target_soc = b.min_soc_mwh
        else:
            if terminal_soc_target_mwh is None:
                raise ValueError("terminal_soc_target_mwh is required for target policy")
            target_soc = float(terminal_soc_target_mwh)
        if not b.min_soc_mwh <= target_soc <= b.max_soc_mwh:
            raise ValueError("terminal SOC target is outside configured bounds")
        if terminal_soc_policy == "preserve_initial":
            m.terminal_soc = Constraint(expr=m.soc[n - 1] == target_soc)
        else:
            # A reserve is a floor, not an equality: do not force the optimizer
            # to dump usable energy merely to finish exactly at the reserve.
            m.terminal_soc = Constraint(expr=m.soc[n - 1] >= target_soc)

    # In the default domain (no must-run conventional output and a positive
    # battery throughput cost), internal cycling is strictly dominated: it
    # loses energy, incurs wear, and earns no export/negative-price revenue.
    # Must-run generation can create excess energy, and zero wear can create a
    # degenerate optimum, so use mode binaries in those exceptional cases.
    battery_exclusivity_binary = conventional_min_output_mw > 0 or b.degradation_cost_eur_mwh <= 0
    if battery_exclusivity_binary:
        m.battery_mode = Var(m.T, domain=Binary)
        m.charge_mode_limit = Constraint(
            m.T, rule=lambda _m, t: m.charge[t] <= b.charge_limit_mw * m.battery_mode[t]
        )
        m.discharge_mode_limit = Constraint(
            m.T, rule=lambda _m, t: m.discharge[t] <= b.discharge_limit_mw * (1 - m.battery_mode[t])
        )
    cost_terms = {
        "solar": sum(solar_cost[t] * m.solar[t] * dt for t in m.T),
        "wind": sum(wind_cost[t] * m.wind[t] * dt for t in m.T),
        "hydro": sum(h.marginal_cost_eur_mwh * m.hydro[t] * dt for t in m.T),
        "conventional": sum(conv_cost[t] * m.conventional[t] * dt for t in m.T),
        "emergency_grid": sum(emergency_cost[t] * m.emergency[t] * dt for t in m.T),
        "battery_degradation": sum(
            b.degradation_cost_eur_mwh * (m.charge[t] + m.discharge[t]) * dt for t in m.T
        ),
        "unserved_demand_voll": sum(
            asset_config.value_of_lost_load_eur_mwh * m.unserved[t] * dt for t in m.T
        ),
    }
    m.objective = Objective(expr=sum(cost_terms.values()), sense=minimize)

    if normalized_commitment is not None:
        # L1 deviations allow any necessary adjustment while preferring the
        # smallest total change across the same eight setpoints for both models.
        keys = tuple(commitment_variables)
        m.commitment_dev_pos = Var(keys, domain=NonNegativeReals)
        m.commitment_dev_neg = Var(keys, domain=NonNegativeReals)
        m.commitment_links = ConstraintList()
        for key, variable_name in commitment_variables.items():
            model_variable = getattr(m, variable_name)
            m.commitment_links.add(
                model_variable[0] - normalized_commitment[key]
                == m.commitment_dev_pos[key] - m.commitment_dev_neg[key]
            )
        m.commitment_deviation = sum(
            m.commitment_dev_pos[key] + m.commitment_dev_neg[key] for key in keys
        )
        m.objective.deactivate()
        m.commitment_objective = Objective(expr=m.commitment_deviation, sense=minimize)

    try:
        solver = SolverFactory(solver_name)
        if not solver.available(False):
            raise DispatchSolveError(
                f"Solver {solver_name!r} is unavailable; install highspy for HiGHS."
            )
        solve_started = time.perf_counter()
        solved = solver.solve(m)
        term = solved.solver.termination_condition
        if term != TerminationCondition.optimal:
            raise DispatchSolveError(
                f"Dispatch recourse adjustment did not solve optimally "
                f"(status={solved.solver.status}, termination={term})."
            )
        if normalized_commitment is not None:
            min_deviation = float(value(m.commitment_deviation))
            m.commitment_deviation_cap = Constraint(expr=m.commitment_deviation <= min_deviation + 1e-6)
            m.commitment_objective.deactivate()
            m.objective.activate()
            solved = solver.solve(m)
        if h.reservoir_capacity_mwh is not None:
            # Spill has no configured market cost. Select the least-spill
            # schedule among operating-cost-optimal schedules instead of
            # allowing a zero-cost degeneracy to discard usable reservoir water.
            min_operating_cost = float(value(m.objective))
            m.operating_cost_cap = Constraint(
                expr=m.objective.expr <= min_operating_cost + 1e-7
            )
            m.objective.deactivate()
            m.spill_objective = Objective(
                expr=sum(m.hydro_spill[t] * dt for t in m.T), sense=minimize
            )
            solved = solver.solve(m)
    except DispatchSolveError:
        raise
    except Exception as exc:
        if "feasible solution was not found" in str(exc).lower():
            raise DispatchSolveError(
                f"Dispatch did not solve optimally (infeasible or no feasible solution): {exc}"
            ) from exc
        raise DispatchSolveError(f"Dispatch solver {solver_name!r} failed: {exc}") from exc
    term = solved.solver.termination_condition
    if term != TerminationCondition.optimal:
        raise DispatchSolveError(
            f"Dispatch did not solve optimally (status={solved.solver.status}, termination={term})."
        )

    records = []
    for t in range(n):
        row = {
            "demand_mw": demand[t],
            "solar_available_mw": solar_av[t],
            "solar_mw": value(m.solar[t]),
            "solar_to_demand_mw": value(m.solar_to_demand[t]),
            "solar_to_battery_mw": value(m.solar_to_battery[t]),
            "solar_curtailment_mw": value(m.solar_curtail[t]),
            "wind_available_mw": wind_av[t],
            "wind_mw": value(m.wind[t]),
            "wind_to_demand_mw": value(m.wind_to_demand[t]),
            "wind_to_battery_mw": value(m.wind_to_battery[t]),
            "wind_curtailment_mw": value(m.wind_curtail[t]),
            "hydro_mw": value(m.hydro[t]),
            "hydro_to_demand_mw": value(m.hydro_to_demand[t]),
            "hydro_to_battery_mw": value(m.hydro_to_battery[t]),
            "hydro_inflow_mw": inflow[t],
            "hydro_remaining_mwh": max(0.0, float(value(m.hydro_remaining[t]))),
            "hydro_spill_mw": value(m.hydro_spill[t]),
            "hydro_generation_limit_mw": hydro_limits[t],
            "conventional_mw": value(m.conventional[t]),
            "battery_charge_mw": value(m.charge[t]),
            "battery_discharge_mw": value(m.discharge[t]),
            "soc_mwh": float(np.clip(value(m.soc[t]), b.min_soc_mwh, b.max_soc_mwh)),
            "emergency_mw": value(m.emergency[t]),
            "unserved_mw": value(m.unserved[t]),
            # Retain the exact rates used by the objective so downstream
            # reports can explain time-varying prices without reconstructing
            # them from a separate configuration snapshot.
            "solar_cost_rate_eur_mwh": solar_cost[t],
            "wind_cost_rate_eur_mwh": wind_cost[t],
            "hydro_cost_rate_eur_mwh": h.marginal_cost_eur_mwh,
            "conventional_cost_rate_eur_mwh": conv_cost[t],
            "emergency_cost_rate_eur_mwh": emergency_cost[t],
            "battery_degradation_rate_eur_mwh": b.degradation_cost_eur_mwh,
            "unserved_voll_rate_eur_mwh": asset_config.value_of_lost_load_eur_mwh,
        }
        if normalized_commitment is not None:
            row["commitment_policy"] = "minimum_l1_setpoint_adjustment_then_configured_cost"
            row["commitment_deviation_mw"] = float(sum(
                abs(float(row[key]) - normalized_commitment[key])
                for key in commitment_variables
            ))
        row["cost_solar_eur"] = solar_cost[t] * row["solar_mw"] * dt
        row["cost_wind_eur"] = wind_cost[t] * row["wind_mw"] * dt
        row["cost_hydro_eur"] = h.marginal_cost_eur_mwh * row["hydro_mw"] * dt
        row["cost_conventional_eur"] = conv_cost[t] * row["conventional_mw"] * dt
        row["cost_emergency_eur"] = emergency_cost[t] * row["emergency_mw"] * dt
        row["cost_battery_degradation_eur"] = (
            b.degradation_cost_eur_mwh
            * (row["battery_charge_mw"] + row["battery_discharge_mw"])
            * dt
        )
        row["cost_unserved_voll_eur"] = (
            asset_config.value_of_lost_load_eur_mwh * row["unserved_mw"] * dt
        )
        row["cost_total_eur"] = sum(
            v for k, v in row.items() if k.startswith("cost_") and k != "cost_total_eur"
        )
        records.append(row)
    if timestamps is None:
        index = pd.RangeIndex(n, name="interval")
    else:
        if len(timestamps) != n:
            raise ValueError("timestamps length must match forecast horizon")
        index = pd.Index(timestamps, name="timestamp")
    schedule = pd.DataFrame(records, index=index)
    breakdown = {
        "solar": float(schedule.cost_solar_eur.sum()),
        "wind": float(schedule.cost_wind_eur.sum()),
        "hydro": float(schedule.cost_hydro_eur.sum()),
        "conventional": float(schedule.cost_conventional_eur.sum()),
        "emergency_grid": float(schedule.cost_emergency_eur.sum()),
        "battery_degradation": float(schedule.cost_battery_degradation_eur.sum()),
        "unserved_demand_voll": float(schedule.cost_unserved_voll_eur.sum()),
    }
    total = float(value(m.objective))
    if not np.isclose(sum(breakdown.values()), total, rtol=1e-7, atol=1e-5):
        raise RuntimeError("reported cost breakdown does not reconcile with the objective")
    return DispatchResult(
        schedule=schedule,
        total_cost_eur=total,
        cost_breakdown_eur=breakdown,
        solver_status=str(solved.solver.status),
        termination_condition=str(term),
        formulation="MILP" if battery_exclusivity_binary else "LP",
        solve_time_s=time.perf_counter() - solve_started,
        initial_state=state,
        interval_hours=dt,
    )
