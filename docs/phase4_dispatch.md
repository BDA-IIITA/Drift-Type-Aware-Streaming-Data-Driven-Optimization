# Phase 4: Multi-period dispatch

GridDRIFT's dispatch module is an independent, single-area 24-hour optimizer.
It accepts arrays, so a later replay integration can pass the existing MLP's
24 forecasts without making this module depend on forecasting or ADWIN. It
plans all hours together; the standalone demo uses deterministic simulated
profiles only.

## Formulation

For each interval `t`, the decision variables are renewable dispatch and
curtailment, hydro generation and remaining reservoir energy, conventional
output, battery charge/discharge and end-of-interval SOC, emergency supply,
and unserved demand. The objective sums the configured EUR/MWh marginal costs
times MWh, battery throughput degradation, and VOLL for unmet load.

The hourly balance is

```text
solar + wind + hydro + conventional + battery_discharge + emergency + unserved
    = demand + battery_charge
```

This is the standard load-shedding convention: unserved demand is the part of
load not met by physical generation. Renewable allocation is separate:

```text
solar_dispatch + solar_curtailment = solar_available
wind_dispatch  + wind_curtailment  = wind_available
```

The apparent placement of `unserved` on the left is deliberate and
mathematically equivalent to subtracting unserved load from demand; it is not a
source of electricity.

Battery energy uses end-of-interval SOC:

```text
SOC[t] = SOC[t-1] + eta_charge * charge[t] * dt
                     - discharge[t] * dt / eta_discharge
```

Hydro has its own state:

```text
reservoir[t] = reservoir[t-1] + inflow[t] * dt - hydro[t] * dt
```

Thus the hourly power limit and total MWh budget are distinct, and a caller can
pass the returned first-hour state into the next rolling solve. The default
terminal policy returns battery SOC to its opening level; alternatives are a
minimum reserve, a configurable reserve target (a lower bound, not an equality),
or no terminal constraint. Terminal policy is applied to planned SOC only; the
API separately exposes the state after the first hour for rolling execution.

The default model is an LP. With nonnegative energy costs, positive battery
throughput cost, no must-run conventional output, and curtailable renewables,
simultaneous charge/discharge is strictly dominated: it loses energy and adds
degradation without earning a sale or negative-price credit. If a positive
must-run minimum output is configured, or the throughput cost is zero (where
degenerate cycling may appear), binary battery modes are activated and HiGHS
solves a MILP. Negative energy costs are rejected for this LP contract. This
is not a unit-commitment model: a configured minimum output is assumed to apply
in every interval; startup, shutdown, minimum up/down times, and startup costs
are absent.

The default assets are deliberately a small representative system (MW-scale),
not a model of ENTSO-E's tens-of-thousands-of-MW national load. Solar and wind
profiles are explicitly simulated. Hydro inflows and time-varying conventional
and emergency marginal costs are optional input arrays. Generation/storage
costs are paid when energy is produced/charged; battery discharge does not pay
the source generation cost a second time.

## Reference review

Both supplied reference checkouts declare MIT licenses. The `economic-dispatch-pyomo`
builder documents that its code was reconstructed from a specification rather
than transcribed from an original implementation; its generator ramp and
storage relationships were useful formulation checks. The
`forecast-driven-bess-dispatch` builder implements a single-site LP with SOC,
curtailment, throughput cost, and rolling-state carry-over; those equations
were reviewed against its source and formulation docs. GridDRIFT uses a newly
written single-area implementation for this project's assets and power
balance; no reference source file was copied. This preserves attribution and
avoids importing either project's separate API/config model.

## Files and commands

- `configs/assets.yaml`: capacities, power limits, cost assumptions, VOLL,
  efficiencies, and horizon policy.
- `configs/scenarios/*.yaml`: six deterministic 24-hour scenarios, including a
  fair myopic comparison in `ramp_flexibility.yaml`.
- `src/griddrift/dispatch/engine.py`: validated API and one Pyomo model.
- `src/griddrift/dispatch/baselines.py`: physically constrained one-hour greedy
  baseline, carrying battery, hydro, and conventional states.
- `scripts/run_dispatch_demo.py`: CSV/JSON exports and PNG schedule/SOC plots.

Install Pyomo, HiGHS, and plotting support in the active environment:

```powershell
python -m pip install -e ".[optimization,dev]"
```

Run the complete demo:

```powershell
python scripts/run_dispatch_demo.py --assets configs/assets.yaml --scenarios configs/scenarios --output artifacts/dispatch/phase4-demo
```

Run dispatch-only tests, then the full existing project tests:

```powershell
python -m pytest tests/test_dispatch.py tests/test_dispatch_scenarios.py -q -p no:cacheprovider
python -m pytest -q -p no:cacheprovider
```

Each scenario directory contains its 24-row schedule CSV and full result JSON;
the `ramp_flexibility` directory also contains the greedy schedule. It includes
power balance, source generation, renewable curtailment, SOC, hydro budget,
emergency/unserved energy, and cost components. A schedule PNG shows dispatch
and battery/hydro states. The top-level `scenario_summary.csv/json` gives cost,
hydro use, emergency/unserved totals, solver status, and the greedy comparison.

The greedy baseline solves one interval at a time with the same asset costs,
initial states, forecasts, and ramp/resource limits. For the corrected cost
comparison, both greedy and multi-period schedules have an unconstrained final
SOC and receive the same terminal stored-energy settlement: SOC deficits are
valued at the input electricity needed to restore them (including charge
losses), while excess SOC is valued at its deliverable energy (including
discharge losses). The fair comparison is written as
`multi_period_comparable.csv` and its adjusted costs appear in the scenario
summary. The default optimizer schedule remains a separate terminal-preserving
plan. This remains a myopic-versus-horizon-aware comparison, not a claim that
the policies have identical foresight.
