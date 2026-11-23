# Phase 5: integrated rolling replay

## Hourly event order

The replay begins at the first timestamp after the saved model's training-label
boundary, with the prior historical demand available as observed context. At
every issue time `t`, it:

1. Uses only demand through `t` to issue one simultaneous 24-output MLP forecast
   for `t+1` through `t+24`.
2. Issues separate calendar-pattern solar/wind availability forecasts for those
   same timestamps. These forecasts do not read the hidden simulated actuals.
3. Solves one 24-interval Pyomo plan using current SOC, hydro budget, and
   conventional output. Only the first future interval is due next.
4. When the next hour arrives, reveals its demand and generated resource
   outcomes, evaluates the fixed-24-hour forecast issued 24 hours earlier,
   updates persistent ADWIN, and optionally attempts validated adaptation.
5. Re-solves one interval using the actual newly revealed demand/resources as
   balancing recourse, charges its realized cost, carries forward physical
   state, and immediately forecasts/plans the next 24 hours.

The dashboard therefore does not animate the MLP predictions arriving one by
one. Historical actuals are solid green; all 24 future predictions are a dashed
blue curve, separated by a vertical current-time marker. The plan table and
forecast have identical future UTC timestamps. The chart's demand remains in
original MW; the schedule also shows the original forecast and the explicitly
scaled dispatch demand. The default representative-grid factor is 0.001 and is
configurable in `configs/integration.yaml` and the UI.

## Dispatch and accounting

The default `renewable_only` policy adds source allocations:

```text
solar_to_demand + solar_to_battery + solar_curtail = solar_available
wind_to_demand  + wind_to_battery  + wind_curtail  = wind_available
battery_charge = solar_to_battery + wind_to_battery
```

The demand balance uses direct renewable allocation, not total renewable
generation; this prevents double counting. Conventional, hydro, and emergency
generation cannot charge the battery under the default policy. The optional
`unrestricted` comparison mode permits other sources, while renewable energy
sent to storage remains itemized. Renewable generation costs are paid in the
charging interval; later battery discharge pays only its throughput/degradation
cost. SOC efficiency is applied on the stored-energy side for charge and the
withdrawn-energy side for discharge.

Planned 24-hour cost is an objective for forecasted conditions. Realized hourly
cost is calculated separately after actual observations arrive. Actual
reconciliation may deviate from the planned first row, but remains subject to
the same ramp, SOC, hydro, power, and emergency/unserved constraints. Planned and
realized schedules are kept separately. The current offline run's displayed
accumulated realized cost covers already completed intervals; the displayed
plan cost covers the new current 24-hour forecast.

Hydro has no automatic daily refill. It only changes through actual dispatch
and any explicit inflow supplied to the optimizer. Battery SOC, hydro energy,
and generator output carry between every realized interval. The current
integrated scenario configures zero hydro inflow.

## Drift and forecasting

The static baseline is an untouched Phase 2 MLP. The adaptive branch begins as a
copy and receives the same observed demand. Both forecast registries issue
24-hour trajectories; only the prediction whose horizon is exactly 24 hours is
fed to ADWIN when its target actual arrives. The adaptive policy uses the
existing Phase 3.5 validated candidate gate and cooldown. Model adaptation is
separate from dispatch: dispatch is re-solved every hour regardless of drift.
ADWIN detections and before/after error levels are logged separately from
adaptation attempts. The dashboard does not claim adaptive improvement; static
forecasting is selected initially because Phase 3.5 favored the frozen model.

## Validation performed

- Full existing and new test suite: `42 passed`.
- Targeted Ruff checks passed for the integration, dashboard, dispatch, CLI, and
  added test modules.
- Streamlit `AppTest` rendered the app without exceptions and a Step click
  advanced the simulation without exceptions.
- A 24-hour DE-LU cached-data replay produced a 24-value forecast and 24-row
  plan, then 24 realized intervals. It saved outputs under
  `artifacts/integrated/offline-demo/`.
- Dispatch examples were rerun to
  `artifacts/dispatch/phase5-fair-comparison/`. For `ramp_flexibility`, both
  comparable schedules have unconstrained terminal SOC and share one
  terminal-energy valuation. The greedy adjusted cost was EUR 58,998.25 and
  multi-period adjusted cost EUR 44,285.45, a saving of EUR 14,712.80 (24.94%)
  on this synthetic scenario. It is a single illustrative scenario, not a
  general performance guarantee.

## Current limits

- Renewable actuals and forecasts are simulated, not ENTSO-E measurements or
  external weather-service forecasts.
- Dispatch uses a representative scaled grid, not national-scale generation
  assets. Forecast evaluation remains at original demand scale.
- The implemented adaptive option uses the existing validated policy; the
  four-arm static/adaptive × greedy/optimizer batch evaluation remains future
  work. The dashboard includes the static/adaptive error comparison and the
  integrated horizon-aware dispatch.
- Candidate-model adaptation is intentionally not a page-refresh operation and
  only occurs on an eligible alarm during adaptive replay. Starting with the
  static policy monitors ADWIN but does not adapt weights.
- The optimizer has no unit commitment/startup logic, transmission network,
  renewable forecast learning, or configured hydro inflow in the integrated
  demo.
