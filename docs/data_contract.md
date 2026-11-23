# Phase 1 data contract

GridDRIFT ingestion uses one timezone-aware UTC `DatetimeIndex` at the configured
target resolution. Demand is named `demand_mw` and is average MW over its source
interval; normalisation therefore aggregates sub-hourly intervals with a mean.
The Phase 2 MLP specifically requires a one-hour input grid because its horizon
is the next 24 hourly intervals.

The Phase 1 smoke scenario contains these simulated columns:

| Column | Meaning |
| --- | --- |
| `demand_mw` | Actual demand, MW |
| `solar_actual_mw` / `wind_actual_mw` | Simulated realized available renewable power, MW |
| `solar_forecast_mw` / `wind_forecast_mw` | Simulated forecast available power, MW |

Forecast and actual values are deliberately distinct. Later replay stages will
only expose an actual row after its simulated interval has completed.

For Phase 2 demand forecasts, an issue timestamp is the final observed hourly
timestamp. Its horizon 1 target is issue + 1 hour, and its horizon 24 target is
issue + 24 hours. This convention is stored in every forecast registry row.

ENTSO-E demand is expected as timestamp/value data. `normalize_hourly_demand`
converts timestamps to UTC, averages duplicate/sub-hourly intervals to hourly
MW, and rejects missing hourly periods rather than filling future information.
