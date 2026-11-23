# Phase 2: real-data forecasting contract

`fetch_entsoe_actual_load` is cache-first. It returns a successful local cache
without inspecting `ENTSOE_API_TOKEN`; on an uncached request it reads that
environment variable only to initialize `entsoe-py`. The token is never written,
included in a filename, or added to an exception message.

The ENTSO-E client can return native sub-hourly values. GridDRIFT's configured
`data.resolution` determines the target grid, and power values aggregate by
mean. The supplied MLP workflow requires `1h`; changing the ingestion target
does not turn a non-hourly series into a valid 24-hour MLP input.

The downloader treats date-only `end_date` as inclusive. Thus the default
`2024-12-31` is sent to the client with the exclusive bound
`2025-01-01T00:00Z`. Missing requested intervals cause an error and do not
create a cache entry.

Training constructs one direct target vector for each issue time. Features use
only observations at or before the issue timestamp; labels are the next 24
hours. The chronological validation suffix is never used to fit either scaler
or MLP. The model artifact contains the estimator, both fitted scalers, feature
specification, version, train boundary, and validation MAE/RMSE.
