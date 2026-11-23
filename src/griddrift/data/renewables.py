"""Preprocess hourly ENTSO-E renewable forecasts and realized generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "demand_mw",
    "solar_actual_mw",
    "wind_onshore_actual_mw",
    "wind_offshore_actual_mw",
    "solar_day_ahead_forecast_mw",
    "wind_onshore_day_ahead_forecast_mw",
    "wind_offshore_day_ahead_forecast_mw",
    "day_ahead_price_eur_mwh",
)

NUMERIC_COLUMNS = (
    "demand_mw",
    "solar_actual_mw",
    "wind_onshore_actual_mw",
    "wind_offshore_actual_mw",
    "wind_actual_mw",
    "solar_day_ahead_forecast_mw",
    "wind_onshore_day_ahead_forecast_mw",
    "wind_offshore_day_ahead_forecast_mw",
    "day_ahead_price_eur_mwh",
    "hydro_run_of_river_and_pondage_actual_mw",
    "hydro_water_reservoir_actual_mw",
    "hydro_pumped_storage_actual_generation_mw",
    "hydro_reported_actual_mw",
)

PHYSICAL_NONNEGATIVE_COLUMNS = tuple(
    column for column in NUMERIC_COLUMNS if column != "day_ahead_price_eur_mwh"
)

ISSUE_TIME_CANDIDATES = {
    "solar_forecast_issue_timestamp_utc": (
        "solar_forecast_issue_timestamp_utc",
        "solar_forecast_issue_timestamp",
        "solar_issue_timestamp_utc",
        "solar_issue_timestamp",
        "forecast_issue_timestamp_utc",
        "forecast_issue_timestamp",
        "issue_timestamp_utc",
        "issue_timestamp",
    ),
    "wind_forecast_issue_timestamp_utc": (
        "wind_forecast_issue_timestamp_utc",
        "wind_forecast_issue_timestamp",
        "wind_issue_timestamp_utc",
        "wind_issue_timestamp",
        "forecast_issue_timestamp_utc",
        "forecast_issue_timestamp",
        "issue_timestamp_utc",
        "issue_timestamp",
    ),
}


class RenewableDataError(ValueError):
    """Raised when the renewable source cannot safely be normalized."""


class RenewableResourceProvider:
    """As-of resource inputs for historical replay, separate forecasts from actuals.

    When the source has no issue timestamps, the configured assumption treats
    a target day's forecast as available at 00:00 UTC on the previous day.
    Forecast values are never filled from actual generation; missing or not-yet
    available forecasts are returned as zero with explicit flags.
    """

    ISSUE_POLICY_PREVIOUS_DAY_00_UTC = "previous_day_00_utc_assumption"

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        *,
        issue_time_policy: str = ISSUE_POLICY_PREVIOUS_DAY_00_UTC,
    ) -> None:
        if issue_time_policy != self.ISSUE_POLICY_PREVIOUS_DAY_00_UTC:
            raise RenewableDataError(f"Unsupported issue-time policy: {issue_time_policy}")
        frame = _read_source(source)
        required = {
            "target_timestamp_utc",
            "solar_actual_mw",
            "wind_actual_mw",
            "solar_day_ahead_forecast_mw",
            "wind_day_ahead_forecast_mw",
        }
        if not required.issubset(frame.columns):
            raise RenewableDataError(
                f"Renewable replay table is missing columns: {sorted(required - set(frame.columns))}"
            )
        index = pd.DatetimeIndex(pd.to_datetime(frame["target_timestamp_utc"], utc=True, errors="coerce"))
        if index.isna().any() or index.has_duplicates or not index.is_monotonic_increasing:
            raise RenewableDataError("Renewable replay targets must be valid, unique, and chronological")
        frame = frame.drop(columns="target_timestamp_utc").copy()
        frame.index = index
        frame.index.name = "target_timestamp_utc"
        for column in required - {"target_timestamp_utc"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        issue_columns = (
            "solar_forecast_issue_timestamp_utc",
            "wind_forecast_issue_timestamp_utc",
        )
        for column in issue_columns:
            if column in frame:
                raw = frame[column]
                parsed = pd.to_datetime(raw, utc=True, errors="coerce")
                if (raw.notna() & parsed.isna()).any():
                    raise RenewableDataError(f"Invalid issue timestamps in {column!r}")
                frame[column] = parsed
            else:
                frame[column] = pd.NaT
        self._frame = frame
        self.issue_time_policy = issue_time_policy

    @property
    def target_timestamps(self) -> pd.DatetimeIndex:
        """Available target hours, exposed for replay coverage validation."""
        return self._frame.index.copy()

    @staticmethod
    def _utc_timestamp(value: object) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")

    def forecast_at(
        self,
        issue_time: object,
        target_timestamps: pd.DatetimeIndex,
        *,
        scale: float,
    ) -> pd.DataFrame:
        """Return only resource forecasts available under the issue-time rule."""
        issue = self._utc_timestamp(issue_time)
        targets = pd.DatetimeIndex(target_timestamps)
        if targets.tz is None:
            raise RenewableDataError("Resource target timestamps must be timezone-aware")
        targets = targets.tz_convert("UTC")
        if targets.has_duplicates or not targets.is_monotonic_increasing:
            raise RenewableDataError("Resource targets must be unique and chronological")
        if not np.isfinite(scale) or scale <= 0:
            raise RenewableDataError("representative-grid scale must be finite and positive")
        missing_targets = targets.difference(self._frame.index)
        if len(missing_targets):
            raise RenewableDataError(
                f"Renewable source does not cover all requested target hours; first missing: {missing_targets[0]}"
            )
        rows = self._frame.reindex(targets)
        result = pd.DataFrame(index=targets)
        result.index.name = "target_timestamp_utc"
        for resource in ("solar", "wind"):
            forecast_column = f"{resource}_day_ahead_forecast_mw"
            issue_column = f"{resource}_forecast_issue_timestamp_utc"
            raw_values = rows[forecast_column]
            issue_stamps = rows[issue_column]
            assumed_issue = targets.normalize() - pd.Timedelta(days=1)
            effective_issue = issue_stamps.where(issue_stamps.notna(), assumed_issue)
            available_asof = effective_issue <= issue
            missing_value = raw_values.isna()
            used_values = raw_values.where(available_asof & ~missing_value, 0.0)
            result[f"{resource}_available_mw"] = used_values.to_numpy(dtype=float) * scale
            result[f"{resource}_forecast_issue_time_assumed"] = issue_stamps.isna().to_numpy()
            result[f"{resource}_forecast_available_asof"] = available_asof.to_numpy()
            result[f"{resource}_forecast_value_missing"] = missing_value.to_numpy()
        result["issue_time_policy"] = self.issue_time_policy
        result["forecast_issue_timestamp_utc"] = issue
        return result

    def actual_at(self, timestamp: object, *, scale: float) -> dict[str, object]:
        """Return the exact observed renewable actuals for one revealed hour."""
        target = self._utc_timestamp(timestamp)
        if target not in self._frame.index:
            raise RenewableDataError(f"Renewable actual data does not cover revealed hour {target}")
        if not np.isfinite(scale) or scale <= 0:
            raise RenewableDataError("representative-grid scale must be finite and positive")
        row = self._frame.loc[target]
        solar, wind = row["solar_actual_mw"], row["wind_actual_mw"]
        if pd.isna(solar) or pd.isna(wind):
            raise RenewableDataError(f"Renewable actual observation is missing at revealed hour {target}")
        if solar < 0 or wind < 0:
            raise RenewableDataError(f"Negative renewable actual at revealed hour {target}")
        return {
            "solar_actual_mw": float(solar) * scale,
            "wind_actual_mw": float(wind) * scale,
            "source_timestamp_utc": target,
            "source": "ENTSO-E realized generation",
        }


def _read_source(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source, low_memory=False)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    if invalid.any():
        raise RenewableDataError(
            f"Column {column!r} has {int(invalid.sum())} nonnumeric nonmissing values"
        )
    return numeric.astype(float)


def _issue_timestamp(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    source_column = next((column for column in candidates if column in frame.columns), None)
    if source_column is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    raw = frame[source_column]
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if (raw.notna() & parsed.isna()).any():
        raise RenewableDataError(f"Invalid forecast issue timestamps in {source_column!r}")
    return parsed


def preprocess_entsoe_renewables(
    source: str | Path | pd.DataFrame,
    *,
    wind_sum_tolerance_mw: float = 1e-6,
) -> pd.DataFrame:
    """Return a complete hourly UTC table while preserving missing observations.

    Forecast issue times are copied and UTC-normalized when present. If the
    source does not provide them, the corresponding output fields remain null;
    this function never infers publication times from the ``day_ahead`` label.
    Missing hourly rows are inserted with null measurements and marked through
    ``source_row_present``. No imputation or forecast-to-actual substitution is
    performed.
    """
    frame = _read_source(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RenewableDataError(f"Missing required source columns: {missing}")
    if not np.isfinite(wind_sum_tolerance_mw) or wind_sum_tolerance_mw < 0:
        raise ValueError("wind_sum_tolerance_mw must be finite and nonnegative")

    raw_timestamps = frame["timestamp"]
    timestamps = pd.to_datetime(raw_timestamps, utc=True, errors="coerce")
    invalid_timestamps = timestamps.isna()
    if invalid_timestamps.any():
        raise RenewableDataError(
            f"timestamp contains {int(invalid_timestamps.sum())} missing or invalid values"
        )
    if timestamps.duplicated().any():
        raise RenewableDataError(
            f"timestamp contains {int(timestamps.duplicated().sum())} duplicate rows"
        )
    off_hour = timestamps != timestamps.dt.floor("h")
    if off_hour.any():
        raise RenewableDataError(f"timestamp contains {int(off_hour.sum())} non-hourly values")

    source_indexed = frame.drop(columns="timestamp").copy()
    source_indexed.index = pd.DatetimeIndex(timestamps, name="target_timestamp_utc")
    source_indexed = source_indexed.sort_index()
    complete_hours = pd.date_range(
        source_indexed.index.min(), source_indexed.index.max(), freq="h", name="target_timestamp_utc"
    )
    clean = source_indexed.reindex(complete_hours)
    clean["source_row_present"] = clean.index.isin(source_indexed.index)

    for column in NUMERIC_COLUMNS:
        if column in clean.columns:
            clean[column] = _numeric_column(clean, column)
    for column in PHYSICAL_NONNEGATIVE_COLUMNS:
        if column not in clean.columns:
            continue
        negative = clean[column].lt(0)
        if negative.any():
            raise RenewableDataError(
                f"{column!r} contains {int(negative.sum())} negative physical MW values"
            )

    wind_components = clean[["wind_onshore_actual_mw", "wind_offshore_actual_mw"]]
    calculated_actual_wind = wind_components.sum(axis=1, min_count=2)
    if "wind_actual_mw" in clean.columns:
        supplied_actual_wind = clean["wind_actual_mw"]
        comparable = supplied_actual_wind.notna() & calculated_actual_wind.notna()
        inconsistent = comparable & (
            (supplied_actual_wind - calculated_actual_wind).abs() > wind_sum_tolerance_mw
        )
        if inconsistent.any():
            raise RenewableDataError(
                "wind_actual_mw disagrees with onshore plus offshore actual generation "
                f"in {int(inconsistent.sum())} rows"
            )
        actual_wind = supplied_actual_wind.combine_first(calculated_actual_wind)
    else:
        actual_wind = calculated_actual_wind

    wind_forecast = clean[
        ["wind_onshore_day_ahead_forecast_mw", "wind_offshore_day_ahead_forecast_mw"]
    ].sum(axis=1, min_count=2)
    result = pd.DataFrame(index=complete_hours)
    result.index.name = "target_timestamp_utc"
    result["source_row_present"] = clean["source_row_present"].astype(bool)
    result["demand_actual_mw"] = clean["demand_mw"]
    result["solar_actual_mw"] = clean["solar_actual_mw"]
    result["wind_onshore_actual_mw"] = clean["wind_onshore_actual_mw"]
    result["wind_offshore_actual_mw"] = clean["wind_offshore_actual_mw"]
    result["wind_actual_mw"] = actual_wind
    result["solar_day_ahead_forecast_mw"] = clean["solar_day_ahead_forecast_mw"]
    result["wind_onshore_day_ahead_forecast_mw"] = clean[
        "wind_onshore_day_ahead_forecast_mw"
    ]
    result["wind_offshore_day_ahead_forecast_mw"] = clean[
        "wind_offshore_day_ahead_forecast_mw"
    ]
    result["wind_day_ahead_forecast_mw"] = wind_forecast
    result["day_ahead_market_price_eur_mwh"] = clean["day_ahead_price_eur_mwh"]
    # Preserve source hydro categories only as historical references. In
    # particular, reservoir generation is an output and must never be treated
    # as reservoir inflow or stored-energy state.
    for column in (
        "hydro_run_of_river_and_pondage_actual_mw",
        "hydro_water_reservoir_actual_mw",
        "hydro_pumped_storage_actual_generation_mw",
        "hydro_reported_actual_mw",
    ):
        if column in clean:
            result[column] = clean[column]
            result[f"{column.removesuffix('_mw')}_missing"] = clean[column].isna()
    for output_column, candidates in ISSUE_TIME_CANDIDATES.items():
        result[output_column] = _issue_timestamp(clean, candidates)

    result["demand_actual_missing"] = result["demand_actual_mw"].isna()
    result["solar_actual_missing"] = result["solar_actual_mw"].isna()
    result["wind_actual_missing"] = result["wind_actual_mw"].isna()
    result["wind_onshore_actual_missing"] = result["wind_onshore_actual_mw"].isna()
    result["wind_offshore_actual_missing"] = result["wind_offshore_actual_mw"].isna()
    result["solar_forecast_missing"] = result["solar_day_ahead_forecast_mw"].isna()
    result["wind_forecast_missing"] = result["wind_day_ahead_forecast_mw"].isna()
    result["day_ahead_market_price_missing"] = result["day_ahead_market_price_eur_mwh"].isna()
    result["any_renewable_forecast_missing"] = (
        result["solar_forecast_missing"] | result["wind_forecast_missing"]
    )
    result.insert(0, "target_timestamp_utc", result.index)
    return result.reset_index(drop=True)


def preprocess_entsoe_renewables_csv(
    source_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Preprocess a source CSV and write the normalized hourly CSV."""
    source = Path(source_path)
    destination = Path(output_path)
    metadata_path = destination.with_suffix(".metadata.json")
    if (destination.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing renewable output or metadata: {destination}"
        )
    result = preprocess_entsoe_renewables(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    metadata = {
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_file": str(destination),
        "rows": len(result),
        "target_first_utc": result["target_timestamp_utc"].iloc[0].isoformat(),
        "target_last_utc": result["target_timestamp_utc"].iloc[-1].isoformat(),
        "source_row_count": int(result["source_row_present"].sum()),
        "source_hour_gaps_inserted": int((~result["source_row_present"]).sum()),
        "units": {
            "demand_and_generation": "MW; source units retained",
            "day_ahead_market_price": "EUR/MWh; source units retained",
            "target_timestamp": "UTC",
        },
        "wind_aggregation": {
            "actual": "onshore plus offshore validated against source wind_actual_mw when present",
            "forecast": "onshore plus offshore, only when both component forecasts are present",
        },
        "forecast_issue_times": {
            "solar_available_rows": int(
                result["solar_forecast_issue_timestamp_utc"].notna().sum()
            ),
            "wind_available_rows": int(
                result["wind_forecast_issue_timestamp_utc"].notna().sum()
            ),
            "note": "Source issue timestamps are preserved when present; none are inferred.",
        },
        "missing_forecast_rows": {
            "solar": int(result["solar_forecast_missing"].sum()),
            "wind": int(result["wind_forecast_missing"].sum()),
        },
        "geography_note": "DE-LU is inferred from source filename; source contains no region column.",
        "no_imputation": True,
        "actual_generation_is_not_used_as_forecast": True,
        "hydro_generation_fields_are_reference_only": True,
        "hydro_generation_is_not_reservoir_inflow_or_storage": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result
