"""Historical PECD reservoir-hydro references and modeled inflow scenarios."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

RESERVOIR_TYPE = "Reservoir"
TURBINE_CAPACITY_LABEL = "Total turbining capacity (MW)"
RESERVOIR_CAPACITY_LABEL = "Reservoir capacity (GWh)"
RESERVOIR_INFLOW_COLUMN = "Cumulated inflow into reservoirs per week in GWh"
MWH_PER_GWH = 1_000.0
HOURS_PER_WEEK = 168.0


class HydroDataError(ValueError):
    """Raised when PECD reservoir data cannot be safely interpreted."""


@dataclass(frozen=True)
class ReservoirParameters:
    zone: str
    technology: str
    turbine_capacity_mw: float
    reservoir_capacity_gwh: float
    reservoir_capacity_mwh: float
    requested_initial_filling_fraction: float
    initial_filling_fraction: float
    initial_reservoir_energy_mwh: float
    capacity_reference_year: str
    initial_state_source: str
    source_file: str
    minimum_reservoir_energy_mwh: float = 0.0


def load_reservoir_parameters_json(source: str | Path) -> ReservoirParameters:
    """Load the Stage 3 parameter artifact without altering its assumptions."""
    with Path(source).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    try:
        parameters = ReservoirParameters(**raw)
    except (TypeError, KeyError) as exc:
        raise HydroDataError(f"Invalid reservoir parameter artifact: {source}") from exc
    if parameters.zone != "DE00" or parameters.technology != RESERVOIR_TYPE:
        raise HydroDataError("Only DE00 Reservoir parameters are supported by this integration")
    if parameters.reservoir_capacity_mwh <= 0 or not (
        0 <= parameters.initial_reservoir_energy_mwh <= parameters.reservoir_capacity_mwh
    ):
        raise HydroDataError("Reservoir parameter energy values are outside physical bounds")
    if not 0 <= parameters.minimum_reservoir_energy_mwh <= parameters.initial_reservoir_energy_mwh:
        raise HydroDataError("Minimum reservoir energy must be between zero and the initial energy")
    return parameters


def load_assumed_hourly_inflow_series(source: str | Path) -> pd.Series:
    """Load the Stage 3 modeled hourly MWh inflow series as a UTC-indexed Series."""
    frame = pd.read_csv(source, low_memory=False)
    value_column = (
        "synthetic_2025_hydro_inflow_mwh"
        if "synthetic_2025_hydro_inflow_mwh" in frame.columns
        else "assumed_inflow_mwh_per_hour"
    )
    required = {"target_timestamp_utc", value_column}
    if not required.issubset(frame.columns):
        raise HydroDataError(
            f"Hourly inflow scenario is missing columns: {sorted(required - set(frame.columns))}"
        )
    index = pd.DatetimeIndex(pd.to_datetime(frame.target_timestamp_utc, utc=True, errors="coerce"))
    values = pd.to_numeric(frame[value_column], errors="coerce").astype(float)
    if index.isna().any() or index.has_duplicates or not index.is_monotonic_increasing:
        raise HydroDataError("Hourly inflow timestamps must be valid, unique, and chronological")
    if values.isna().any() or not values.map(math.isfinite).all() or (values < 0).any():
        raise HydroDataError("Hourly inflow values must be finite and nonnegative")
    if len(index) > 1 and not (index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise HydroDataError("Hourly inflow scenario must have exactly one-hour spacing")
    return pd.Series(values.to_numpy(), index=index, name=value_column)


def _read(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source, low_memory=False)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_de00_reservoir_parameters(
    capacity_source: str | Path | pd.DataFrame,
    *,
    initial_filling_fraction: float = 0.5,
    minimum_reservoir_energy_mwh: float = 0.0,
) -> ReservoirParameters:
    """Read only the DE00 Reservoir technology and clamp its initial fill assumption."""
    if not math.isfinite(initial_filling_fraction) or not math.isfinite(minimum_reservoir_energy_mwh):
        raise HydroDataError("initial filling fraction and minimum reservoir energy must be finite")
    frame = _read(capacity_source)
    required = {"zone", "type", "variable", "value"}
    if not required.issubset(frame.columns):
        raise HydroDataError(f"Capacity table is missing columns: {sorted(required - set(frame.columns))}")
    reservoir = frame.loc[
        frame["zone"].astype(str).eq("DE00") & frame["type"].astype(str).eq(RESERVOIR_TYPE)
    ]

    def parameter(label: str) -> float:
        selected = reservoir.loc[reservoir["variable"].eq(label), "value"]
        values = pd.to_numeric(selected, errors="coerce").dropna()
        if len(values) != 1 or not math.isfinite(float(values.iloc[0])):
            raise HydroDataError(f"Expected exactly one valid DE00 Reservoir value for {label!r}")
        if float(values.iloc[0]) <= 0:
            raise HydroDataError(f"DE00 Reservoir parameter {label!r} must be positive")
        return float(values.iloc[0])

    turbine_mw = parameter(TURBINE_CAPACITY_LABEL)
    capacity_gwh = parameter(RESERVOIR_CAPACITY_LABEL)
    effective_fraction = min(1.0, max(0.0, float(initial_filling_fraction)))
    initial_energy_mwh = capacity_gwh * MWH_PER_GWH * effective_fraction
    minimum_energy_mwh = float(minimum_reservoir_energy_mwh)
    if not 0 <= minimum_energy_mwh <= initial_energy_mwh:
        raise HydroDataError("minimum reservoir energy must be within zero and configured initial energy")
    source_name = (
        str(Path(capacity_source))
        if not isinstance(capacity_source, pd.DataFrame)
        else "in-memory capacity data"
    )
    return ReservoirParameters(
        zone="DE00",
        technology=RESERVOIR_TYPE,
        turbine_capacity_mw=turbine_mw,
        reservoir_capacity_gwh=capacity_gwh,
        reservoir_capacity_mwh=capacity_gwh * MWH_PER_GWH,
        requested_initial_filling_fraction=float(initial_filling_fraction),
        initial_filling_fraction=effective_fraction,
        initial_reservoir_energy_mwh=initial_energy_mwh,
        minimum_reservoir_energy_mwh=minimum_energy_mwh,
        capacity_reference_year="not specified in source CSV; historical PECD reference, not verified 2025 capacity",
        initial_state_source="scenario assumption; Germany DE00 level records are absent from the supplied levels file",
        source_file=source_name,
    )


def build_seasonal_reservoir_inflow_profile(
    inflow_source: str | Path | pd.DataFrame,
    *,
    zone: str = "DE00",
) -> pd.DataFrame:
    """Aggregate historical reservoir weekly totals by ISO-style week number using the median."""
    frame = _read(inflow_source)
    required = {"zone", "week", "year", RESERVOIR_INFLOW_COLUMN}
    if not required.issubset(frame.columns):
        raise HydroDataError(f"Inflow table is missing columns: {sorted(required - set(frame.columns))}")
    selected = frame.loc[frame["zone"].astype(str).eq(zone), ["zone", "week", "year", RESERVOIR_INFLOW_COLUMN]].copy()
    if selected.empty:
        raise HydroDataError(f"No hydro inflows found for zone {zone!r}")
    selected["week"] = pd.to_numeric(selected["week"], errors="coerce")
    selected["year"] = pd.to_numeric(selected["year"], errors="coerce")
    selected[RESERVOIR_INFLOW_COLUMN] = pd.to_numeric(
        selected[RESERVOIR_INFLOW_COLUMN], errors="coerce"
    )
    if selected[["week", "year", RESERVOIR_INFLOW_COLUMN]].isna().any().any():
        raise HydroDataError("Selected reservoir inflow rows contain missing/non-numeric week, year, or GWh")
    if not selected["week"].between(1, 53).all():
        raise HydroDataError("PECD week values must be in the range 1..53")
    if selected.duplicated(["zone", "week", "year"]).any():
        raise HydroDataError("Duplicate zone/week/year rows in reservoir inflow source")
    if (selected[RESERVOIR_INFLOW_COLUMN] < 0).any():
        raise HydroDataError("Reservoir weekly inflow cannot be negative")

    profile = (
        selected.groupby("week", as_index=False)
        .agg(
            median_weekly_inflow_gwh=(RESERVOIR_INFLOW_COLUMN, "median"),
            historical_year_count=("year", "nunique"),
            historical_sample_count=("year", "size"),
        )
        .rename(columns={"week": "iso_week"})
        .sort_values("iso_week")
        .reset_index(drop=True)
    )
    profile["median_weekly_inflow_mwh"] = profile["median_weekly_inflow_gwh"] * MWH_PER_GWH
    profile["assumed_hourly_inflow_mwh"] = (
        profile["median_weekly_inflow_mwh"] / HOURS_PER_WEEK
    )
    profile["scenario_semantics"] = "historical median allocated uniformly across 168 hours; modeled, not observed hourly inflow"
    return profile


def build_assumed_hourly_inflow_scenario(
    profile: pd.DataFrame,
    *,
    year: int = 2025,
) -> pd.DataFrame:
    """Map weekly median inflow to hourly UTC timestamps using ISO week numbers."""
    if not 1 <= int(year) <= 9998:
        raise HydroDataError("year must be between 1 and 9998")
    required = {"iso_week", "median_weekly_inflow_gwh", "assumed_hourly_inflow_mwh"}
    if not required.issubset(profile.columns):
        raise HydroDataError(f"Seasonal profile is missing columns: {sorted(required - set(profile.columns))}")
    start = pd.Timestamp(year=int(year), month=1, day=1, tz="UTC")
    stop = pd.Timestamp(year=int(year) + 1, month=1, day=1, tz="UTC")
    timestamps = pd.date_range(start, stop - pd.Timedelta(hours=1), freq="h", name="target_timestamp_utc")
    weeks = timestamps.isocalendar().week.astype(int)
    indexed = profile.set_index("iso_week")
    missing_weeks = sorted(set(weeks.unique()) - set(indexed.index.astype(int)))
    if missing_weeks:
        raise HydroDataError(f"Seasonal profile has no historical reference for ISO weeks {missing_weeks}")
    result = pd.DataFrame(
        {
            "target_timestamp_utc": timestamps,
            "iso_week": weeks.to_numpy(),
            "historical_median_weekly_inflow_gwh": weeks.map(indexed["median_weekly_inflow_gwh"]).to_numpy(),
            "assumed_inflow_mwh_per_hour": weeks.map(indexed["assumed_hourly_inflow_mwh"]).to_numpy(),
            "inflow_scenario": "historical seasonal profile; constant within ISO week; not observed 2025 inflow",
            "provenance": "synthetic, derived from DE00 historical weekly-inflow medians",
        }
    )
    result["synthetic_2025_hydro_inflow_mwh"] = result["assumed_inflow_mwh_per_hour"]
    return result


def clamp_initial_reservoir_energy(
    energy_mwh: float,
    capacity_mwh: float,
) -> float:
    """Clamp an initial reservoir-energy input to its physical range."""
    if not math.isfinite(energy_mwh) or not math.isfinite(capacity_mwh) or capacity_mwh <= 0:
        raise HydroDataError("Reservoir energy and positive capacity must be finite")
    return min(capacity_mwh, max(0.0, energy_mwh))


def reservoir_balance_step(
    current_energy_mwh: float,
    inflow_energy_mwh: float,
    generation_energy_mwh: float,
    capacity_mwh: float,
) -> tuple[float, float]:
    """Return next reservoir energy and spill under energy conservation.

    The relation is next = current + inflow - generation - spill. This helper
    documents/test-drives accounting; it does not modify the dispatch model.
    """
    values = (current_energy_mwh, inflow_energy_mwh, generation_energy_mwh, capacity_mwh)
    if any(not math.isfinite(value) for value in values):
        raise HydroDataError("Reservoir balance values must be finite")
    if capacity_mwh <= 0 or current_energy_mwh < 0 or current_energy_mwh > capacity_mwh:
        raise HydroDataError("Current reservoir energy must be within a positive capacity")
    if inflow_energy_mwh < 0 or generation_energy_mwh < 0:
        raise HydroDataError("Inflow and hydro generation energy must be nonnegative")
    available = current_energy_mwh + inflow_energy_mwh
    if generation_energy_mwh > available + 1e-9:
        raise HydroDataError("Hydro generation exceeds current stored energy plus inflow")
    before_spill = max(0.0, available - generation_energy_mwh)
    spill = max(0.0, before_spill - capacity_mwh)
    next_energy = before_spill - spill
    return next_energy, spill


def write_de00_hydro_scenario(
    capacity_path: str | Path,
    inflow_path: str | Path,
    level_path: str | Path,
    *,
    output_dir: str | Path,
    year: int = 2025,
    initial_filling_fraction: float = 0.5,
    minimum_reservoir_energy_mwh: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write reference parameters, seasonal profile, and modeled hourly inflow scenario."""
    output = Path(output_dir)
    paths = {
        "parameters": output / "pecd_de00_reservoir_parameters.json",
        "profile": output / "pecd_de00_seasonal_weekly_inflow.csv",
        "hourly_scenario": output / f"pecd_de00_assumed_hourly_inflow_{year}.csv",
        "provenance": output / "pecd_de00_hydro_provenance.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite hydro outputs: " + ", ".join(map(str, existing)))

    params = load_de00_reservoir_parameters(
        capacity_path,
        initial_filling_fraction=initial_filling_fraction,
        minimum_reservoir_energy_mwh=minimum_reservoir_energy_mwh,
    )
    profile = build_seasonal_reservoir_inflow_profile(inflow_path, zone=params.zone)
    hourly = build_assumed_hourly_inflow_scenario(profile, year=year)
    hourly["hydro_turbine_generation_limit_mw"] = params.turbine_capacity_mw
    hourly["maximum_reservoir_energy_mwh"] = params.reservoir_capacity_mwh
    hourly["scenario_random_seed"] = 42
    levels = _read(level_path)
    if not {"zone", "year", "week"}.issubset(levels.columns):
        raise HydroDataError("Reservoir-level table must contain zone, year, and week columns")
    german_levels = levels.loc[levels["zone"].astype(str).eq(params.zone)]
    level_column = next((column for column in levels.columns if "Reservoir levels at beginning" in column), None)
    measured_level_count = (
        int(german_levels[level_column].notna().sum()) if level_column is not None and not german_levels.empty else 0
    )

    output.mkdir(parents=True, exist_ok=True)
    paths["parameters"].write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    profile.to_csv(paths["profile"], index=False)
    hourly.to_csv(paths["hourly_scenario"], index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    provenance = {
        "zone": params.zone,
        "technology": params.technology,
        "capacity_source": {"file": str(Path(capacity_path)), "sha256": _sha256(capacity_path)},
        "inflow_source": {"file": str(Path(inflow_path)), "sha256": _sha256(inflow_path)},
        "reservoir_level_source": {"file": str(Path(level_path)), "sha256": _sha256(level_path)},
        "historical_inflow_period": [int(_read(inflow_path).year.min()), int(_read(inflow_path).year.max())],
        "profile_aggregation": "median of DE00 reservoir weekly GWh totals by week number over historical years",
        "hourly_allocation": "median weekly GWh * 1000 MWh/GWh / 168 hours; constant within each ISO week",
        "scenario_year": int(year),
        "scenario_is_observed_2025_inflow": False,
        "week_alignment": "assumes PECD week numbers correspond to ISO week-of-year; hourly UTC timestamps map by ISO week number; week 53 is used only in ISO years that contain it",
        "germany_reservoir_level_rows": len(german_levels),
        "germany_measured_level_values": measured_level_count,
        "initial_filling_fraction_assumed": params.initial_filling_fraction,
        "spill_handling_note": "Reservoir balance helper returns spill when inflow would exceed capacity; optimizer integration is deferred to Stage 4.",
    }
    paths["provenance"].write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return paths
