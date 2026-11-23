"""Read-only inventory for Phase 5.5 source datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/experiments/phase55/stage1_data_inventory.csv"
SOURCES = {
    "entsoe_renewables": ROOT / "entsoe_de_lu_2025_hourly_features.csv",
    "pecd_hydro_capacity": ROOT / "PECD-hydro-capacities.csv",
    "pecd_hydro_inflows": ROOT / "PECD-hydro-weekly-inflows.csv",
    "pecd_hydro_reservoir_levels": ROOT / "PECD-hydro-weekly-reservoir-levels.csv",
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def inspect(name: str, path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    result: dict[str, object] = {
        "dataset": name,
        "filename": path.name,
        "path": str(path.relative_to(ROOT)),
        "file_bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": _json(list(frame.columns)),
        "missing_by_column": _json({col: int(count) for col, count in frame.isna().sum().items()}),
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "duplicate_key_rows": "",
        "time_coverage_start": "",
        "time_coverage_end": "",
        "time_basis": "",
        "timestamp_format_timezone": "Not applicable; no timestamp column.",
        "timestamp_invalid": "Not applicable",
        "timestamp_duplicate_count": "Not applicable",
        "timestamp_gaps_over_1h": "Not applicable",
        "region_codes": "",
        "units_and_semantics": "",
        "quality_notes": "",
    }
    if "timestamp" in frame:
        parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        ordered = parsed.dropna().sort_values()
        diffs = ordered.diff().dropna()
        timestamp_values = frame["timestamp"].dropna().astype(str)
        result.update(
            time_coverage_start=ordered.min().isoformat() if len(ordered) else "",
            time_coverage_end=ordered.max().isoformat() if len(ordered) else "",
            time_basis="Hourly timestamped observations",
            timestamp_format_timezone=(
                f"Example {timestamp_values.iloc[0]}; "
                f"{'all values end in Z' if timestamp_values.str.endswith('Z').all() else 'timezone notation varies'}; UTC normalized"
            ),
            timestamp_invalid=int(parsed.isna().sum()),
            timestamp_duplicate_count=int(parsed.duplicated().sum()),
            timestamp_gaps_over_1h=int((diffs > pd.Timedelta(hours=1)).sum()),
            units_and_semantics=(
                "Columns explicitly use _mw and _eur_mwh suffixes; demand, generation and forecasts are MW; "
                "day_ahead_price_eur_mwh is EUR/MWh. Timestamp strings use ISO-8601 Z (UTC)."
            ),
        )
        for col in frame.columns:
            if col.endswith("_mw"):
                numeric = pd.to_numeric(frame[col], errors="coerce")
                if (numeric < 0).any():
                    result["quality_notes"] = (
                        f"Negative values in {col}: {int((numeric < 0).sum())}; inspect before modeling."
                    )
                    break
        if not any(c in frame.columns for c in ("zone", "region", "country", "bidding_zone")):
            result["region_codes"] = "No region column; geography inferred only from filename, not independently verifiable."
    if "zone" in frame:
        zones = sorted(frame["zone"].dropna().astype(str).unique().tolist())
        result["region_codes"] = _json(zones)
        result["quality_notes"] = f"zone is the geographic key; {len(zones)} distinct codes."
        if {"type", "variable"}.issubset(frame.columns):
            result["duplicate_key_rows"] = int(frame.duplicated(["zone", "type", "variable"]).sum())
        elif {"week", "year"}.issubset(frame.columns):
            result["duplicate_key_rows"] = int(frame.duplicated(["zone", "week", "year"]).sum())
    if name == "pecd_hydro_capacity":
        result["units_and_semantics"] = "Long-format technology parameters; turbine capacity is MW; reservoir capacity is GWh per variable label."
        result["time_basis"] = "Static parameter table; reference year not specified in CSV."
        de = frame[frame["zone"].astype(str).eq("DE00")]
        result["quality_notes"] = f"DE00 parameter rows: {len(de)}; types: {_json(sorted(de.type.dropna().unique().tolist()))}."
    elif name == "pecd_hydro_inflows":
        result["units_and_semantics"] = "Weekly energy totals in GWh in both inflow columns; historical seasonal reference, not 2025 observations."
        de = frame[frame["zone"].astype(str).eq("DE00")]
        result["time_coverage_start"] = str(int(de.year.min())) if len(de) else ""
        result["time_coverage_end"] = str(int(de.year.max())) if len(de) else ""
        result["time_basis"] = "Historical weekly records; `year` and `week` index, no timestamps or timezone."
        result["quality_notes"] = (
            f"DE00 rows={len(de)}; years={de.year.nunique() if len(de) else 0}; "
            f"weeks={de.week.min() if len(de) else ''}..{de.week.max() if len(de) else ''}."
        )
    elif name == "pecd_hydro_reservoir_levels":
        result["units_and_semantics"] = "Weekly beginning-of-week reservoir level ratios, dimensionless 0..1; NA indicates unavailable measurements."
        de = frame[frame["zone"].astype(str).eq("DE00")]
        result["time_coverage_start"] = str(int(frame.year.min())) if len(frame) else ""
        result["time_coverage_end"] = str(int(frame.year.max())) if len(frame) else ""
        result["time_basis"] = "Historical weekly records; `year` and `week` index, no timestamps or timezone."
        result["quality_notes"] = (
            f"DE00 rows={len(de)}; nonmissing observed level entries="
            f"{int(de.iloc[:, 5].notna().sum()) if len(de) else 0}."
        )
    return result


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing requested source file(s): " + ", ".join(missing))
    records = [inspect(name, path) for name, path in SOURCES.items()]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT, index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(f"\nSaved inventory: {OUTPUT}")


if __name__ == "__main__":
    main()
