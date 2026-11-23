"""Create the Phase 5.5 clean hourly renewable data table."""

from __future__ import annotations

import argparse
from pathlib import Path

from griddrift.data.renewables import preprocess_entsoe_renewables_csv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "entsoe_de_lu_2025_hourly_features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/entsoe_de_lu_2025_renewables_hourly.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    clean = preprocess_entsoe_renewables_csv(
        args.input,
        args.output,
        overwrite=args.overwrite,
    )
    missing = {
        "solar_forecast_missing": int(clean["solar_forecast_missing"].sum()),
        "wind_forecast_missing": int(clean["wind_forecast_missing"].sum()),
        "missing_source_hours": int((~clean["source_row_present"]).sum()),
    }
    print(f"Saved {len(clean)} hourly records to {args.output}")
    print(f"Saved provenance metadata to {args.output.with_suffix('.metadata.json')}")
    print(f"Missingness flags: {missing}")


if __name__ == "__main__":
    main()
