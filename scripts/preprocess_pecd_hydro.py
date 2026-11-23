"""Create a PECD-derived DE00 reservoir reference and assumed inflow scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from griddrift.data.hydro import write_de00_hydro_scenario

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path, default=ROOT / "PECD-hydro-capacities.csv")
    parser.add_argument("--inflows", type=Path, default=ROOT / "PECD-hydro-weekly-inflows.csv")
    parser.add_argument("--levels", type=Path, default=ROOT / "PECD-hydro-weekly-reservoir-levels.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--initial-filling-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-reservoir-energy-mwh", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = write_de00_hydro_scenario(
        args.capacity,
        args.inflows,
        args.levels,
        output_dir=args.output_dir,
        year=args.year,
        initial_filling_fraction=args.initial_filling_fraction,
        minimum_reservoir_energy_mwh=args.minimum_reservoir_energy_mwh,
        overwrite=args.overwrite,
    )
    for label, path in paths.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
