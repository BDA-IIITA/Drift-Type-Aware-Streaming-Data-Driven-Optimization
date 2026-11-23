"""Merge contiguous cached ENTSO-E actual-load CSV ranges without overwriting them."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Cached ENTSO-E CSVs to merge")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frames = [pd.read_csv(path, parse_dates=["timestamp"]) for path in args.inputs]
    data = pd.concat(frames, ignore_index=True)
    if list(data.columns) != ["timestamp", "demand_mw"]:
        raise SystemExit("Expected CSV columns: timestamp,demand_mw")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="raise")
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if data["timestamp"].duplicated().any():
        raise SystemExit("Input ranges overlap: duplicate timestamps found; refusing to merge")
    expected = pd.date_range(data["timestamp"].iloc[0], data["timestamp"].iloc[-1], freq="1h")
    if len(expected) != len(data) or not data["timestamp"].equals(pd.Series(expected)):
        raise SystemExit("Input ranges contain a gap or are not hourly; refusing to merge")
    if args.output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, date_format="%Y-%m-%d %H:%M:%S%z")
    print(
        f"Merged {len(data):,} hourly rows from {data['timestamp'].iloc[0]} "
        f"through {data['timestamp'].iloc[-1]} into {args.output}"
    )


if __name__ == "__main__":
    main()
