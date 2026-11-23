"""Create a clearly labeled demand-shift copy for demonstrating ADWIN drift."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def create_demo_drift(
    input_path: Path,
    output_path: Path,
    *,
    shift_start: str,
    increase_fraction: float = 0.15,
    additional_events: tuple[tuple[str, float], ...] = (),
) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """Copy hourly demand and apply permanent percentage level shifts."""
    events = ((shift_start, increase_fraction), *additional_events)
    if any(not -1 < fraction <= 1 for _, fraction in events):
        raise ValueError("each event fraction must be greater than -1 and at most 1")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")

    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    if list(frame.columns) != ["timestamp", "demand_mw"]:
        raise ValueError("input must have exactly timestamp,demand_mw columns")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["demand_mw"] = pd.to_numeric(frame["demand_mw"], errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if frame.timestamp.duplicated().any():
        raise ValueError("input contains duplicate timestamps")
    expected = pd.date_range(frame.timestamp.iloc[0], frame.timestamp.iloc[-1], freq="1h")
    if not frame.timestamp.equals(pd.Series(expected)):
        raise ValueError("input is not a continuous hourly series")

    timestamps = pd.DatetimeIndex(frame.timestamp)
    starts = []
    changed = pd.Series(False, index=frame.index)
    for event_time, fraction in events:
        start = pd.Timestamp(event_time)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        if start not in timestamps:
            raise ValueError(f"shift_start {start} is not present in the input data")
        starts.append(start)
        mask = frame.timestamp >= start
        frame.loc[mask, "demand_mw"] *= 1 + fraction
        changed |= mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S%z")
    return int(changed.sum()), min(starts), frame.timestamp.iloc[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original hourly demand CSV")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/entsoe_actual_load_DE_LU_demo_drift_15pct.csv"),
    )
    parser.add_argument("--shift-start", default="2025-01-02T00:00:00Z")
    parser.add_argument("--increase-fraction", type=float, default=0.15)
    parser.add_argument(
        "--event", action="append", metavar="UTC_TIME=FRACTION",
        help="Add a level shift; repeat, e.g. --event 2025-01-16T00:00:00Z=-0.1",
    )
    args = parser.parse_args()
    events = []
    for raw_event in args.event or []:
        timestamp, fraction = raw_event.rsplit("=", 1)
        events.append((timestamp, float(fraction)))
    count, start, end = create_demo_drift(
        args.input, args.output, shift_start=args.shift_start,
        increase_fraction=args.increase_fraction, additional_events=tuple(events),
    )
    print(
        f"Created simulated drift demo with {1 + len(events)} level shifts starting {start} "
        f"and ending {end} ({count:,} hourly values changed): {args.output}"
    )


if __name__ == "__main__":
    main()
