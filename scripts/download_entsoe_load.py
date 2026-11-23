"""Download actual ENTSO-E demand using ENTSOE_API_TOKEN; never prints the token."""

from __future__ import annotations

import argparse

from griddrift.config import load_project_config
from griddrift.data import fetch_entsoe_actual_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--cache-dir", default="data/raw/entsoe_cache")
    parser.add_argument(
        "--start",
        help="Override the configured start date (UTC); useful for downloading a missing range",
    )
    parser.add_argument(
        "--end",
        help="Override the configured end date (UTC, inclusive when date-only)",
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    config = load_project_config(args.config)
    frame, path, from_cache = fetch_entsoe_actual_load(
        bidding_zone=config.data.bidding_zone,
        start=args.start or config.data.start_date,
        end=args.end or config.data.end_date,
        resolution=config.data.resolution,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )
    print(f"{'Loaded' if from_cache else 'Downloaded'} {len(frame)} hourly rows: {path}")


if __name__ == "__main__":
    main()
