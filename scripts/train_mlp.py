"""Train and persist a direct 24-hour MLP from canonical hourly demand."""

from __future__ import annotations

import argparse

import pandas as pd

from griddrift.forecasting import train_direct_mlp


def _demand(path: str) -> pd.Series:
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return pd.Series(frame["demand_mw"].to_numpy(), index=index, name="demand_mw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/models/mlp")
    args = parser.parse_args()
    result = train_direct_mlp(_demand(args.input))
    result.model.save(args.output, result.metrics)
    print(f"Saved {result.model.model_version} to {args.output}")
    print(f"train={result.train_samples}, validation={result.validation_samples}, metrics={result.metrics}")


if __name__ == "__main__":
    main()
