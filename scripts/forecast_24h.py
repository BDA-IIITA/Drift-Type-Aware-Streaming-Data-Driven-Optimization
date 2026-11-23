"""Generate and register a direct 24-hour demand forecast from observed history."""

from __future__ import annotations

import argparse

import pandas as pd

from griddrift.forecasting import ForecastRegistry
from griddrift.forecasting.mlp import DirectMLPForecaster
from train_mlp import _demand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", default="artifacts/forecasts/registry.csv")
    args = parser.parse_args()
    demand = _demand(args.input)
    model = DirectMLPForecaster.load(args.model)
    issue_time = demand.index[-1]
    prediction = model.predict_horizon(demand, issue_time)
    registry = ForecastRegistry()
    registry.record_horizon(issue_time, prediction, model.model_version)
    registry.save_csv(args.registry)
    print(prediction.to_frame())
    print(f"Registered {len(prediction)} forecasts in {args.registry}")


if __name__ == "__main__":
    main()
