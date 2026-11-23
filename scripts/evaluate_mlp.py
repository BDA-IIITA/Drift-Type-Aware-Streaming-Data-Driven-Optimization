"""Evaluate a persisted MLP only after its saved chronological train boundary."""

from __future__ import annotations

import argparse

from griddrift.forecasting.mlp import DirectMLPForecaster, evaluate_direct_mlp
from train_mlp import _demand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    metrics = evaluate_direct_mlp(DirectMLPForecaster.load(args.model), _demand(args.input))
    print(metrics)


if __name__ == "__main__":
    main()
