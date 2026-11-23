"""Leakage-safe direct multi-output demand forecasting and forecast provenance."""

from griddrift.forecasting.mlp import DirectMLPForecaster, TrainingResult, train_direct_mlp
from griddrift.forecasting.random_forest import (
    DirectRandomForestForecaster,
    RandomForestTrainingResult,
    train_direct_random_forest,
)
from griddrift.forecasting.registry import ForecastRecord, ForecastRegistry
from griddrift.forecasting.xgboost import (
    DirectXGBoostForecaster,
    XGBoostTrainingResult,
    train_direct_xgboost,
)

__all__ = [
    "DirectMLPForecaster",
    "DirectRandomForestForecaster",
    "DirectXGBoostForecaster",
    "ForecastRecord",
    "ForecastRegistry",
    "RandomForestTrainingResult",
    "TrainingResult",
    "XGBoostTrainingResult",
    "train_direct_mlp",
    "train_direct_random_forest",
    "train_direct_xgboost",
]
