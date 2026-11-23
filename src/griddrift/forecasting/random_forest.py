"""Direct 24-hour Random Forest forecaster using GridDRIFT's past-only features."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from griddrift.forecasting.features import (
    DirectFeatureSpec,
    _validate_demand,
    build_feature_row,
    build_supervised_dataset,
)


@dataclass
class DirectRandomForestForecaster:
    """Native multi-output forest; output columns represent issue+1h through issue+24h."""

    model: RandomForestRegressor
    feature_spec: DirectFeatureSpec
    model_version: str
    feature_columns: list[str]
    train_end_issue_time: pd.Timestamp
    parameters: dict[str, object]

    def predict_horizon(self, history: pd.Series, issue_time: pd.Timestamp) -> pd.Series:
        row = build_feature_row(history, issue_time, self.feature_spec)
        features = pd.DataFrame([row], columns=self.feature_columns)
        prediction = np.clip(np.asarray(self.model.predict(features)).reshape(-1), 0.0, None)
        issue = pd.Timestamp(issue_time)
        issue = issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
        targets = pd.date_range(issue + pd.Timedelta(hours=1), periods=len(prediction), freq="1h")
        return pd.Series(prediction, index=targets, name="predicted_demand_mw")

    def predict_feature_frame(self, features: pd.DataFrame) -> np.ndarray:
        """Predict all horizons for an existing shared feature matrix."""
        predictions = self.model.predict(features.loc[:, self.feature_columns])
        return np.clip(np.asarray(predictions, dtype=float), 0.0, None)

    def save(self, directory: str | Path, metrics: dict[str, float] | None = None) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output / "model.joblib")
        metadata = {
            "model_version": self.model_version,
            "feature_spec": asdict(self.feature_spec),
            "feature_columns": self.feature_columns,
            "train_end_issue_time": pd.Timestamp(self.train_end_issue_time).isoformat(),
            "parameters": self.parameters,
            "metrics": metrics or {},
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        return output

    @classmethod
    def load(cls, directory: str | Path) -> DirectRandomForestForecaster:
        return joblib.load(Path(directory) / "model.joblib")


@dataclass(frozen=True)
class RandomForestTrainingResult:
    model: DirectRandomForestForecaster
    validation_metrics: dict[str, float]
    train_samples: int
    validation_samples: int
    final_fit_samples: int
    validation_train_end_issue_time: pd.Timestamp
    validation_first_issue_time: pd.Timestamp
    training_seconds: float


def _estimator_parameters(
    parameters: dict[str, object] | None, random_state: int
) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": 200,
        "max_features": 1.0,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": random_state,
    }
    if parameters:
        params.update(parameters)
    return params


def fit_direct_random_forest_samples(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    feature_spec: DirectFeatureSpec | None = None,
    parameters: dict[str, object] | None = None,
    random_state: int = 42,
    model_version: str | None = None,
) -> DirectRandomForestForecaster:
    """Fit a fresh forest on caller-selected samples whose full targets are already known."""
    feature_spec = feature_spec or DirectFeatureSpec()
    if len(X) != len(y) or len(X) < 60 or y.shape[1] != feature_spec.horizon_hours:
        raise ValueError("X/y training samples are insufficient or have the wrong horizon")
    if not X.index.equals(y.index):
        raise ValueError("X and y issue timestamps must match exactly")
    params = _estimator_parameters(parameters, random_state)
    estimator = RandomForestRegressor(**params)
    estimator.fit(X.loc[:, X.columns], y.to_numpy())
    version = model_version
    if version is None:
        signature = {
            "parameters": params,
            "first_issue": X.index[0].isoformat(),
            "last_issue": X.index[-1].isoformat(),
            "samples": len(X),
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
        version = f"rf-{digest[:12]}"
    return DirectRandomForestForecaster(
        model=estimator,
        feature_spec=feature_spec,
        model_version=version,
        feature_columns=X.columns.tolist(),
        train_end_issue_time=X.index[-1],
        parameters=params,
    )


def train_direct_random_forest(
    demand: pd.Series,
    *,
    feature_spec: DirectFeatureSpec | None = None,
    train_end_issue_time: pd.Timestamp | None = None,
    training_window_days: int | None = None,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    model_version: str | None = None,
    parameters: dict[str, object] | None = None,
) -> RandomForestTrainingResult:
    """Train a frozen RF with chronological validation and a 24-hour target embargo."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    feature_spec = feature_spec or DirectFeatureSpec()
    demand = _validate_demand(demand)
    if train_end_issue_time is not None:
        cutoff = pd.Timestamp(train_end_issue_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff.tz_convert("UTC")
        demand = demand.loc[: cutoff + pd.Timedelta(hours=feature_spec.horizon_hours)]
    X, y = build_supervised_dataset(demand, feature_spec)
    if train_end_issue_time is not None:
        X, y = X.loc[X.index <= cutoff], y.loc[X.index <= cutoff]
    if training_window_days is not None:
        if training_window_days < 1:
            raise ValueError("training_window_days must be positive")
        keep = X.index > X.index[-1] - pd.Timedelta(days=training_window_days)
        X, y = X.loc[keep], y.loc[keep]
    if len(X) < 100:
        raise ValueError("not enough eligible samples for Random Forest training")

    validation_count = max(feature_spec.horizon_hours, int(len(X) * validation_fraction))
    validation_start = len(X) - validation_count
    validation_first_issue = X.index[validation_start]
    train_mask = X.index + pd.Timedelta(hours=feature_spec.horizon_hours) < validation_first_issue
    train_X, train_y = X.loc[train_mask], y.loc[train_mask]
    valid_X, valid_y = X.iloc[validation_start:], y.iloc[validation_start:]
    if len(train_X) < 60 or len(valid_X) < feature_spec.horizon_hours:
        raise ValueError("not enough samples after chronological 24-hour validation embargo")

    params = _estimator_parameters(parameters, random_state)
    started = time.perf_counter()
    validation_model = RandomForestRegressor(**params)
    validation_model.fit(train_X, train_y.to_numpy())
    predictions = np.clip(validation_model.predict(valid_X), 0.0, None)
    actual = valid_y.to_numpy()
    metrics = {
        "mae_mw": float(mean_absolute_error(actual.ravel(), predictions.ravel())),
        "rmse_mw": float(np.sqrt(mean_squared_error(actual.ravel(), predictions.ravel()))),
        "fixed_24h_mae_mw": float(mean_absolute_error(actual[:, -1], predictions[:, -1])),
        "fixed_24h_rmse_mw": float(np.sqrt(mean_squared_error(actual[:, -1], predictions[:, -1]))),
    }
    model = fit_direct_random_forest_samples(
        X, y, feature_spec=feature_spec, parameters=params,
        random_state=random_state, model_version=model_version,
    )
    elapsed = time.perf_counter() - started
    return RandomForestTrainingResult(
        model=model,
        validation_metrics=metrics,
        train_samples=len(train_X),
        validation_samples=len(valid_X),
        final_fit_samples=len(X),
        validation_train_end_issue_time=train_X.index[-1],
        validation_first_issue_time=valid_X.index[0],
        training_seconds=elapsed,
    )
