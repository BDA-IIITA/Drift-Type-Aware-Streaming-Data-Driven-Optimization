"""Direct multi-output MLP training, persistence, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from griddrift.forecasting.features import DirectFeatureSpec, build_feature_row, build_supervised_dataset


@dataclass(frozen=True)
class TrainingResult:
    model: "DirectMLPForecaster"
    train_samples: int
    validation_samples: int
    metrics: dict[str, float]
    eligible_window_start: pd.Timestamp
    eligible_window_end: pd.Timestamp


@dataclass
class DirectMLPForecaster:
    model: MLPRegressor
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_spec: DirectFeatureSpec
    model_version: str
    feature_columns: list[str]
    train_end_issue_time: pd.Timestamp
    eligible_window_start: pd.Timestamp | None = None
    eligible_window_end: pd.Timestamp | None = None

    def predict_horizon(self, history: pd.Series, issue_time: pd.Timestamp) -> pd.Series:
        row = build_feature_row(history, issue_time, self.feature_spec)
        X = pd.DataFrame([row], columns=self.feature_columns)
        scaled = self.feature_scaler.transform(X)
        prediction = self.target_scaler.inverse_transform(self.model.predict(scaled))[0]
        prediction = np.clip(prediction, 0.0, None)
        issue = pd.Timestamp(issue_time)
        issue = issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
        targets = pd.date_range(
            issue + pd.Timedelta(hours=1), periods=self.feature_spec.horizon_hours, freq="1h"
        )
        return pd.Series(prediction, index=targets, name="predicted_demand_mw")

    def save(self, directory: str | Path, metrics: dict[str, float]) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output / "model.joblib")
        metadata = {
            "model_version": self.model_version,
            "feature_spec": asdict(self.feature_spec),
            "feature_columns": self.feature_columns,
            "train_end_issue_time": self.train_end_issue_time.isoformat(),
            "eligible_window_start": (
                getattr(self, "eligible_window_start", None).isoformat()
                if getattr(self, "eligible_window_start", None)
                else None
            ),
            "eligible_window_end": (
                getattr(self, "eligible_window_end", None).isoformat()
                if getattr(self, "eligible_window_end", None)
                else None
            ),
            "metrics": metrics,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, directory: str | Path) -> "DirectMLPForecaster":
        return joblib.load(Path(directory) / "model.joblib")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_mw": float(mean_absolute_error(actual, predicted)),
        "rmse_mw": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def train_direct_mlp(
    demand: pd.Series,
    *,
    feature_spec: DirectFeatureSpec = DirectFeatureSpec(),
    validation_fraction: float = 0.2,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 300,
    random_state: int = 42,
    training_window_days: int | None = None,
    model_version: str | None = None,
) -> TrainingResult:
    """Fit scalers and MLP on the chronological training prefix only."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    X, y = build_supervised_dataset(demand, feature_spec)
    if training_window_days is not None:
        if training_window_days < 1:
            raise ValueError("training_window_days must be >= 1")
        window_start = X.index[-1] - pd.Timedelta(days=training_window_days)
        X, y = X.loc[X.index >= window_start], y.loc[X.index >= window_start]
    eligible_window_start, eligible_window_end = X.index.min(), X.index.max()
    validation_samples = max(1, int(len(X) * validation_fraction))
    train_samples = len(X) - validation_samples
    if train_samples < 10:
        raise ValueError("not enough chronological training samples")
    X_train, X_validation = X.iloc[:train_samples], X.iloc[train_samples:]
    y_train, y_validation = y.iloc[:train_samples], y.iloc[train_samples:]

    feature_scaler = StandardScaler().fit(X_train)
    target_scaler = StandardScaler().fit(y_train)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=False,
    ).fit(feature_scaler.transform(X_train), target_scaler.transform(y_train))
    predicted = target_scaler.inverse_transform(model.predict(feature_scaler.transform(X_validation)))
    result = DirectMLPForecaster(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_spec=feature_spec,
        model_version=model_version or f"mlp-{uuid4().hex[:12]}",
        feature_columns=X.columns.tolist(),
        train_end_issue_time=X_train.index[-1],
        eligible_window_start=eligible_window_start,
        eligible_window_end=eligible_window_end,
    )
    return TrainingResult(
        model=result,
        train_samples=train_samples,
        validation_samples=validation_samples,
        metrics=_metrics(y_validation.to_numpy(), predicted),
        eligible_window_start=eligible_window_start,
        eligible_window_end=eligible_window_end,
    )


def evaluate_direct_mlp(model: DirectMLPForecaster, demand: pd.Series) -> dict[str, float]:
    """Evaluate only samples after the saved chronological training boundary."""
    X, y = build_supervised_dataset(demand, model.feature_spec)
    test = X.index > model.train_end_issue_time
    if not test.any():
        raise ValueError("no held-out samples exist after the training boundary")
    X_test, y_test = X.loc[test], y.loc[test]
    predicted = model.target_scaler.inverse_transform(
        model.model.predict(model.feature_scaler.transform(X_test))
    )
    return _metrics(y_test.to_numpy(), predicted)
