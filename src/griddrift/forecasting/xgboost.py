"""Direct 24-horizon XGBoost forecaster using GridDRIFT's past-only features."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from griddrift.forecasting.features import (
    DirectFeatureSpec,
    _validate_demand,
    build_feature_row,
    build_supervised_dataset,
)


def _calendar_columns(step: int) -> list[str]:
    prefix = f"target_{step:02d}h_"
    return [
        f"{prefix}{name}"
        for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos")
    ]


@dataclass
class DirectXGBoostForecaster:
    """Native multi-output tree ensemble; columns always target issue+1h through +24h."""

    models: list[object]
    feature_spec: DirectFeatureSpec
    model_version: str
    feature_columns: list[str]
    train_end_issue_time: pd.Timestamp
    parameters: dict[str, object]
    multi_output: bool = False
    eligible_window_start: pd.Timestamp | None = None
    eligible_window_end: pd.Timestamp | None = None

    def _features_for_step(self, row: dict[str, float], step: int) -> pd.DataFrame:
        past_columns = [c for c in self.feature_columns if not c.startswith("target_")]
        columns = past_columns + _calendar_columns(step)
        return pd.DataFrame([[row[c] for c in columns]], columns=columns)

    def predict_horizon(self, history: pd.Series, issue_time: pd.Timestamp) -> pd.Series:
        row = build_feature_row(history, issue_time, self.feature_spec)
        if self.multi_output:
            X = pd.DataFrame([row], columns=self.feature_columns)
            prediction = np.asarray(self.models[0].predict(X)).reshape(-1)
        else:
            prediction = np.asarray([
                float(model.predict(self._features_for_step(row, step))[0])
                for step, model in enumerate(self.models, start=1)
            ])
        prediction = np.clip(prediction, 0.0, None)
        issue = pd.Timestamp(issue_time)
        issue = issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
        targets = pd.date_range(issue + pd.Timedelta(hours=1), periods=len(prediction), freq="1h")
        return pd.Series(prediction, index=targets, name="predicted_demand_mw")

    def predict_feature_frame(self, features: pd.DataFrame) -> np.ndarray:
        """Vectorized prediction for already-built, past-only feature rows."""
        if self.multi_output:
            return np.clip(np.asarray(self.models[0].predict(features.loc[:, self.feature_columns])), 0.0, None)
        past_columns = [c for c in self.feature_columns if not c.startswith("target_")]
        output = np.empty((len(features), self.feature_spec.horizon_hours), dtype=float)
        for step, model in enumerate(self.models, start=1):
            columns = past_columns + _calendar_columns(step)
            output[:, step - 1] = model.predict(features.loc[:, columns])
        return np.clip(output, 0.0, None)

    def save(self, directory: str | Path, metrics: dict[str, float] | None = None) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output / "model.joblib")
        metadata = {
            "model_version": self.model_version,
            "feature_spec": asdict(self.feature_spec),
            "feature_columns": self.feature_columns,
            "train_end_issue_time": pd.Timestamp(self.train_end_issue_time).isoformat(),
            "eligible_window_start": self.eligible_window_start.isoformat() if self.eligible_window_start is not None else None,
            "eligible_window_end": self.eligible_window_end.isoformat() if self.eligible_window_end is not None else None,
            "parameters": self.parameters,
            "metrics": metrics or {},
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, directory: str | Path) -> DirectXGBoostForecaster:
        return joblib.load(Path(directory) / "model.joblib")


@dataclass(frozen=True)
class XGBoostTrainingResult:
    model: DirectXGBoostForecaster
    validation_metrics: dict[str, float]
    train_samples: int
    validation_samples: int
    final_fit_samples: int
    validation_train_end_issue_time: pd.Timestamp
    validation_first_issue_time: pd.Timestamp
    training_seconds: float


def fit_direct_xgboost_samples(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    feature_spec: DirectFeatureSpec | None = None,
    parameters: dict[str, object] | None = None,
    random_state: int = 42,
    model_version: str | None = None,
) -> DirectXGBoostForecaster:
    """Fit a fresh direct model on caller-selected, fully eligible samples."""
    from xgboost import XGBRegressor

    feature_spec = feature_spec or DirectFeatureSpec()

    if len(X) != len(y) or len(X) < 60 or y.shape[1] != feature_spec.horizon_hours:
        raise ValueError("X/y training samples are insufficient or have the wrong horizon")
    params: dict[str, object] = {
        "n_estimators": 80, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0,
        "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 4,
        "random_state": random_state, "verbosity": 0, "multi_strategy": "one_output_per_tree",
    }
    if parameters:
        params.update(parameters)
    estimator = XGBRegressor(**params)
    estimator.fit(X.loc[:, X.columns], y.to_numpy())
    return DirectXGBoostForecaster(
        models=[estimator],
        feature_spec=feature_spec,
        model_version=model_version or f"xgb-{uuid4().hex[:12]}",
        feature_columns=X.columns.tolist(),
        train_end_issue_time=X.index[-1],
        parameters=params,
        multi_output=True,
        eligible_window_start=X.index[0],
        eligible_window_end=X.index[-1],
    )


def train_direct_xgboost(
    demand: pd.Series,
    *,
    feature_spec: DirectFeatureSpec | None = None,
    train_end_issue_time: pd.Timestamp | None = None,
    training_window_days: int | None = None,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    model_version: str | None = None,
    parameters: dict[str, object] | None = None,
) -> XGBoostTrainingResult:
    """Train from scratch; validation is chronological and labels are complete."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised in environments without optional extra
        raise RuntimeError("XGBoost support is not installed; install the [forecasting] extra") from exc
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    feature_spec = feature_spec or DirectFeatureSpec()
    demand = _validate_demand(demand)
    if train_end_issue_time is not None:
        cutoff = pd.Timestamp(train_end_issue_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff.tz_convert("UTC")
        # Avoid constructing supervised features for years of data beyond the frozen
        # training boundary. Labels through cutoff+24h were observable at that boundary.
        demand = demand.loc[: cutoff + pd.Timedelta(hours=feature_spec.horizon_hours)]
    X, y = build_supervised_dataset(demand, feature_spec)
    if train_end_issue_time is not None:
        X, y = X.loc[X.index <= cutoff], y.loc[X.index <= cutoff]
    if training_window_days is not None:
        if training_window_days < 1:
            raise ValueError("training_window_days must be positive")
        X, y = X.loc[X.index > X.index[-1] - pd.Timedelta(days=training_window_days)], y.loc[
            X.index > X.index[-1] - pd.Timedelta(days=training_window_days)
        ]
    if len(X) < 100:
        raise ValueError("not enough eligible samples for XGBoost training")

    # A 24-hour issue-time embargo ensures no training labels overlap the validation targets.
    validation_count = max(24, int(len(X) * validation_fraction))
    validation_start = len(X) - validation_count
    validation_first_issue = X.index[validation_start]
    train_mask = X.index + pd.Timedelta(hours=feature_spec.horizon_hours) < validation_first_issue
    train_X, train_y = X.loc[train_mask], y.loc[train_mask]
    valid_X, valid_y = X.iloc[validation_start:], y.iloc[validation_start:]
    if len(train_X) < 60 or len(valid_X) < 24:
        raise ValueError("not enough samples after chronological 24-hour validation embargo")

    params: dict[str, object] = {
        "n_estimators": 80,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": random_state,
        "multi_strategy": "one_output_per_tree",
        "verbosity": 0,
    }
    if parameters:
        params.update(parameters)
    validation_predictions = np.empty(valid_y.shape, dtype=float)
    started = time.perf_counter()
    # Validate on the chronological tail, then fit the frozen model on every eligible row
    # through the requested cutoff. The validation labels are historical, never test labels.
    validation_model = XGBRegressor(**params)
    validation_model.fit(train_X, train_y.to_numpy())
    validation_predictions[:, :] = validation_model.predict(valid_X)
    elapsed = time.perf_counter() - started
    actual, predicted = valid_y.to_numpy(), np.clip(validation_predictions, 0.0, None)
    metrics = {
        "mae_mw": float(mean_absolute_error(actual.ravel(), predicted.ravel())),
        "rmse_mw": float(np.sqrt(mean_squared_error(actual.ravel(), predicted.ravel()))),
        "fixed_24h_mae_mw": float(mean_absolute_error(actual[:, -1], predicted[:, -1])),
        "fixed_24h_rmse_mw": float(np.sqrt(mean_squared_error(actual[:, -1], predicted[:, -1]))),
    }
    frozen_started = time.perf_counter()
    model = fit_direct_xgboost_samples(
        X, y, feature_spec=feature_spec, parameters=params,
        random_state=random_state, model_version=model_version,
    )
    elapsed += time.perf_counter() - frozen_started
    return XGBoostTrainingResult(
        model, metrics, len(train_X), len(valid_X), len(X), train_X.index[-1],
        valid_X.index[0], elapsed,
    )
