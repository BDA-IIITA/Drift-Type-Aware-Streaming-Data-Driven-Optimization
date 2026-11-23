"""Sequential, leakage-safe historical replay with drift-triggered adaptation."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from griddrift.drift import ADWINConfig, ErrorDistributionChange, PersistentADWIN
from griddrift.forecasting.features import (
    _build_feature_row_validated,
    _validate_demand,
)
from griddrift.forecasting.mlp import DirectMLPForecaster, TrainingResult, train_direct_mlp
from griddrift.forecasting.registry import ForecastRegistry


class ErrorDetector(Protocol):
    def update(self, absolute_error_mw: float) -> bool: ...


@dataclass(frozen=True)
class ReplayConfig:
    horizon_hours: int = 24
    retraining_window_days: int = 60
    cooldown_days: int = 7
    adaptation_policy: str = "original"
    adaptation_strategy: str = "scratch"
    retraining_window_months: int | None = None
    validation_days: int = 7
    min_error_increase_mw: float = 250.0
    min_error_increase_fraction: float = 0.10
    min_validation_improvement_fraction: float = 0.02
    fine_tune_epochs: int = 5
    fine_tune_learning_rate_factor: float = 0.1
    fine_tune_max_updates: int = 8
    fine_tune_batch_size: int = 200
    fine_tune_random_state: int = 42
    validation_periods: int = 1
    representative_samples_per_month: int = 24
    adwin: ADWINConfig = field(default_factory=ADWINConfig)
    adaptation_max_iter: int = 300
    adaptation_validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.horizon_hours != 24:
            raise ValueError("Phase 3 replay monitors the fixed 24-hour-ahead stream")
        if self.retraining_window_days < 1 or self.cooldown_days < 0:
            raise ValueError("retraining window must be positive and cooldown non-negative")
        if self.adaptation_policy not in {"original", "increase_only", "validated"}:
            raise ValueError("adaptation_policy must be original, increase_only, or validated")
        if self.adaptation_strategy not in {
            "scratch", "fine_tune", "recent_plus_history", "gentle_fine_tune",
            "mixed_history_fine_tune",
        }:
            raise ValueError("unsupported adaptation_strategy")
        if self.retraining_window_months is not None and self.retraining_window_months < 1:
            raise ValueError("retraining_window_months must be >= 1")
        if self.validation_days < 1 or self.fine_tune_epochs < 1:
            raise ValueError("validation_days and fine_tune_epochs must be >= 1")
        if self.validation_periods < 1:
            raise ValueError("validation_periods must be >= 1")
        if not 0 < self.fine_tune_learning_rate_factor <= 1:
            raise ValueError("fine_tune_learning_rate_factor must be in (0, 1]")
        if self.fine_tune_max_updates < 1 or self.fine_tune_batch_size < 1:
            raise ValueError("fine-tune update and batch limits must be >= 1")
        if not 0 <= self.min_validation_improvement_fraction <= 1:
            raise ValueError("minimum validation improvement must be between 0 and 1")
        if self.min_error_increase_mw < 0 or self.min_error_increase_fraction < 0:
            raise ValueError("error increase thresholds must be non-negative")


@dataclass(frozen=True)
class AdaptationEvent:
    drift_timestamp: pd.Timestamp
    previous_model_version: str
    new_model_version: str | None
    training_window_start: pd.Timestamp | None
    training_window_end: pd.Timestamp | None
    latest_training_target_timestamp: pd.Timestamp | None
    outcome: str
    error_mean_before_mw: float | None = None
    error_mean_after_mw: float | None = None
    error_change_direction: str | None = None
    adwin_window_width_before: int | None = None
    adwin_window_width_after: int | None = None
    candidate_validation_mae_mw: float | None = None
    incumbent_validation_mae_mw: float | None = None
    validation_improvement_fraction: float | None = None
    training_strategy: str | None = None
    validation_first_target_timestamp: pd.Timestamp | None = None
    validation_last_target_timestamp: pd.Timestamp | None = None
    candidate_training_seconds: float | None = None


@dataclass
class ReplayResult:
    static_errors: pd.DataFrame
    adaptive_errors: pd.DataFrame
    adaptation_events: list[AdaptationEvent]
    static_registry: ForecastRegistry
    adaptive_registry: ForecastRegistry
    final_adaptive_model: DirectMLPForecaster
    runtime_seconds: float = 0.0
    inference_runtime_seconds: float = 0.0

    @property
    def candidate_training_runtime_seconds(self) -> float:
        return float(sum(event.candidate_training_seconds or 0.0 for event in self.adaptation_events))

    @staticmethod
    def _metrics(errors: pd.DataFrame) -> dict[str, float]:
        if errors.empty:
            return {"mae_mw": float("nan"), "rmse_mw": float("nan"), "count": 0.0}
        absolute = errors["absolute_error_mw"].to_numpy(dtype=float)
        return {
            "mae_mw": float(absolute.mean()),
            "rmse_mw": float(np.sqrt(np.mean(np.square(absolute)))),
            "count": float(len(errors)),
        }

    @property
    def static_metrics(self) -> dict[str, float]:
        return self._metrics(self.static_errors)

    @property
    def adaptive_metrics(self) -> dict[str, float]:
        return self._metrics(self.adaptive_errors)

    @property
    def drift_count(self) -> int:
        # Every event originates from an ADWIN alarm.  Some alarms deliberately
        # do not retrain because the cooldown protects against churn.
        return len(self.adaptation_events)

    @property
    def adaptation_count(self) -> int:
        return sum(event.outcome == "adapted" for event in self.adaptation_events)

    @property
    def adaptation_attempt_count(self) -> int:
        return sum(
            event.outcome in {"adapted", "candidate_rejected"}
            or event.outcome.startswith("candidate_rejected:")
            for event in self.adaptation_events
        )

    @property
    def rejected_adaptation_count(self) -> int:
        return sum(
            event.outcome == "candidate_rejected" or event.outcome.startswith("candidate_rejected:")
            for event in self.adaptation_events
        )

    def adaptation_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(event) for event in self.adaptation_events])


class StreamingReplay:
    """Replay demand in timestamp order; adaptation occurs only on accepted ADWIN alarms."""

    def __init__(
        self,
        demand: pd.Series,
        static_model: DirectMLPForecaster,
        *,
        config: ReplayConfig = ReplayConfig(),
        detector: ErrorDetector | None = None,
    ) -> None:
        # Validate the replay clock once. Building all supervised rows here is
        # unnecessary and expensive; rows are made only when fitting a candidate.
        self.demand = _validate_demand(demand)
        self.static_model = static_model
        self.adaptive_model = copy.deepcopy(static_model)
        self.config = config
        # Keep an untouched detector template so reset is reproducible even in
        # tests that inject a deterministic detector.
        self._detector_template = copy.deepcopy(detector) if detector is not None else None
        self.detector = detector or PersistentADWIN(config.adwin)
        self.static_registry = ForecastRegistry()
        self.adaptive_registry = ForecastRegistry()
        self.events: list[AdaptationEvent] = []
        self.last_adaptation_timestamp: pd.Timestamp | None = None

    def reset(self) -> None:
        self.adaptive_model = copy.deepcopy(self.static_model)
        self.static_registry = ForecastRegistry()
        self.adaptive_registry = ForecastRegistry()
        self.events = []
        self.last_adaptation_timestamp = None
        self.detector = (
            copy.deepcopy(self._detector_template)
            if self._detector_template is not None
            else PersistentADWIN(self.config.adwin)
        )

    def _first_issue_position(self) -> int:
        index = self.demand.index
        start = self.static_model.train_end_issue_time + pd.Timedelta(hours=self.config.horizon_hours)
        position = int(index.searchsorted(start, side="left"))
        if position >= len(index):
            raise ValueError("demand ends before the frozen model's labels are fully observed")
        return position

    def _can_adapt(self, timestamp: pd.Timestamp) -> bool:
        if self.last_adaptation_timestamp is None:
            return True
        return timestamp >= self.last_adaptation_timestamp + pd.Timedelta(days=self.config.cooldown_days)

    def _event_fields(self, change: ErrorDistributionChange | None) -> dict[str, object]:
        if change is None:
            return {}
        return {
            "error_mean_before_mw": change.error_mean_before_mw,
            "error_mean_after_mw": change.error_mean_after_mw,
            "error_change_direction": change.direction,
            "adwin_window_width_before": change.window_width_before,
            "adwin_window_width_after": change.window_width_after,
        }

    def _alarm_is_meaningful_increase(self, change: ErrorDistributionChange | None) -> bool:
        if change is None or change.direction != "error_increased":
            return False
        increase = change.error_mean_after_mw - change.error_mean_before_mw
        relative = increase / max(change.error_mean_before_mw, 1.0)
        return (
            increase >= self.config.min_error_increase_mw
            and relative >= self.config.min_error_increase_fraction
        )

    @staticmethod
    def _validation_predictions(
        model: DirectMLPForecaster, X: pd.DataFrame
    ) -> np.ndarray:
        scaled = model.feature_scaler.transform(X[model.feature_columns])
        scaled_prediction = model.model.predict(scaled)
        target_columns = getattr(model.target_scaler, "feature_names_in_", None)
        scaled_frame = (
            pd.DataFrame(scaled_prediction, columns=target_columns)
            if target_columns is not None
            else scaled_prediction
        )
        return model.target_scaler.inverse_transform(scaled_frame)[:, -1]

    @staticmethod
    def _build_selected_samples(
        observed: pd.Series,
        issue_times: pd.DatetimeIndex,
        model: DirectMLPForecaster,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build only selected past-only rows instead of rebuilding all history per alarm."""
        spec = model.feature_spec
        positions = observed.index.get_indexer(issue_times)
        if (positions < 0).any():
            raise ValueError("selected training or validation issue is not observed")
        rows: list[dict[str, float]] = []
        targets: list[np.ndarray] = []
        for issue, position in zip(issue_times, positions):
            history = observed.iloc[max(0, position + 1 - spec.lookback_hours) : position + 1]
            rows.append(_build_feature_row_validated(history, issue, spec))
            targets.append(
                observed.iloc[position + 1 : position + 1 + spec.horizon_hours].to_numpy(dtype=float)
            )
        X = pd.DataFrame(rows, index=pd.DatetimeIndex(issue_times, name="issue_timestamp"))
        X = X[model.feature_columns]
        y = pd.DataFrame(
            targets,
            index=X.index,
            columns=[f"target_{step:02d}h" for step in range(1, spec.horizon_hours + 1)],
        )
        return X, y

    def _fit_validated_candidate(
        self, timestamp: pd.Timestamp, old: DirectMLPForecaster, version: str
    ) -> tuple[
        DirectMLPForecaster,
        pd.DataFrame,
        pd.DataFrame,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
        float,
    ]:
        observed = self.demand.loc[:timestamp]
        spec = old.feature_spec
        first_position = spec.lookback_hours - 1
        last_position_exclusive = len(observed) - spec.horizon_hours
        if last_position_exclusive <= first_position:
            raise ValueError("not enough observed history for candidate validation")
        eligible_issues = pd.DatetimeIndex(
            observed.index[first_position:last_position_exclusive], name="issue_timestamp"
        )
        eligible_end = eligible_issues[-1]
        validation_length = pd.Timedelta(days=self.config.validation_days)
        validation_masks = []
        for period in range(self.config.validation_periods):
            period_end = eligible_end - period * 2 * validation_length
            period_start = period_end - validation_length + pd.Timedelta(hours=1)
            validation_masks.append((eligible_issues >= period_start) & (eligible_issues <= period_end))
        validation_mask = np.logical_or.reduce(validation_masks)
        validation_issues = eligible_issues[validation_mask]
        # Embargo issue rows whose 24-hour labels would overlap the first
        # validation target. This keeps validation outcomes unseen in fitting.
        validation_first_issue = (
            validation_issues[0]
            if not validation_issues.empty
            else eligible_end - validation_length + pd.Timedelta(hours=1)
        )
        trainable_issues = eligible_issues[
            eligible_issues + pd.Timedelta(hours=self.config.horizon_hours) <= validation_first_issue
        ]
        if len(validation_issues) < 24 or len(trainable_issues) < 100:
            raise ValueError("not enough fully observed samples for candidate validation")

        fit_end = trainable_issues[-1]
        if self.config.retraining_window_months is not None:
            start = fit_end - pd.DateOffset(months=self.config.retraining_window_months)
        else:
            start = fit_end - pd.Timedelta(days=self.config.retraining_window_days)
        recent_issues = trainable_issues[trainable_issues > start]
        train_issues = recent_issues

        if self.config.adaptation_strategy in {"recent_plus_history", "mixed_history_fine_tune"}:
            older_issues = trainable_issues[trainable_issues <= start]
            # Keep a deterministic, seasonally distributed sample of prior
            # months so the candidate retains annual/seasonal context.
            pieces = []
            month_keys = older_issues.strftime("%Y-%m")
            for _, group in pd.Series(older_issues, index=older_issues).groupby(month_keys, sort=True):
                stride = max(1, len(group) // self.config.representative_samples_per_month)
                pieces.extend(group.iloc[::stride].index.tolist()[: self.config.representative_samples_per_month])
            historical_issues = pd.DatetimeIndex(pieces)
            train_issues = historical_issues.append(recent_issues).sort_values()
        if len(train_issues) < 100:
            raise ValueError("not enough candidate training samples")
        X_train, y_train = self._build_selected_samples(observed, train_issues, old)
        X_validation, y_validation = self._build_selected_samples(observed, validation_issues, old)

        training_started = time.perf_counter()
        if self.config.adaptation_strategy in {
            "fine_tune", "gentle_fine_tune", "mixed_history_fine_tune"
        }:
            candidate_model: MLPRegressor = copy.deepcopy(old.model)
            X_scaled = old.feature_scaler.transform(X_train[old.feature_columns])
            y_scaled = old.target_scaler.transform(y_train)
            if self.config.adaptation_strategy == "fine_tune":
                for _ in range(self.config.fine_tune_epochs):
                    candidate_model.partial_fit(X_scaled, y_scaled)
            else:
                # sklearn's partial_fit continues the copied weights and the
                # copied Adam moments. Its public API has no learning-rate
                # override for an already-fitted estimator, so lower the
                # copied optimizer's step size without reinitializing it.
                optimizer = getattr(candidate_model, "_optimizer", None)
                if optimizer is None or not hasattr(optimizer, "learning_rate"):
                    raise RuntimeError("gentle fine-tuning requires sklearn's fitted Adam optimizer")
                lower_rate = float(optimizer.learning_rate) * self.config.fine_tune_learning_rate_factor
                optimizer.learning_rate = lower_rate
                optimizer.learning_rate_init = lower_rate
                candidate_model.learning_rate_init = lower_rate

                batch_size = min(self.config.fine_tune_batch_size, len(X_scaled))
                max_samples = min(len(X_scaled), self.config.fine_tune_max_updates * batch_size)
                rng = np.random.default_rng(self.config.fine_tune_random_state)
                selected = rng.permutation(len(X_scaled))[:max_samples]
                for offset in range(0, len(selected), batch_size):
                    batch = selected[offset : offset + batch_size]
                    candidate_model.partial_fit(X_scaled[batch], y_scaled[batch])
            candidate_training_seconds = time.perf_counter() - training_started
            candidate = DirectMLPForecaster(
                model=candidate_model,
                feature_scaler=old.feature_scaler,
                target_scaler=old.target_scaler,
                feature_spec=old.feature_spec,
                model_version=version,
                feature_columns=old.feature_columns,
                train_end_issue_time=X_train.index[-1],
                eligible_window_start=X_train.index[0],
                eligible_window_end=X_train.index[-1],
            )
        else:
            feature_scaler = StandardScaler().fit(X_train[old.feature_columns])
            target_scaler = StandardScaler().fit(y_train)
            model = MLPRegressor(
                hidden_layer_sizes=tuple(old.model.hidden_layer_sizes),
                activation=old.model.activation,
                solver="adam",
                max_iter=self.config.adaptation_max_iter,
                random_state=42,
                early_stopping=False,
            ).fit(feature_scaler.transform(X_train[old.feature_columns]), target_scaler.transform(y_train))
            candidate_training_seconds = time.perf_counter() - training_started
            candidate = DirectMLPForecaster(
                model=model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                feature_spec=old.feature_spec,
                model_version=version,
                feature_columns=old.feature_columns,
                train_end_issue_time=X_train.index[-1],
                eligible_window_start=X_train.index[0],
                eligible_window_end=X_train.index[-1],
            )

        # All validation labels end at or before the just-revealed timestamp.
        validation_latest_target = X_validation.index[-1] + pd.Timedelta(hours=self.config.horizon_hours)
        if validation_latest_target > timestamp:
            raise RuntimeError("candidate validation contains a target not yet observed")
        validation_first_target = X_validation.index[0] + pd.Timedelta(hours=1)
        validation_last_target = validation_latest_target
        return (
            candidate,
            X_validation,
            y_validation,
            X_train.index[0],
            X_train.index[-1],
            validation_first_target,
            validation_last_target,
            candidate_training_seconds,
        )

    def _fit_unvalidated_mixed_history_candidate(
        self, timestamp: pd.Timestamp, old: DirectMLPForecaster, version: str
    ) -> tuple[DirectMLPForecaster, pd.Timestamp, pd.Timestamp, float]:
        """Fine-tune on eligible recent rows plus representative older history.

        This is used by the synthetic demonstration only. It skips validation,
        but retains leakage-safe target eligibility and the existing scalers.
        """
        observed = self.demand.loc[:timestamp]
        first_position = old.feature_spec.lookback_hours - 1
        last_position_exclusive = len(observed) - old.feature_spec.horizon_hours
        if last_position_exclusive <= first_position:
            raise ValueError("not enough observed history for mixed-history adaptation")
        eligible_issues = pd.DatetimeIndex(
            observed.index[first_position:last_position_exclusive], name="issue_timestamp"
        )
        fit_end = eligible_issues[-1]
        if self.config.retraining_window_months is not None:
            recent_start = fit_end - pd.DateOffset(months=self.config.retraining_window_months)
        else:
            recent_start = fit_end - pd.Timedelta(days=self.config.retraining_window_days)
        recent_issues = eligible_issues[eligible_issues > recent_start]
        older_issues = eligible_issues[eligible_issues <= recent_start]

        historical_samples: list[pd.Timestamp] = []
        month_keys = older_issues.strftime("%Y-%m")
        for _, month in pd.Series(older_issues, index=older_issues).groupby(month_keys, sort=True):
            stride = max(1, len(month) // self.config.representative_samples_per_month)
            historical_samples.extend(
                month.iloc[::stride].index.tolist()[: self.config.representative_samples_per_month]
            )
        train_issues = pd.DatetimeIndex(historical_samples).append(recent_issues).sort_values()
        if len(train_issues) < 100:
            raise ValueError("not enough fully observed samples for mixed-history adaptation")
        X_train, y_train = self._build_selected_samples(observed, train_issues, old)
        latest_target = X_train.index[-1] + pd.Timedelta(hours=self.config.horizon_hours)
        if latest_target > timestamp:
            raise RuntimeError("mixed-history adaptation contains a target not yet observed")

        candidate_model: MLPRegressor = copy.deepcopy(old.model)
        X_scaled = old.feature_scaler.transform(X_train[old.feature_columns])
        y_scaled = old.target_scaler.transform(y_train)
        optimizer = getattr(candidate_model, "_optimizer", None)
        if optimizer is None or not hasattr(optimizer, "learning_rate"):
            raise RuntimeError("mixed-history fine-tuning requires the fitted Adam optimizer")
        lower_rate = float(optimizer.learning_rate) * self.config.fine_tune_learning_rate_factor
        optimizer.learning_rate = lower_rate
        optimizer.learning_rate_init = lower_rate
        candidate_model.learning_rate_init = lower_rate

        training_started = time.perf_counter()
        batch_size = min(self.config.fine_tune_batch_size, len(X_scaled))
        max_samples = min(len(X_scaled), self.config.fine_tune_max_updates * batch_size)
        rng = np.random.default_rng(self.config.fine_tune_random_state)
        selected = rng.permutation(len(X_scaled))[:max_samples]
        for offset in range(0, len(selected), batch_size):
            batch = selected[offset : offset + batch_size]
            candidate_model.partial_fit(X_scaled[batch], y_scaled[batch])
        training_seconds = time.perf_counter() - training_started
        candidate = DirectMLPForecaster(
            model=candidate_model,
            feature_scaler=old.feature_scaler,
            target_scaler=old.target_scaler,
            feature_spec=old.feature_spec,
            model_version=version,
            feature_columns=old.feature_columns,
            train_end_issue_time=X_train.index[-1],
            eligible_window_start=X_train.index[0],
            eligible_window_end=X_train.index[-1],
        )
        return candidate, X_train.index[0], X_train.index[-1], training_seconds

    def _adapt(self, timestamp: pd.Timestamp, change: ErrorDistributionChange | None) -> None:
        old = self.adaptive_model
        change_fields = self._event_fields(change)
        if not self._can_adapt(timestamp):
            self.events.append(
                AdaptationEvent(
                    timestamp, old.model_version, None, None, None, None, "cooldown_skipped", **change_fields
                )
            )
            return
        if self.config.adaptation_policy in {"increase_only", "validated"} and not self._alarm_is_meaningful_increase(change):
            self.events.append(
                AdaptationEvent(
                    timestamp, old.model_version, None, None, None, None,
                    "error_increase_not_meaningful", **change_fields
                )
            )
            return
        observed = self.demand.loc[:timestamp]
        adaptation_number = sum(event.outcome in {"adapted", "candidate_rejected"} for event in self.events) + 1
        version = f"{self.static_model.model_version}.adapt{adaptation_number}"
        if self.config.adaptation_policy == "validated":
            try:
                (
                    candidate,
                    X_validation,
                    y_validation,
                    train_start,
                    train_end,
                    validation_first_target,
                    validation_last_target,
                    candidate_training_seconds,
                ) = self._fit_validated_candidate(timestamp, old, version)
                incumbent_mae = float(
                    mean_absolute_error(y_validation.iloc[:, -1], self._validation_predictions(old, X_validation))
                )
                candidate_mae = float(
                    mean_absolute_error(
                        y_validation.iloc[:, -1], self._validation_predictions(candidate, X_validation)
                    )
                )
            except ValueError as exc:
                self.last_adaptation_timestamp = timestamp
                self.events.append(
                    AdaptationEvent(
                        timestamp, old.model_version, None, None, None, None,
                        f"candidate_rejected: {exc}", training_strategy=self.config.adaptation_strategy,
                        **change_fields
                    )
                )
                return
            improvement = (incumbent_mae - candidate_mae) / max(incumbent_mae, 1.0)
            latest_target = train_end + pd.Timedelta(hours=self.config.horizon_hours)
            if latest_target > timestamp:
                raise RuntimeError("candidate training contains a target not yet observed")
            accepted = improvement >= self.config.min_validation_improvement_fraction
            if accepted:
                self.adaptive_model = candidate
            self.last_adaptation_timestamp = timestamp
            self.events.append(
                AdaptationEvent(
                    timestamp,
                    old.model_version,
                    candidate.model_version if accepted else None,
                    train_start,
                    train_end,
                    latest_target,
                    "adapted" if accepted else "candidate_rejected",
                    candidate_validation_mae_mw=candidate_mae,
                    incumbent_validation_mae_mw=incumbent_mae,
                    validation_improvement_fraction=improvement,
                    training_strategy=self.config.adaptation_strategy,
                    validation_first_target_timestamp=validation_first_target,
                    validation_last_target_timestamp=validation_last_target,
                    candidate_training_seconds=candidate_training_seconds,
                    **change_fields,
                )
            )
            return
        if (
            self.config.adaptation_policy == "increase_only"
            and self.config.adaptation_strategy == "mixed_history_fine_tune"
        ):
            try:
                candidate, train_start, train_end, training_seconds = (
                    self._fit_unvalidated_mixed_history_candidate(timestamp, old, version)
                )
            except ValueError as exc:
                self.events.append(
                    AdaptationEvent(
                        timestamp, old.model_version, None, None, None, None,
                        f"skipped: {exc}", training_strategy=self.config.adaptation_strategy,
                        **change_fields,
                    )
                )
                return
            latest_target = train_end + pd.Timedelta(hours=self.config.horizon_hours)
            self.adaptive_model = candidate
            self.last_adaptation_timestamp = timestamp
            self.events.append(
                AdaptationEvent(
                    timestamp,
                    old.model_version,
                    candidate.model_version,
                    train_start,
                    train_end,
                    latest_target,
                    "adapted",
                    training_strategy=self.config.adaptation_strategy,
                    candidate_training_seconds=training_seconds,
                    **change_fields,
                )
            )
            return
        try:
            trained: TrainingResult = train_direct_mlp(
                observed,
                feature_spec=old.feature_spec,
                hidden_layer_sizes=tuple(old.model.hidden_layer_sizes),
                max_iter=self.config.adaptation_max_iter,
                random_state=old.model.random_state,
                validation_fraction=self.config.adaptation_validation_fraction,
                training_window_days=self.config.retraining_window_days,
                model_version=version,
            )
        except ValueError as exc:
            self.events.append(
                AdaptationEvent(
                    timestamp, old.model_version, None, None, None, None, f"skipped: {exc}", **change_fields
                )
            )
            return
        latest_target = trained.eligible_window_end + pd.Timedelta(hours=self.config.horizon_hours)
        if latest_target > timestamp:
            raise RuntimeError("adaptation attempted to use a training target that is not observed yet")
        self.adaptive_model = trained.model
        self.last_adaptation_timestamp = timestamp
        self.events.append(
            AdaptationEvent(
                timestamp,
                old.model_version,
                trained.model.model_version,
                trained.eligible_window_start,
                trained.eligible_window_end,
                latest_target,
                "adapted",
                **change_fields,
            )
        )

    def run(
        self,
        *,
        start_timestamp: pd.Timestamp | None = None,
        end_timestamp: pd.Timestamp | None = None,
    ) -> ReplayResult:
        started = time.perf_counter()
        static_rows: list[pd.DataFrame] = []
        adaptive_rows: list[pd.DataFrame] = []
        inference_runtime = 0.0
        stop = pd.Timestamp(end_timestamp) if end_timestamp is not None else self.demand.index[-1]
        stop = stop.tz_localize("UTC") if stop.tz is None else stop.tz_convert("UTC")
        first_position = self._first_issue_position()
        if start_timestamp is not None:
            start = pd.Timestamp(start_timestamp)
            start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
            first_position = max(first_position, int(self.demand.index.searchsorted(start, side="left")))

        for position in range(first_position, len(self.demand)):
            timestamp = self.demand.index[position]
            if timestamp > stop:
                break
            actual = float(self.demand.iloc[position])
            static_match = self.static_registry.fixed_horizon_errors(
                timestamp, actual, self.config.horizon_hours
            )
            adaptive_match = self.adaptive_registry.fixed_horizon_errors(
                timestamp, actual, self.config.horizon_hours
            )
            if not static_match.empty:
                static_rows.append(static_match)
            if not adaptive_match.empty:
                adaptive_rows.append(adaptive_match)
                drift = self.detector.update(float(adaptive_match.iloc[0]["absolute_error_mw"]))
                if drift:
                    self._adapt(timestamp, getattr(self.detector, "last_change", None))

            # The feature specification needs precisely this past-only window.
            # Supplying the full replay history would repeatedly validate data
            # that cannot affect the current forecast.
            lookback = self.static_model.feature_spec.lookback_hours
            history = self.demand.iloc[max(0, position + 1 - lookback) : position + 1]
            inference_started = time.perf_counter()
            self.static_registry.record_horizon(
                timestamp,
                self.static_model.predict_horizon(history, timestamp),
                self.static_model.model_version,
            )
            self.adaptive_registry.record_horizon(
                timestamp,
                self.adaptive_model.predict_horizon(history, timestamp),
                self.adaptive_model.model_version,
            )
            inference_runtime += time.perf_counter() - inference_started

        empty = pd.DataFrame()
        return ReplayResult(
            static_errors=pd.concat(static_rows, ignore_index=True) if static_rows else empty,
            adaptive_errors=pd.concat(adaptive_rows, ignore_index=True) if adaptive_rows else empty,
            adaptation_events=list(self.events),
            static_registry=self.static_registry,
            adaptive_registry=self.adaptive_registry,
            final_adaptive_model=self.adaptive_model,
            runtime_seconds=time.perf_counter() - started,
            inference_runtime_seconds=inference_runtime,
        )
