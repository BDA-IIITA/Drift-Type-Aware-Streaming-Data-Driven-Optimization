"""Leakage-safe fixed-24-hour ADWIN replay for direct XGBoost models."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from griddrift.drift import ADWINConfig, PersistentADWIN
from griddrift.forecasting.features import (
    _validate_demand,
    build_feature_row,
    build_supervised_dataset,
)
from griddrift.forecasting.registry import ForecastRegistry
from griddrift.forecasting.xgboost import DirectXGBoostForecaster, fit_direct_xgboost_samples
from griddrift.simulation.replay import AdaptationEvent


@dataclass(frozen=True)
class XGBoostReplayConfig:
    strategy: str = "recent"
    recent_window_days: int = 60
    cooldown_days: int = 7
    validation_days: int = 7
    min_error_increase_mw: float = 250.0
    min_error_increase_fraction: float = 0.10
    min_validation_improvement_fraction: float = 0.02
    random_state: int = 42
    adwin: ADWINConfig = field(default_factory=ADWINConfig)

    def __post_init__(self) -> None:
        if self.strategy not in {"full_history", "recent"}:
            raise ValueError("strategy must be full_history or recent")
        if self.recent_window_days < 1 or self.cooldown_days < 0 or self.validation_days < 1:
            raise ValueError("window and validation days must be positive; cooldown non-negative")


@dataclass
class XGBoostReplayResult:
    static_errors: pd.DataFrame
    adaptive_errors: pd.DataFrame
    adaptation_events: pd.DataFrame
    static_registry: ForecastRegistry
    adaptive_registry: ForecastRegistry
    final_model: DirectXGBoostForecaster
    candidate_models: dict[str, DirectXGBoostForecaster]
    runtime_seconds: float
    inference_runtime_seconds: float


class XGBoostStreamingReplay:
    """Persistent, step-triggered XGBoost ADWIN and validated adaptation state."""

    def __init__(
        self,
        frozen_model: DirectXGBoostForecaster,
        *,
        config: XGBoostReplayConfig | None = None,
        detector: object | None = None,
    ) -> None:
        self.static_model = frozen_model
        self.adaptive_model = copy.deepcopy(frozen_model)
        self.config = config or XGBoostReplayConfig(
            strategy="full_history", min_validation_improvement_fraction=0.005
        )
        self.detector = detector or PersistentADWIN(self.config.adwin)
        self.static_registry = ForecastRegistry()
        self.adaptive_registry = ForecastRegistry()
        self.events: list[AdaptationEvent] = []
        self.candidate_models: dict[str, DirectXGBoostForecaster] = {}
        self.last_attempt_timestamp: pd.Timestamp | None = None
        self.attempt_count = 0

    @staticmethod
    def _event_fields(change) -> dict[str, object]:
        return {
            "error_mean_before_mw": getattr(change, "error_mean_before_mw", None),
            "error_mean_after_mw": getattr(change, "error_mean_after_mw", None),
            "error_change_direction": getattr(change, "direction", None),
            "adwin_window_width_before": getattr(change, "window_width_before", None),
            "adwin_window_width_after": getattr(change, "window_width_after", None),
        }

    def observe_error(self, timestamp: pd.Timestamp, absolute_error_mw: float, observed: pd.Series) -> bool:
        """Update this model family's detector once for an arrived 24h target."""
        if not self.detector.update(float(absolute_error_mw)):
            return False
        change = getattr(self.detector, "last_change", None)
        self._adapt(pd.Timestamp(timestamp), change, observed)
        return True

    def _adapt(self, timestamp: pd.Timestamp, change, observed: pd.Series) -> None:
        old = self.adaptive_model
        fields = self._event_fields(change)
        direction = getattr(change, "direction", None)
        before = float(getattr(change, "error_mean_before_mw", 0.0) or 0.0)
        after = float(getattr(change, "error_mean_after_mw", 0.0) or 0.0)
        increase = after - before
        meaningful = (
            direction in {"error_increased", "increase"}
            and increase >= self.config.min_error_increase_mw
            and increase / max(before, 1.0) >= self.config.min_error_increase_fraction
        )
        if not meaningful:
            self.events.append(AdaptationEvent(
                timestamp, old.model_version, None, None, None, None,
                "error_increase_not_meaningful", training_strategy="full_history", **fields,
            ))
            return
        if (
            self.last_attempt_timestamp is not None
            and timestamp < self.last_attempt_timestamp + pd.Timedelta(days=self.config.cooldown_days)
        ):
            self.events.append(AdaptationEvent(
                timestamp, old.model_version, None, None, None, None,
                "cooldown_skipped", training_strategy="full_history", **fields,
            ))
            return

        self.last_attempt_timestamp = timestamp
        self.attempt_count += 1
        version = f"{self.static_model.model_version}.adapt{self.attempt_count}"
        try:
            observed = observed.loc[:timestamp]
            X, y = build_supervised_dataset(observed, old.feature_spec)
            eligible_issues = X.index[y.notna().all(axis=1).to_numpy()]
            validation_start_target = timestamp - pd.Timedelta(days=self.config.validation_days) + pd.Timedelta(hours=1)
            validation_issues = eligible_issues[
                (eligible_issues + pd.Timedelta(hours=24) >= validation_start_target)
                & (eligible_issues + pd.Timedelta(hours=24) <= timestamp)
            ]
            if len(validation_issues) < 24:
                raise ValueError("insufficient fully observed validation samples")
            first_validation_issue = validation_issues[0]
            train_issues = eligible_issues[
                eligible_issues + pd.Timedelta(hours=24) < first_validation_issue
            ]
            if len(train_issues) < 60:
                raise ValueError("insufficient fully observed full-history training samples")

            started = time.perf_counter()
            candidate = fit_direct_xgboost_samples(
                X.loc[train_issues], y.loc[train_issues],
                feature_spec=old.feature_spec,
                parameters=old.parameters,
                random_state=self.config.random_state,
                model_version=version,
            )
            training_seconds = time.perf_counter() - started
            self.candidate_models[version] = candidate
            validation_y = y.loc[validation_issues].iloc[:, -1]
            candidate_mae = float(mean_absolute_error(
                validation_y, candidate.predict_feature_frame(X.loc[validation_issues])[:, -1]
            ))
            incumbent_mae = float(mean_absolute_error(
                validation_y, old.predict_feature_frame(X.loc[validation_issues])[:, -1]
            ))
            improvement = (incumbent_mae - candidate_mae) / max(incumbent_mae, 1.0)
            accepted = improvement >= self.config.min_validation_improvement_fraction
            if accepted:
                self.adaptive_model = candidate
            train_start, train_end = train_issues[0], train_issues[-1]
            latest_target = train_end + pd.Timedelta(hours=24)
            validation_first = validation_issues[0] + pd.Timedelta(hours=24)
            validation_last = validation_issues[-1] + pd.Timedelta(hours=24)
            if latest_target > timestamp or validation_last > timestamp:
                raise RuntimeError("XGBoost candidate used a target not observed at adaptation time")
            self.events.append(AdaptationEvent(
                timestamp, old.model_version, candidate.model_version if accepted else None,
                train_start, train_end, latest_target,
                "adapted" if accepted else "candidate_rejected",
                candidate_validation_mae_mw=candidate_mae,
                incumbent_validation_mae_mw=incumbent_mae,
                validation_improvement_fraction=improvement,
                training_strategy="full_history",
                validation_first_target_timestamp=validation_first,
                validation_last_target_timestamp=validation_last,
                candidate_training_seconds=training_seconds,
                **fields,
            ))
        except (ValueError, RuntimeError) as exc:
            self.events.append(AdaptationEvent(
                timestamp, old.model_version, None, None, None, None,
                f"candidate_rejected: {exc}", training_strategy="full_history", **fields,
            ))


def _metric(errors: pd.DataFrame) -> dict[str, float]:
    values = errors["absolute_error_mw"].to_numpy(dtype=float)
    if not len(values):
        return {"mae_mw": float("nan"), "rmse_mw": float("nan"), "count": 0.0}
    return {
        "mae_mw": float(values.mean()),
        "rmse_mw": float(np.sqrt(np.mean(values**2))),
        "count": float(len(values)),
    }


def run_xgboost_replay(
    demand: pd.Series,
    frozen_model: DirectXGBoostForecaster,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    evaluation_start: pd.Timestamp | None = None,
    config: XGBoostReplayConfig | None = None,
    parameters: dict[str, object] | None = None,
    detector: object | None = None,
) -> XGBoostReplayResult:
    """Reveal actuals sequentially; only fully revealed samples may train candidates."""
    started = time.perf_counter()
    config = config or XGBoostReplayConfig()
    demand = _validate_demand(demand)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    evaluation_start = pd.Timestamp(evaluation_start or start)
    evaluation_start = evaluation_start.tz_localize("UTC") if evaluation_start.tz is None else evaluation_start.tz_convert("UTC")
    if start < frozen_model.train_end_issue_time + pd.Timedelta(hours=24):
        raise ValueError("replay cannot start until frozen-model training targets have been observed")
    if end > demand.index[-1] or end < start:
        raise ValueError("replay interval is outside the available demand observations")

    # Materialize no observations after replay end. Recent-window runs also retain only
    # the history they can legally use (plus feature lookback and validation embargo).
    # The forecast history remains the same past-only demand series at each issue time.
    sample_demand = demand.loc[:end]
    if config.strategy == "recent":
        context_days = config.recent_window_days + config.validation_days + 1
        context_start = start - pd.Timedelta(days=context_days) - pd.Timedelta(
            hours=frozen_model.feature_spec.lookback_hours
        )
        sample_demand = sample_demand.loc[context_start:]
    X, y = build_supervised_dataset(sample_demand, frozen_model.feature_spec)

    replay_index = demand.index[(demand.index >= start) & (demand.index <= end)]
    issue_rows = X.reindex(replay_index)
    missing_issue_features = issue_rows.isna().all(axis=1)
    if missing_issue_features.any():
        future_issue_rows = [
            build_feature_row(demand.loc[:issue], issue, frozen_model.feature_spec)
            for issue in replay_index[missing_issue_features.to_numpy()]
        ]
        issue_rows.loc[missing_issue_features] = pd.DataFrame(
            future_issue_rows, index=replay_index[missing_issue_features.to_numpy()]
        ).loc[:, issue_rows.columns]
    inference_started = time.perf_counter()
    static_predictions = frozen_model.predict_feature_frame(issue_rows)
    inference_runtime = time.perf_counter() - inference_started
    static_fixed: dict[pd.Timestamp, float] = {}
    for issue, values in zip(issue_rows.index, static_predictions):
        static_fixed[issue + pd.Timedelta(hours=24)] = float(values[-1])

    static_registry, adaptive_registry = ForecastRegistry(), ForecastRegistry()
    static_rows: list[dict[str, object]] = []
    adaptive_rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    candidate_models: dict[str, DirectXGBoostForecaster] = {}
    adaptive_model = copy.deepcopy(frozen_model)
    adaptive_predictions: dict[pd.Timestamp, tuple[float, str]] = {}
    detector = detector or PersistentADWIN(config.adwin)
    last_attempt: pd.Timestamp | None = None
    attempt_number = 0

    for timestamp in replay_index:
        # At this instant actual demand at `timestamp` has just arrived. Compare it only
        # with predictions issued exactly 24 hours earlier (the fixed 24h lead).
        if timestamp in static_fixed and timestamp in adaptive_predictions:
            actual = float(demand.loc[timestamp])
            static_error = abs(actual - static_fixed[timestamp])
            adaptive_prediction, forecast_model_version = adaptive_predictions[timestamp]
            adaptive_error = abs(actual - adaptive_prediction)
            if timestamp >= evaluation_start:
                static_rows.append({"target_timestamp": timestamp, "issue_timestamp": timestamp - pd.Timedelta(hours=24),
                                    "predicted_demand_mw": static_fixed[timestamp], "actual_demand_mw": actual,
                                    "absolute_error_mw": static_error, "model_version": frozen_model.model_version})
                adaptive_rows.append({"target_timestamp": timestamp, "issue_timestamp": timestamp - pd.Timedelta(hours=24),
                                      "predicted_demand_mw": adaptive_prediction, "actual_demand_mw": actual,
                                      "absolute_error_mw": adaptive_error, "model_version": forecast_model_version})
            drift = detector.update(adaptive_error)
            change = detector.last_change if drift else None
            if drift:
                now = timestamp
                before = float(change.error_mean_before_mw) if change else 0.0
                after = float(change.error_mean_after_mw) if change else 0.0
                increase = after - before
                meaningful = increase > 0 and increase >= max(
                    config.min_error_increase_mw, config.min_error_increase_fraction * max(before, 1.0)
                )
                event: dict[str, object] = {
                    "drift_timestamp": now, "previous_model_version": adaptive_model.model_version,
                    "new_model_version": None, "error_mean_before_mw": before,
                    "error_mean_after_mw": after, "error_direction": change.direction if change else "unknown",
                    "strategy": config.strategy, "window_days": config.recent_window_days if config.strategy == "recent" else None,
                    "candidate_validation_mae_mw": None, "incumbent_validation_mae_mw": None,
                    "validation_improvement_fraction": None, "training_seconds": 0.0, "decision": "skipped",
                }
                in_cooldown = last_attempt is not None and now < last_attempt + pd.Timedelta(days=config.cooldown_days)
                if not meaningful:
                    event["decision"] = "skipped_not_meaningful_error_increase"
                elif in_cooldown:
                    event["decision"] = "skipped_cooldown"
                else:
                    last_attempt = now
                    attempt_number += 1
                    # Every candidate label must end no later than `now`; the most recent
                    # 7 days form validation and a 24h issue-time embargo precedes them.
                    eligible = X.index + pd.Timedelta(hours=24) <= now
                    eligible_issues = X.index[eligible]
                    validation_start_target = now - pd.Timedelta(days=config.validation_days) + pd.Timedelta(hours=1)
                    validation_issues = eligible_issues[
                        (eligible_issues + pd.Timedelta(hours=24) >= validation_start_target)
                        & (eligible_issues + pd.Timedelta(hours=24) <= now)
                    ]
                    if len(validation_issues) >= 24:
                        first_val_issue = validation_issues[0]
                        train_issues = eligible_issues[eligible_issues + pd.Timedelta(hours=24) < first_val_issue]
                        if config.strategy == "recent" and len(train_issues):
                            train_issues = train_issues[
                                train_issues > train_issues[-1] - pd.Timedelta(days=config.recent_window_days)
                            ]
                        if len(train_issues) >= 60:
                            fit_started = time.perf_counter()
                            try:
                                candidate = fit_direct_xgboost_samples(
                                    X.loc[train_issues], y.loc[train_issues],
                                    feature_spec=frozen_model.feature_spec,
                                    parameters=parameters or frozen_model.parameters,
                                    random_state=config.random_state,
                                    model_version=f"{frozen_model.model_version}.adapt{attempt_number}",
                                )
                                candidate_models[candidate.model_version] = candidate
                                event["training_seconds"] = time.perf_counter() - fit_started
                                event["training_first_issue_timestamp"] = train_issues[0]
                                event["training_last_issue_timestamp"] = train_issues[-1]
                                event["training_latest_target_timestamp"] = train_issues[-1] + pd.Timedelta(hours=24)
                                event["validation_first_target_timestamp"] = validation_issues[0] + pd.Timedelta(hours=24)
                                event["validation_last_target_timestamp"] = validation_issues[-1] + pd.Timedelta(hours=24)
                                val_X, val_y = X.loc[validation_issues], y.loc[validation_issues].iloc[:, -1]
                                candidate_mae = mean_absolute_error(val_y, candidate.predict_feature_frame(val_X)[:, -1])
                                incumbent_mae = mean_absolute_error(val_y, adaptive_model.predict_feature_frame(val_X)[:, -1])
                                improvement = (incumbent_mae - candidate_mae) / max(float(incumbent_mae), 1.0)
                                event["candidate_validation_mae_mw"] = float(candidate_mae)
                                event["incumbent_validation_mae_mw"] = float(incumbent_mae)
                                event["validation_improvement_fraction"] = float(improvement)
                                accepted = improvement >= config.min_validation_improvement_fraction
                                event["decision"] = "accepted" if accepted else "rejected_validation"
                                if accepted:
                                    adaptive_model = candidate
                                    event["new_model_version"] = candidate.model_version
                            except (ValueError, RuntimeError) as exc:
                                event["decision"] = f"rejected_candidate_error: {exc}"
                        else:
                            event["decision"] = "rejected_insufficient_training_samples"
                    else:
                        event["decision"] = "rejected_insufficient_validation_samples"
                events.append(event)

        # Generate the complete future horizon after scoring/adaptation. This forecast is
        # issued at the current step and its target values remain hidden until arrival.
        if timestamp in issue_rows.index:
            row_position = issue_rows.index.get_loc(timestamp)
            if adaptive_model.model_version == frozen_model.model_version:
                # The frozen model's batch prediction is exactly the deployed model output
                # until an accepted update changes its version; reuse it without extra inference.
                prediction_values = static_predictions[row_position]
            else:
                inference_started = time.perf_counter()
                prediction_values = adaptive_model.predict_horizon(
                    demand.loc[:timestamp], timestamp
                ).to_numpy(dtype=float)
                inference_runtime += time.perf_counter() - inference_started
            static_horizon = pd.Series(
                static_predictions[row_position],
                index=pd.date_range(timestamp + pd.Timedelta(hours=1), periods=24, freq="h"),
            )
            prediction = pd.Series(
                prediction_values,
                index=pd.date_range(timestamp + pd.Timedelta(hours=1), periods=24, freq="h"),
                name="predicted_demand_mw",
            )
            static_registry.record_horizon(timestamp, static_horizon, frozen_model.model_version)
            adaptive_registry.record_horizon(timestamp, prediction, adaptive_model.model_version)
            adaptive_predictions[timestamp + pd.Timedelta(hours=24)] = (
                float(prediction.iloc[-1]), adaptive_model.model_version
            )

    return XGBoostReplayResult(
        static_errors=pd.DataFrame(static_rows), adaptive_errors=pd.DataFrame(adaptive_rows),
        adaptation_events=pd.DataFrame(events), static_registry=static_registry,
        adaptive_registry=adaptive_registry, final_model=adaptive_model, candidate_models=candidate_models,
        runtime_seconds=time.perf_counter() - started, inference_runtime_seconds=inference_runtime,
    )
