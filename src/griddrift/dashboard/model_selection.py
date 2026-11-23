"""Session-safe model-family selection metadata for the dashboard."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from pathlib import Path

ADAPTIVE_XGBOOST = "xgboost"
ADAPTIVE_MLP = "mlp"
DEFAULT_FORECASTING_MODEL = ADAPTIVE_XGBOOST
FORECASTING_MODEL_LABELS = {
    ADAPTIVE_XGBOOST: "Adaptive XGBoost",
    ADAPTIVE_MLP: "Adaptive MLP",
}


def initialize_forecasting_model(
    session_state: MutableMapping,
    *,
    has_existing_mlp_session: bool,
) -> str:
    """Default new sessions to XGBoost; preserve an existing pre-selector MLP session."""
    selected = session_state.get("selected_forecasting_model")
    if selected not in FORECASTING_MODEL_LABELS:
        selected = "mlp" if has_existing_mlp_session else DEFAULT_FORECASTING_MODEL
        session_state["selected_forecasting_model"] = selected
    return selected


def select_forecasting_model(session_state: MutableMapping, model_family: str) -> bool:
    """Persist the selection and pause playback when the requested family changes."""
    if model_family not in FORECASTING_MODEL_LABELS:
        raise ValueError(f"unsupported forecasting model: {model_family}")
    previous = session_state.get("selected_forecasting_model")
    session_state["selected_forecasting_model"] = model_family
    changed = previous != model_family
    if changed:
        session_state["running"] = False
        session_state["model_switch_pending"] = True
    return changed


def model_checkpoint_version(model_family: str, repository_root: str | Path) -> str:
    """Read the saved checkpoint version without loading or training the estimator."""
    root = Path(repository_root)
    if model_family == ADAPTIVE_XGBOOST:
        metadata_path = root / "artifacts/experiments/phase37/static/static-multioutput-20261023/static_xgboost_model/metadata.json"
    elif model_family == ADAPTIVE_MLP:
        metadata_path = root / "artifacts/models/de-lu-mlp/metadata.json"
    else:
        raise ValueError(f"unsupported forecasting model: {model_family}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return str(metadata["model_version"])


def load_selected_forecaster(
    model_family: str,
    *,
    mlp_checkpoint: str | Path,
    xgboost_checkpoint: str | Path,
):
    """Load exactly the selected checkpoint, without fitting or fallback behavior."""
    if model_family == ADAPTIVE_XGBOOST:
        from griddrift.forecasting.xgboost import DirectXGBoostForecaster

        return DirectXGBoostForecaster.load(xgboost_checkpoint)
    if model_family == ADAPTIVE_MLP:
        from griddrift.forecasting.mlp import DirectMLPForecaster

        return DirectMLPForecaster.load(mlp_checkpoint)
    raise ValueError(f"unsupported forecasting model: {model_family}")


def load_selected_adaptation_config(
    model_family: str,
    repository_root: str | Path,
):
    """Build runtime settings from the frozen Phase 3.7/3.9 selections."""
    root = Path(repository_root)
    if model_family == ADAPTIVE_XGBOOST:
        from griddrift.drift import ADWINConfig
        from griddrift.simulation.xgboost_replay import XGBoostReplayConfig

        path = root / "artifacts/experiments/phase37/final/final-holdout-20261024/full_history/summary.json"
        with path.open(encoding="utf-8") as handle:
            selected = json.load(handle)
        return XGBoostReplayConfig(
            strategy=str(selected["strategy"]),
            cooldown_days=int(selected["cooldown_days"]),
            validation_days=7,
            min_validation_improvement_fraction=float(selected["validation_threshold_fraction"]),
            random_state=int(selected["seed"]),
            adwin=ADWINConfig(**selected["adwin"]),
        )
    if model_family == ADAPTIVE_MLP:
        from griddrift.drift import ADWINConfig
        from griddrift.simulation.replay import ReplayConfig

        path = root / "artifacts/experiments/phase39/selected_configuration.json"
        with path.open(encoding="utf-8") as handle:
            selected_root = json.load(handle)
        selected = selected_root["frozen_experiment_configuration"]
        return ReplayConfig(
            adaptation_policy="validated",
            adaptation_strategy=str(selected["adaptation_strategy"]),
            retraining_window_days=int(selected["retraining_window_days"]),
            cooldown_days=int(selected["cooldown_days"]),
            validation_days=int(selected["validation_days"]),
            min_error_increase_mw=float(selected["min_error_increase_mw"]),
            min_error_increase_fraction=float(selected["min_error_increase_fraction"]),
            min_validation_improvement_fraction=float(selected_root["selected_threshold_fraction"]),
            fine_tune_epochs=int(selected["fine_tune_epochs"]),
            fine_tune_learning_rate_factor=float(selected["fine_tune_learning_rate_factor"]),
            fine_tune_max_updates=int(selected["fine_tune_max_updates"]),
            fine_tune_batch_size=int(selected["fine_tune_batch_size"]),
            fine_tune_random_state=int(selected["fine_tune_random_state"]),
            representative_samples_per_month=int(selected["representative_samples_per_month"]),
            adwin=ADWINConfig(**selected["adwin"]),
        )
    raise ValueError(f"unsupported forecasting model: {model_family}")


def model_switch_pending(selected_family: str, active_family: str | None) -> bool:
    """Whether the visible replay still belongs to a different model family."""
    return active_family is None or selected_family != active_family
