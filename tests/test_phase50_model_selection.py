from griddrift.dashboard.model_selection import (
    ADAPTIVE_MLP,
    ADAPTIVE_XGBOOST,
    DEFAULT_FORECASTING_MODEL,
    FORECASTING_MODEL_LABELS,
    initialize_forecasting_model,
    model_checkpoint_version,
    model_switch_pending,
    select_forecasting_model,
)

ROOT = "."


def test_new_session_defaults_to_xgboost_and_legacy_session_keeps_mlp():
    new_session = {}
    assert initialize_forecasting_model(
        new_session, has_existing_mlp_session=False
    ) == ADAPTIVE_XGBOOST
    assert new_session["selected_forecasting_model"] == DEFAULT_FORECASTING_MODEL

    legacy_session = {}
    assert initialize_forecasting_model(
        legacy_session, has_existing_mlp_session=True
    ) == ADAPTIVE_MLP


def test_selection_is_persistent_pauses_playback_and_marks_other_replay_pending():
    simulation = object()
    state = {
        "selected_forecasting_model": ADAPTIVE_MLP,
        "active_forecasting_model": ADAPTIVE_MLP,
        "simulation": simulation,
        "running": True,
    }

    assert select_forecasting_model(state, ADAPTIVE_XGBOOST) is True
    assert state["selected_forecasting_model"] == ADAPTIVE_XGBOOST
    assert state["running"] is False
    assert model_switch_pending(ADAPTIVE_XGBOOST, state["active_forecasting_model"])
    # The old replay remains explicitly MLP-owned; it is never relabeled as XGBoost.
    assert state["simulation"] is simulation


def test_checkpoint_versions_are_read_from_the_matching_metadata():
    assert model_checkpoint_version(ADAPTIVE_XGBOOST, ROOT) == "xgb-de-lu-static-42"
    assert model_checkpoint_version(ADAPTIVE_MLP, ROOT) == "mlp-1209a11e1cf2"
    assert FORECASTING_MODEL_LABELS[ADAPTIVE_XGBOOST] == "Adaptive XGBoost"
