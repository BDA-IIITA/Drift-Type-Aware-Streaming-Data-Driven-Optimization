"""Audit saved Phase 3 forecast-key/version inputs for Phase 4.5; no dispatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from griddrift.simulation.phase45_inputs import load_phase45_forecast_vectors

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "artifacts/experiments/phase38/comparison/common-period-20261024/aligned_predictions.csv"
MLP_REGISTRY = ROOT / "artifacts/experiments/phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct/adaptive_forecast_registry.csv"
XGB_REGISTRY = ROOT / "artifacts/experiments/phase37/final/final-holdout-20261024/full_history/adaptive_forecast_registry.csv"
OUTPUT = ROOT / "artifacts/experiments/phase45/stage2a_input_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite completed Stage 2A audit: {OUTPUT}")
    aligned = pd.read_csv(ALIGNED)
    mlp = pd.read_csv(MLP_REGISTRY)
    xgb = pd.read_csv(XGB_REGISTRY)
    loaded = load_phase45_forecast_vectors(aligned, mlp, xgb)
    if not loaded.vectors:
        raise RuntimeError("no complete matched forecast vectors were found")
    result = {
        "phase": "4.5",
        "checkpoint": "stage2a_forecast_inputs",
        "status": "complete",
        "dispatch_optimization_run": False,
        "forecasting_models_retrained": False,
        "sources": {
            "phase38_aligned_predictions": {"path": str(ALIGNED.relative_to(ROOT)), "sha256": sha256(ALIGNED)},
            "phase36_adaptive_mlp_registry": {"path": str(MLP_REGISTRY.relative_to(ROOT)), "sha256": sha256(MLP_REGISTRY)},
            "phase37_adaptive_xgboost_registry": {"path": str(XGB_REGISTRY.relative_to(ROOT)), "sha256": sha256(XGB_REGISTRY)},
        },
        "aligned_prediction_key_count": loaded.aligned_prediction_keys,
        "complete_issue_vectors": len(loaded.vectors),
        "partial_boundary_issue_times_excluded": loaded.partial_issue_count,
        "first_complete_issue_timestamp": loaded.vectors[0].issue_timestamp.isoformat(),
        "last_complete_issue_timestamp": loaded.vectors[-1].issue_timestamp.isoformat(),
        "forecast_horizon_hours": 24,
        "target_alignment_verified": True,
        "adaptive_registry_prediction_values_match_phase38": True,
        "model_versions_preserved": True,
        "actual_demand_in_forecast_vector_interface": False,
        "actual_demand_used_to_create_forecast_vector": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
