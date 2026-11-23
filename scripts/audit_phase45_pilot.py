"""Preflight the predeclared Phase 4.5 pilot window without solving dispatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from griddrift.simulation.phase45_dispatch import _utc_series
from griddrift.simulation.phase45_inputs import load_phase45_forecast_vectors

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2025-01-01T00:00:00Z")
END = pd.Timestamp("2025-01-07T23:00:00Z")
OUT = ROOT / "artifacts/experiments/phase45/pilot/preregistration_phase45_pilot_2025w01_audited-output.json"
ALIGNED = ROOT / "artifacts/experiments/phase38/comparison/common-period-20261024/aligned_predictions.csv"
MLP_REGISTRY = ROOT / "artifacts/experiments/phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct/adaptive_forecast_registry.csv"
XGB_REGISTRY = ROOT / "artifacts/experiments/phase37/final/final-holdout-20261024/full_history/adaptive_forecast_registry.csv"
CONFIGS = [ROOT / "configs/assets.yaml", ROOT / "configs/integration.yaml"]
CODE = [
    ROOT / "src/griddrift/dispatch/engine.py",
    ROOT / "src/griddrift/simulation/phase45_dispatch.py",
    ROOT / "scripts/run_phase45_dispatch_comparison.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite pilot preregistration: {OUT}")
    aligned = pd.read_csv(ALIGNED)
    actual_frame = aligned[["target_timestamp", "actual_demand_mw"]].copy()
    actual_frame["target_timestamp"] = pd.to_datetime(actual_frame.target_timestamp, utc=True, errors="raise")
    actual_frame["actual_demand_mw"] = pd.to_numeric(actual_frame.actual_demand_mw, errors="raise")
    counts = actual_frame.groupby("target_timestamp").actual_demand_mw.nunique(dropna=False)
    if (counts != 1).any():
        raise ValueError("actual demand is inconsistent across duplicate target timestamps")
    actuals = actual_frame.drop_duplicates("target_timestamp").set_index("target_timestamp").actual_demand_mw
    actuals = _utc_series(actuals)
    vectors = load_phase45_forecast_vectors(
        aligned,
        pd.read_csv(MLP_REGISTRY),
        pd.read_csv(XGB_REGISTRY),
    )
    selected = [vector for vector in vectors.vectors if START <= vector.issue_timestamp <= END]
    expected = pd.date_range(START, END, freq="h")
    observed = pd.DatetimeIndex([vector.issue_timestamp for vector in selected])
    if not observed.equals(expected):
        raise ValueError(f"expected 168 contiguous full forecast vectors; found {len(selected)}")
    execution_targets = pd.DatetimeIndex([vector.target_timestamps[0] for vector in selected])
    missing_actuals = execution_targets.difference(actuals.index)
    if not missing_actuals.empty:
        raise ValueError(f"missing pilot execution actuals: {missing_actuals.tolist()[:5]}")
    if execution_targets[0] != pd.Timestamp("2025-01-01T01:00:00Z") or execution_targets[-1] != pd.Timestamp("2025-01-08T00:00:00Z"):
        raise ValueError("pilot actual execution coverage has unexpected first/last target")

    files = [ALIGNED, MLP_REGISTRY, XGB_REGISTRY, *CONFIGS, *CODE]
    result = {
        "phase": "4.5",
        "stage": "3_short_pilot",
        "status": "preflight_passed_not_yet_run",
        "preflight_revision": "after solver-boundary normalization and full recourse schedule logging; prior preflights preserved",
        "issue_window_utc_inclusive": {"start": START.isoformat(), "end": END.isoformat()},
        "issue_count": len(selected),
        "complete_forecast_vectors": len(selected),
        "forecast_targets_per_issue": 24,
        "executed_actual_targets_utc_inclusive": {
            "start": execution_targets[0].isoformat(),
            "end": execution_targets[-1].isoformat(),
            "count": len(execution_targets),
        },
        "expected_plan_rows": len(selected) * 2 * 24,
        "expected_execution_rows": len(selected) * 2,
        "expected_solver_calls": len(selected) * 6,
        "solver_call_estimate": "6 per issue: two full-horizon plans plus two recourse optimizations, each requiring a deviation solve and a cost solve",
        "shared_settings": {
            "assets_config": "configs/assets.yaml",
            "integration_config": "configs/integration.yaml",
            "resource_scenario": "normal",
            "resource_seed": 42,
            "demand_scale": 0.001,
            "charging_policy": "renewable_only",
            "plan_terminal_soc_policy": "preserve_initial",
            "solver": "appsi_highs",
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in files
        },
        "command": "conda run -n griddrift python scripts/run_phase45_dispatch_comparison.py --start-issue 2025-01-01T00:00Z --end-issue 2025-01-07T23:00Z --output-dir artifacts/experiments/phase45/pilot/phase45-pilot-2025w01-normal-seed42-audited",
        "dispatch_runs_performed_during_preflight": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
