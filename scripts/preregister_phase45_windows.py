"""Preregister disjoint Phase 4.5 Stage 4 windows and verify inputs only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from griddrift.simulation.phase45_dispatch import _utc_series
from griddrift.simulation.phase45_inputs import load_phase45_forecast_vectors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/phase45/runs/stage4_window_preregistration.json"
ALIGNED = ROOT / "artifacts/experiments/phase38/comparison/common-period-20261024/aligned_predictions.csv"
MLP_REGISTRY = ROOT / "artifacts/experiments/phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct/adaptive_forecast_registry.csv"
XGB_REGISTRY = ROOT / "artifacts/experiments/phase37/final/final-holdout-20261024/full_history/adaptive_forecast_registry.csv"
CONFIGS = [ROOT / "configs/assets.yaml", ROOT / "configs/integration.yaml"]
CODE = [
    ROOT / "src/griddrift/dispatch/engine.py",
    ROOT / "src/griddrift/simulation/phase45_dispatch.py",
    ROOT / "scripts/run_phase45_dispatch_comparison.py",
    ROOT / "scripts/validate_phase45_pilot.py",
]
WINDOWS = [
    ("spring-2025w14", pd.Timestamp("2025-04-01T00:00Z"), pd.Timestamp("2025-04-07T23:00Z")),
    ("summer-2025w27", pd.Timestamp("2025-07-01T00:00Z"), pd.Timestamp("2025-07-07T23:00Z")),
    ("autumn-2025w40", pd.Timestamp("2025-10-01T00:00Z"), pd.Timestamp("2025-10-07T23:00Z")),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite Stage 4 preregistration: {OUT}")
    aligned = pd.read_csv(ALIGNED)
    actual_frame = aligned[["target_timestamp", "actual_demand_mw"]].copy()
    actual_frame["target_timestamp"] = pd.to_datetime(actual_frame.target_timestamp, utc=True, errors="raise")
    actual_frame["actual_demand_mw"] = pd.to_numeric(actual_frame.actual_demand_mw, errors="raise")
    counts = actual_frame.groupby("target_timestamp").actual_demand_mw.nunique(dropna=False)
    if (counts != 1).any():
        raise ValueError("actual demand is inconsistent across duplicate target timestamps")
    actuals = actual_frame.drop_duplicates("target_timestamp").set_index("target_timestamp").actual_demand_mw
    actuals = _utc_series(actuals)
    forecast_set = load_phase45_forecast_vectors(
        aligned, pd.read_csv(MLP_REGISTRY), pd.read_csv(XGB_REGISTRY)
    )
    issues = pd.DatetimeIndex([vector.issue_timestamp for vector in forecast_set.vectors])
    result_windows = []
    for name, start, end in WINDOWS:
        expected = pd.date_range(start, end, freq="h")
        selected = [vector for vector in forecast_set.vectors if start <= vector.issue_timestamp <= end]
        observed = pd.DatetimeIndex([vector.issue_timestamp for vector in selected])
        if not observed.equals(expected):
            raise ValueError(f"{name}: expected {len(expected)} complete contiguous vectors, found {len(selected)}")
        targets = pd.DatetimeIndex([vector.target_timestamps[0] for vector in selected])
        missing = targets.difference(actuals.index)
        if not missing.empty:
            raise ValueError(f"{name}: missing execution actuals {missing.tolist()[:3]}")
        run_id = f"phase45-{name}-normal-seed42"
        run_directory = ROOT / "artifacts/experiments/phase45/runs" / run_id
        if run_directory.exists():
            raise FileExistsError(f"refusing to overwrite a Stage 4 output: {run_directory}")
        result_windows.append({
            "window_id": name,
            "issue_start_utc": start.isoformat(),
            "issue_end_utc": end.isoformat(),
            "issue_count": len(selected),
            "executed_target_start_utc": targets[0].isoformat(),
            "executed_target_end_utc": targets[-1].isoformat(),
            "complete_forecast_vectors": len(selected),
            "output_directory": str(run_directory.relative_to(ROOT)),
            "runner_command": (
                "conda run -n griddrift python scripts/run_phase45_dispatch_comparison.py "
                f"--start-issue {start.strftime('%Y-%m-%dT%H:%MZ')} "
                f"--end-issue {end.strftime('%Y-%m-%dT%H:%MZ')} "
                f"--output-dir artifacts/experiments/phase45/runs/{run_id}"
            ),
        })
    if len(issues) != forecast_set.aligned_prediction_keys // 24 and not forecast_set.partial_issue_count:
        # The complete vectors are expected to include every exact 24-row group.
        raise ValueError("forecast issue count did not reconcile with aligned prediction keys")

    files = [ALIGNED, MLP_REGISTRY, XGB_REGISTRY, *CONFIGS, *CODE]
    result = {
        "phase": "4.5",
        "stage": "4_larger_historical_comparison",
        "status": "preregistered_preflight_passed_not_run",
        "selection_rule": "Fixed first seven calendar days of April, July, and October 2025 to span seasonal conditions; windows do not overlap the January development pilot. Dates were chosen before comparing dispatch results.",
        "per_window_initial_state": {
            "battery_soc_mwh": 15.0,
            "hydro_remaining_mwh": 120.0,
            "conventional_output_mw": 0.0,
        },
        "state_policy": "Reset to the same initial state at each disjoint window and carry state continuously within a window. Record terminal battery/hydro/generator state. Do not claim a cost winner if end states differ; no terminal energy price is invented.",
        "common_settings": {
            "assets_config": "configs/assets.yaml",
            "integration_config": "configs/integration.yaml",
            "resource_scenario": "normal",
            "resource_seed": 42,
            "demand_scale": 0.001,
            "battery_charging_policy": "renewable_only",
            "plan_terminal_soc_policy": "preserve_initial",
            "solver": "appsi_highs",
        },
        "windows": result_windows,
        "expected_total_issue_hours": sum(item["issue_count"] for item in result_windows),
        "expected_plan_rows": sum(item["issue_count"] for item in result_windows) * 2 * 24,
        "expected_execution_rows": sum(item["issue_count"] for item in result_windows) * 2,
        "expected_solver_calls": sum(item["issue_count"] for item in result_windows) * 6,
        "pilot_observed_wall_seconds": 35.0,
        "estimated_stage4_wall_seconds": 3 * 35.0,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "dispatch_runs_performed_during_preregistration": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
