"""Run a bounded Phase 4.5 paired dispatch window from saved forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from griddrift.dispatch import load_dispatch_config
from griddrift.simulation.integrated import load_integrated_settings
from griddrift.simulation.phase45_dispatch import run_phase45_dispatch_comparison
from griddrift.simulation.phase45_inputs import Phase45InputSettings, load_phase45_forecast_vectors

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "artifacts/experiments/phase38/comparison/common-period-20261024/aligned_predictions.csv"
MLP_REGISTRY = ROOT / "artifacts/experiments/phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct/adaptive_forecast_registry.csv"
XGB_REGISTRY = ROOT / "artifacts/experiments/phase37/final/final-holdout-20261024/full_history/adaptive_forecast_registry.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actual_series(aligned: pd.DataFrame) -> pd.Series:
    required = {"target_timestamp", "actual_demand_mw"}
    if not required.issubset(aligned.columns):
        raise ValueError(f"aligned predictions must include {sorted(required)}")
    frame = aligned[["target_timestamp", "actual_demand_mw"]].copy()
    frame["target_timestamp"] = pd.to_datetime(frame.target_timestamp, utc=True, errors="raise")
    frame["actual_demand_mw"] = pd.to_numeric(frame.actual_demand_mw, errors="raise")
    counts = frame.groupby("target_timestamp").actual_demand_mw.nunique(dropna=False)
    if (counts != 1).any():
        raise ValueError("aligned actual demand disagrees across forecast horizons")
    actuals = frame.drop_duplicates("target_timestamp").set_index("target_timestamp").actual_demand_mw
    actuals.name = "actual_demand_mw"
    return actuals.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-issue", required=True, help="inclusive UTC issue timestamp, e.g. 2025-01-01T00:00Z")
    parser.add_argument("--end-issue", required=True, help="inclusive UTC issue timestamp, e.g. 2025-01-07T23:00Z")
    parser.add_argument("--output-dir", required=True, help="new, non-existing output directory under phase45")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    allowed_output_root = (ROOT / "artifacts/experiments/phase45").resolve()
    if not output.resolve().is_relative_to(allowed_output_root):
        raise ValueError(f"output directory must be within {allowed_output_root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Phase 4.5 run: {output}")

    aligned = pd.read_csv(ALIGNED)
    vectors = load_phase45_forecast_vectors(
        aligned,
        pd.read_csv(MLP_REGISTRY),
        pd.read_csv(XGB_REGISTRY),
    )
    actuals = _actual_series(aligned)
    assets, dispatch_settings = load_dispatch_config(ROOT / "configs/assets.yaml")
    integration = load_integrated_settings(str(ROOT / "configs/integration.yaml"))
    settings = Phase45InputSettings(
        demand_scale=integration["demand_scale"],
        resource_seed=integration["resource_seed"],
        resource_scenario=integration["resource_scenario"],
        conventional_min_output_mw=dispatch_settings["conventional_min_output_mw"],
        battery_charging_policy=integration["battery_charging_policy"],
        terminal_soc_policy=integration["terminal_soc_policy"],
        terminal_soc_target_mwh=dispatch_settings["terminal_soc_target_mwh"],
        solver_name="appsi_highs",
    )
    output_result = run_phase45_dispatch_comparison(
        vectors.vectors,
        actuals,
        assets,
        settings=settings,
        start_issue=pd.Timestamp(args.start_issue),
        end_issue=pd.Timestamp(args.end_issue),
    )
    paths = output_result.save(output, run_metadata={
        "start_issue": args.start_issue,
        "end_issue": args.end_issue,
        "forecast_vectors_available": len(vectors.vectors),
        "phase38_aligned_sha256": _sha256(ALIGNED),
        "phase36_mlp_registry_sha256": _sha256(MLP_REGISTRY),
        "phase37_xgboost_registry_sha256": _sha256(XGB_REGISTRY),
        "actual_demand_source": str(ALIGNED.relative_to(ROOT)),
        "actual_demand_is_used_only_after_both_issue_plans": True,
    })
    print(json.dumps({"output_files": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
