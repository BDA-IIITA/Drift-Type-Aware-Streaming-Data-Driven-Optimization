"""Export Phase 3.6 holdout MLP candidate rejections for Phase 3.9 Stage 1.

Read-only analysis of an existing event CSV. Does not load or alter models,
train candidates, forecast, or run a replay.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "artifacts/experiments/phase36/final/final-holdout-rerun-20261023/"
    "strategies/gentle_mixed_60d_threshold_05pct/evaluation_adaptation_events.csv"
)
OUTPUT = ROOT / "artifacts/experiments/phase39"
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05)


def main() -> None:
    csv_path = OUTPUT / "stage1_candidate_rejections.csv"
    report_path = OUTPUT / "stage1_diagnosis.md"
    if csv_path.exists() or report_path.exists():
        raise SystemExit("Stage 1 outputs already exist; refusing to overwrite them.")
    if not SOURCE.exists():
        raise SystemExit(f"Saved Phase 3.6 event log not found: {SOURCE}")

    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        events = list(csv.DictReader(handle))
    candidates = [row for row in events if row.get("outcome") == "candidate_rejected"]
    if len(candidates) != 33:
        raise SystemExit(f"Expected 33 rejected holdout candidates, found {len(candidates)}")

    out_fields = [
        "drift_timestamp",
        "current_model_version",
        "incumbent_validation_mae_mw",
        "candidate_validation_mae_mw",
        "validation_improvement_fraction",
        "validation_improvement_percent",
        "candidate_validation_direction",
        "configured_threshold_percent",
        "rejection_reason",
        "would_pass_0pct",
        "would_pass_0_5pct",
        "would_pass_1pct",
        "would_pass_2pct",
        "would_pass_5pct",
        "validation_first_target_timestamp",
        "validation_last_target_timestamp",
    ]
    rows: list[dict[str, str]] = []
    improvements: list[float] = []
    for event in candidates:
        improvement = float(event["validation_improvement_fraction"])
        incumbent = float(event["incumbent_validation_mae_mw"])
        candidate = float(event["candidate_validation_mae_mw"])
        improvements.append(improvement)
        if improvement > 0:
            direction = "candidate_better"
        elif improvement < 0:
            direction = "candidate_worse"
        else:
            direction = "equal_validation_mae"
        rows.append(
            {
                "drift_timestamp": event["drift_timestamp"],
                "current_model_version": event["previous_model_version"],
                "incumbent_validation_mae_mw": f"{incumbent:.12g}",
                "candidate_validation_mae_mw": f"{candidate:.12g}",
                "validation_improvement_fraction": f"{improvement:.12g}",
                "validation_improvement_percent": f"{100 * improvement:.6f}",
                "candidate_validation_direction": direction,
                "configured_threshold_percent": "5.000000",
                "rejection_reason": (
                    f"Candidate validation improvement ({100 * improvement:.4f}%) was below the configured 5.0000% minimum."
                ),
                "would_pass_0pct": str(improvement >= 0).lower(),
                "would_pass_0_5pct": str(improvement >= 0.005).lower(),
                "would_pass_1pct": str(improvement >= 0.01).lower(),
                "would_pass_2pct": str(improvement >= 0.02).lower(),
                "would_pass_5pct": str(improvement >= 0.05).lower(),
                "validation_first_target_timestamp": event["validation_first_target_timestamp"],
                "validation_last_target_timestamp": event["validation_last_target_timestamp"],
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        threshold: sum(value >= threshold for value in improvements)
        for threshold in THRESHOLDS
    }
    positive = sum(value > 0 for value in improvements)
    worse = sum(value < 0 for value in improvements)
    equal = sum(value == 0 for value in improvements)
    report = f"""# Phase 3.9 Stage 1 — Why the MLP holdout candidates were rejected

## Scope and source

Read-only analysis of the 33 `candidate_rejected` events in the saved Phase 3.6 evaluation log. No model was loaded or changed; no training, prediction, or replay was run. The file also contains alarms that did not meet the error-increase trigger and cooldown skips; these were not candidate attempts and are excluded from the 33-candidate analysis.

Source: `{SOURCE.relative_to(ROOT).as_posix()}`
Extract: `stage1_candidate_rejections.csv`

## Findings

- The selected configuration required at least **5% validation MAE improvement**. The implementation computes `(incumbent MAE - candidate MAE) / incumbent MAE` and accepts only when that fraction is greater than or equal to the configured threshold.
- **{positive} of 33** candidates had positive validation improvement, but all were below 5%; **{worse} of 33** had worse validation MAE than the incumbent; **{equal}** were exactly tied.
- Hypothetical one-step counts on these already-recorded candidates: **0%: {counts[0.0]}/33; 0.5%: {counts[0.005]}/33; 1%: {counts[0.01]}/33; 2%: {counts[0.02]}/33; 5%: {counts[0.05]}/33.**
- The largest observed candidate improvement was **{max(improvements) * 100:.2f}%**; the largest degradation was **{min(improvements) * 100:.2f}%**.

All 33 were rejected because their measured validation improvement did not reach the configured 5% gate. A positive but sub-threshold candidate is not necessarily a useful update on later unseen data; validation only records candidate-vs-incumbent performance on the held-back validation segment.

## Important interpretation

The hypothetical counts do **not** estimate how many updates a full replay at each threshold would accept. If an earlier candidate were accepted, the deployed model would change, which changes later candidate initialization, validation comparisons, and potentially the replay path. These counts are a diagnostic of the actual 5%-threshold candidate set only. The 2025–2026 period was examined in earlier phases and must be treated as exploratory for any new threshold analysis.

## Reproduction

```powershell
conda activate griddrift
python scripts/analyze_phase39_mlp_rejections.py
```
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Analyzed {len(candidates)} saved rejected candidates.")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    print("Hypothetical passes:", ", ".join(f"{100*t:g}%={counts[t]}" for t in THRESHOLDS))


if __name__ == "__main__":
    main()
