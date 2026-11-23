"""Generate the explicitly simulated Phase 1 smoke scenario as a CSV file."""

from __future__ import annotations

from pathlib import Path

from griddrift.data.synthetic import make_synthetic_scenario


def main() -> None:
    frame, metadata = make_synthetic_scenario(periods=24 * 14, seed=42)
    output = Path("data/processed/synthetic_scenario.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=True)
    print(f"Wrote {len(frame)} simulated hourly rows to {output}")
    print(f"Renewable source: {metadata.renewable_source}")


if __name__ == "__main__":
    main()
