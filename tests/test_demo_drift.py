import pandas as pd
import pytest

from scripts.create_demo_drift import create_demo_drift


def test_demo_drift_writes_separate_copy_and_only_changes_values_after_shift(tmp_path):
    source = tmp_path / "actual.csv"
    output = tmp_path / "demo_drift.csv"
    timestamps = pd.date_range("2025-01-01", periods=48, freq="1h", tz="UTC")
    original = pd.DataFrame({"timestamp": timestamps, "demand_mw": 100.0})
    original.to_csv(source, index=False)

    count, shift_start, final_timestamp = create_demo_drift(
        source, output, shift_start="2025-01-02T00:00:00Z", increase_fraction=0.15
    )

    demo = pd.read_csv(output, parse_dates=["timestamp"])
    assert count == 24
    assert shift_start == timestamps[24]
    assert final_timestamp == timestamps[-1]
    assert demo.demand_mw.iloc[:24].tolist() == [100.0] * 24
    assert demo.demand_mw.iloc[24:].tolist() == pytest.approx([115.0] * 24)
    assert pd.read_csv(source).demand_mw.tolist() == [100.0] * 48


def test_demo_drift_applies_three_permanent_shifts_without_touching_source(tmp_path):
    source = tmp_path / "actual.csv"
    output = tmp_path / "three_shifts.csv"
    timestamps = pd.date_range("2025-01-01", periods=24 * 40, freq="1h", tz="UTC")
    original = pd.DataFrame({"timestamp": timestamps, "demand_mw": 100.0})
    original.to_csv(source, index=False)

    changed, first_shift, final_timestamp = create_demo_drift(
        source,
        output,
        shift_start="2025-01-02T00:00:00Z",
        increase_fraction=0.15,
        additional_events=(
            ("2025-01-16T00:00:00Z", -0.10),
            ("2025-01-26T00:00:00Z", 0.15),
        ),
    )

    demo = pd.read_csv(output, parse_dates=["timestamp"])
    assert changed == len(demo) - 24
    assert first_shift == pd.Timestamp("2025-01-02T00:00:00Z")
    assert final_timestamp == timestamps[-1]
    # Shifts are permanent and compound from the point each event begins.
    assert demo.demand_mw.iloc[23] == 100.0
    assert demo.demand_mw.iloc[24] == pytest.approx(115.0)
    assert demo.demand_mw.iloc[15 * 24] == pytest.approx(103.5)
    assert abs(demo.demand_mw.iloc[25 * 24] - 119.025) < 1e-9
    assert pd.read_csv(source).demand_mw.eq(100.0).all()


def test_demo_drift_writes_sidecar_with_true_shift_times(tmp_path):
    import json

    from scripts.create_demo_drift import demo_metadata_path

    source = tmp_path / "actual.csv"
    output = tmp_path / "demo.csv"
    timestamps = pd.date_range("2025-01-01", periods=24 * 40, freq="1h", tz="UTC")
    pd.DataFrame({"timestamp": timestamps, "demand_mw": 100.0}).to_csv(source, index=False)

    create_demo_drift(
        source, output, shift_start="2025-01-20T00:00:00Z", increase_fraction=0.15,
        additional_events=(("2025-01-30T00:00:00Z", -0.1),),
    )

    metadata = json.loads(demo_metadata_path(output).read_text())
    assert [event["fraction"] for event in metadata["events"]] == [0.15, -0.1]
    assert pd.Timestamp(metadata["events"][0]["start"]) == pd.Timestamp("2025-01-20T00:00:00Z")
    # Default recommended start leaves two undrifted weeks for ADWIN.
    assert pd.Timestamp(metadata["recommended_replay_start"]) == pd.Timestamp("2025-01-06T00:00:00Z")
    with pytest.raises(FileExistsError):
        create_demo_drift(source, output, shift_start="2025-01-20T00:00:00Z")
