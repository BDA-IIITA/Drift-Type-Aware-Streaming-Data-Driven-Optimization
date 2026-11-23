"""Provenance-preserving forecast records for later streaming error matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ForecastRecord:
    issue_timestamp: pd.Timestamp
    target_timestamp: pd.Timestamp
    horizon_hours: int
    predicted_demand_mw: float
    model_version: str

    def __post_init__(self) -> None:
        issue = pd.Timestamp(self.issue_timestamp)
        target = pd.Timestamp(self.target_timestamp)
        issue = issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
        target = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
        if self.horizon_hours < 1 or target != issue + pd.Timedelta(hours=self.horizon_hours):
            raise ValueError("target timestamp must match issue timestamp and horizon")
        if self.predicted_demand_mw < 0:
            raise ValueError("predicted demand must be non-negative")
        object.__setattr__(self, "issue_timestamp", issue)
        object.__setattr__(self, "target_timestamp", target)


class ForecastRegistry:
    def __init__(self, records: list[ForecastRecord] | None = None) -> None:
        self._records = records or []
        self._by_target: dict[pd.Timestamp, list[ForecastRecord]] = {}
        self._record_keys: set[tuple[pd.Timestamp, pd.Timestamp, str]] = set()
        for record in self._records:
            self._by_target.setdefault(record.target_timestamp, []).append(record)
            self._record_keys.add((record.issue_timestamp, record.target_timestamp, record.model_version))

    def record_horizon(
        self, issue_timestamp: pd.Timestamp, prediction: pd.Series, model_version: str
    ) -> list[ForecastRecord]:
        if prediction.empty:
            raise ValueError("prediction horizon cannot be empty")
        issue = pd.Timestamp(issue_timestamp)
        issue = issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
        created: list[ForecastRecord] = []
        for position, (target, value) in enumerate(prediction.items(), start=1):
            record = ForecastRecord(issue, pd.Timestamp(target), position, float(value), model_version)
            key = (record.issue_timestamp, record.target_timestamp, record.model_version)
            if key in self._record_keys:
                raise ValueError("duplicate forecast record")
            self._records.append(record)
            self._by_target.setdefault(record.target_timestamp, []).append(record)
            self._record_keys.add(key)
            created.append(record)
        return created

    def match_actual(self, timestamp: pd.Timestamp, actual_demand_mw: float) -> pd.DataFrame:
        target = pd.Timestamp(timestamp)
        target = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
        if actual_demand_mw < 0:
            raise ValueError("actual demand must be non-negative")
        rows = []
        for record in self._by_target.get(target, []):
            row = asdict(record)
            row["actual_demand_mw"] = float(actual_demand_mw)
            row["absolute_error_mw"] = abs(actual_demand_mw - record.predicted_demand_mw)
            rows.append(row)
        return pd.DataFrame(rows)

    def fixed_horizon_errors(
        self, timestamp: pd.Timestamp, actual_demand_mw: float, horizon_hours: int = 24
    ) -> pd.DataFrame:
        matches = self.match_actual(timestamp, actual_demand_mw)
        if matches.empty:
            # Keep the first streaming intervals uneventful: no forecast can
            # match until a 24-hour-old issue exists.
            return matches
        return matches.loc[matches["horizon_hours"] == horizon_hours].reset_index(drop=True)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self._records])

    def save_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output, index=False)
        return output
