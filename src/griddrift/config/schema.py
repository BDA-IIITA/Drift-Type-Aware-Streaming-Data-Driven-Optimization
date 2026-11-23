"""Validated configuration for reproducible GridDRIFT scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class ResourceConfig:
    capacity_mw: float
    marginal_cost_eur_mwh: float

    def __post_init__(self) -> None:
        if self.capacity_mw < 0:
            raise ValueError("resource capacity_mw must be >= 0")
        if self.marginal_cost_eur_mwh < 0:
            raise ValueError("resource marginal cost must be >= 0")


@dataclass(frozen=True)
class HydroConfig(ResourceConfig):
    initial_energy_budget_mwh: float
    reservoir_capacity_mwh: float | None = None
    minimum_storage_mwh: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.initial_energy_budget_mwh < 0:
            raise ValueError("hydro initial_energy_budget_mwh must be >= 0")
        if self.reservoir_capacity_mwh is not None:
            if self.reservoir_capacity_mwh < 0:
                raise ValueError("hydro reservoir_capacity_mwh must be >= 0")
            if self.initial_energy_budget_mwh > self.reservoir_capacity_mwh:
                raise ValueError("hydro initial energy budget cannot exceed reservoir capacity")
            if self.minimum_storage_mwh > self.reservoir_capacity_mwh:
                raise ValueError("hydro minimum storage cannot exceed reservoir capacity")
        if self.minimum_storage_mwh < 0:
            raise ValueError("hydro minimum_storage_mwh must be >= 0")
        if self.initial_energy_budget_mwh < self.minimum_storage_mwh:
            raise ValueError("hydro initial energy budget cannot be below minimum storage")


@dataclass(frozen=True)
class BatteryConfig:
    energy_capacity_mwh: float
    min_soc_mwh: float
    max_soc_mwh: float
    initial_soc_mwh: float
    charge_limit_mw: float
    discharge_limit_mw: float
    charge_efficiency: float
    discharge_efficiency: float
    degradation_cost_eur_mwh: float

    def __post_init__(self) -> None:
        if self.energy_capacity_mwh <= 0:
            raise ValueError("battery energy_capacity_mwh must be > 0")
        if not 0 <= self.min_soc_mwh <= self.initial_soc_mwh <= self.max_soc_mwh:
            raise ValueError("battery SOC values must satisfy min <= initial <= max")
        if self.max_soc_mwh > self.energy_capacity_mwh:
            raise ValueError("battery max_soc_mwh cannot exceed energy_capacity_mwh")
        if self.charge_limit_mw < 0 or self.discharge_limit_mw < 0:
            raise ValueError("battery power limits must be >= 0")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("battery efficiencies must be in (0, 1]")
        if self.degradation_cost_eur_mwh < 0:
            raise ValueError("battery degradation cost must be >= 0")


@dataclass(frozen=True)
class AssetConfig:
    interval_hours: float
    solar: ResourceConfig
    wind: ResourceConfig
    hydro: HydroConfig
    conventional: ResourceConfig
    emergency_grid: ResourceConfig
    conventional_ramp_up_mw_per_interval: float
    conventional_ramp_down_mw_per_interval: float
    value_of_lost_load_eur_mwh: float
    battery: BatteryConfig

    def __post_init__(self) -> None:
        if self.interval_hours <= 0:
            raise ValueError("interval_hours must be > 0")
        if self.conventional_ramp_up_mw_per_interval < 0:
            raise ValueError("conventional ramp-up limit must be >= 0")
        if self.conventional_ramp_down_mw_per_interval < 0:
            raise ValueError("conventional ramp-down limit must be >= 0")
        if self.value_of_lost_load_eur_mwh <= self.emergency_grid.marginal_cost_eur_mwh:
            raise ValueError("VOLL must exceed emergency-grid marginal cost")


@dataclass(frozen=True)
class DataConfig:
    bidding_zone: str
    start_date: str
    end_date: str
    timezone: str
    demand_timestamp_column: str
    demand_value_column: str
    resolution: str = "1h"

    def __post_init__(self) -> None:
        if self.timezone != "UTC":
            raise ValueError("canonical GridDRIFT data timezone must be UTC")
        if not self.bidding_zone:
            raise ValueError("bidding_zone is required")
        try:
            interval = pd.Timedelta(self.resolution)
        except ValueError as exc:
            raise ValueError("data resolution must be a pandas duration such as '1h' or '30min'") from exc
        if interval <= pd.Timedelta(0):
            raise ValueError("data resolution must be positive")


@dataclass(frozen=True)
class DriftConfig:
    """Replay-only Phase 3 settings; dispatch is intentionally not configured here."""

    delta: float = 0.002
    clock: int = 32
    min_window_length: int = 5
    retraining_window_days: int = 60
    cooldown_days: int = 7
    min_error_increase_mw: float = 250.0
    min_error_increase_fraction: float = 0.10
    validation_days: int = 7
    min_validation_improvement_fraction: float = 0.02
    fine_tune_epochs: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.delta < 1:
            raise ValueError("drift delta must be between 0 and 1")
        if self.clock < 1 or self.min_window_length < 1:
            raise ValueError("drift clock and min_window_length must be >= 1")
        if self.retraining_window_days < 1 or self.cooldown_days < 0:
            raise ValueError("drift training window must be positive and cooldown non-negative")
        if self.min_error_increase_mw < 0 or self.min_error_increase_fraction < 0:
            raise ValueError("drift error-increase thresholds must be non-negative")
        if self.validation_days < 1 or self.fine_tune_epochs < 1:
            raise ValueError("validation_days and fine_tune_epochs must be >= 1")
        if not 0 <= self.min_validation_improvement_fraction <= 1:
            raise ValueError("minimum validation improvement must be between 0 and 1")


@dataclass(frozen=True)
class ProjectConfig:
    currency: str
    data: DataConfig
    assets: AssetConfig
    drift: DriftConfig = field(default_factory=DriftConfig)

    def __post_init__(self) -> None:
        if self.currency != "EUR":
            raise ValueError("Phase 1 supports EUR costs only")


def _resource(raw: dict[str, Any]) -> ResourceConfig:
    return ResourceConfig(**raw)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load and validate the explicit scenario assumptions from YAML."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_project_config(raw)


def parse_project_config(raw: dict[str, Any]) -> ProjectConfig:
    """Validate a decoded project configuration mapping."""
    if not isinstance(raw, dict):
        raise TypeError("project configuration must be a YAML mapping")

    assets = raw["assets"]
    battery = BatteryConfig(**assets["battery"])
    asset_config = AssetConfig(
        interval_hours=assets["interval_hours"],
        solar=_resource(assets["solar"]),
        wind=_resource(assets["wind"]),
        hydro=HydroConfig(**assets["hydro"]),
        conventional=_resource(assets["conventional"]),
        emergency_grid=_resource(assets["emergency_grid"]),
        conventional_ramp_up_mw_per_interval=assets[
            "conventional_ramp_up_mw_per_interval"
        ],
        conventional_ramp_down_mw_per_interval=assets[
            "conventional_ramp_down_mw_per_interval"
        ],
        value_of_lost_load_eur_mwh=assets["value_of_lost_load_eur_mwh"],
        battery=battery,
    )
    return ProjectConfig(
        currency=raw["currency"],
        data=DataConfig(**raw["data"]),
        assets=asset_config,
        drift=DriftConfig(**raw.get("drift", {})),
    )
