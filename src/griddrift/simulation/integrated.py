"""Leakage-safe hourly integration of demand forecasts and rolling dispatch."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import yaml

from griddrift.config.schema import AssetConfig, HydroConfig
from griddrift.data.hydro import ReservoirParameters
from griddrift.data.renewables import RenewableResourceProvider
from griddrift.dispatch import DispatchState, optimize_dispatch
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.forecasting.xgboost import DirectXGBoostForecaster
from griddrift.simulation.replay import ReplayConfig, StreamingReplay
from griddrift.simulation.xgboost_replay import XGBoostReplayConfig, XGBoostStreamingReplay


@dataclass(frozen=True)
class IntegratedConfig:
    demand_scale: float = 0.001
    solar_cost_blocks_eur_mwh: tuple[float, ...] = (8, 9, 12, 14, 11, 8)
    wind_cost_blocks_eur_mwh: tuple[float, ...] = (12, 11, 10, 9, 11, 13)
    resource_seed: int = 42
    resource_scenario: str = "normal"
    battery_charging_policy: str = "renewable_only"
    allow_hydro_battery_charging: bool = False
    policy: str = "static"
    adaptation: ReplayConfig = field(
        default_factory=lambda: ReplayConfig(adaptation_policy="validated")
    )
    xgboost_adaptation: XGBoostReplayConfig = field(
        default_factory=lambda: XGBoostReplayConfig(
            strategy="full_history", min_validation_improvement_fraction=0.005
        )
    )
    terminal_soc_policy: str = "preserve_initial"
    renewable_forecast_issue_policy: str = RenewableResourceProvider.ISSUE_POLICY_PREVIOUS_DAY_00_UTC

    def __post_init__(self) -> None:
        for rates in (self.solar_cost_blocks_eur_mwh, self.wind_cost_blocks_eur_mwh):
            if len(rates) != 6 or any(not np.isfinite(rate) or rate < 0 for rate in rates):
                raise ValueError("renewable cost blocks must contain six finite nonnegative rates")
        if self.demand_scale <= 0:
            raise ValueError("demand_scale must be positive")
        if self.policy not in {"static", "adaptive"}:
            raise ValueError("policy must be static or adaptive")
        if self.resource_scenario not in {"normal", "low_renewables", "wind_scarcity"}:
            raise ValueError("unsupported resource_scenario")
        if self.battery_charging_policy not in {"renewable_only", "unrestricted"}:
            raise ValueError("unsupported battery_charging_policy")
        if self.allow_hydro_battery_charging and self.battery_charging_policy != "renewable_only":
            raise ValueError("allow_hydro_battery_charging requires renewable_only charging")
        if self.renewable_forecast_issue_policy != RenewableResourceProvider.ISSUE_POLICY_PREVIOUS_DAY_00_UTC:
            raise ValueError("unsupported renewable forecast issue-time policy")


def load_integrated_settings(path: str) -> dict:
    """Load the small integration YAML, keeping project defaults configurable."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    settings = {
        "solar_cost_blocks_eur_mwh": tuple(map(float, raw.get("solar_cost_blocks_eur_mwh", (8, 9, 12, 14, 11, 8)))),
        "wind_cost_blocks_eur_mwh": tuple(map(float, raw.get("wind_cost_blocks_eur_mwh", (12, 11, 10, 9, 11, 13)))),
        "demand_scale": float(raw.get("representative_grid_scale", 0.001)),
        "renewable_data_mode": str(raw.get("renewable_data_mode", "simulated")),
        "resource_seed": int(raw.get("resource_seed", 42)),
        "resource_scenario": str(raw.get("resource_scenario", "normal")),
        "policy": str(raw.get("forecasting_policy", "static")),
        "battery_charging_policy": str(raw.get("battery_charging_policy", "renewable_only")),
        "allow_hydro_battery_charging": bool(raw.get("allow_hydro_battery_charging", True)),
        "terminal_soc_policy": str(raw.get("terminal_soc_policy", "preserve_initial")),
        "renewable_forecast_issue_policy": str(
            raw.get(
                "renewable_forecast_issue_policy",
                RenewableResourceProvider.ISSUE_POLICY_PREVIOUS_DAY_00_UTC,
            )
        ),
        "replay_speed_seconds_per_hour": float(raw.get("replay_speed_seconds_per_hour", 2.0)),
    }
    if settings["renewable_data_mode"] not in {"simulated", "entsoe_pecd"}:
        raise ValueError("renewable_data_mode must be 'simulated' or 'entsoe_pecd'")
    IntegratedConfig(**{
        key: value for key, value in settings.items()
        if key not in {"replay_speed_seconds_per_hour", "renewable_data_mode"}
    })
    return settings


def simulated_resource_actuals(
    index: pd.DatetimeIndex, assets: AssetConfig, seed: int, scenario: str = "normal"
) -> pd.DataFrame:
    """Create hidden, reproducible resource outcomes, separate from forecasts."""
    rng = np.random.default_rng(seed)
    hour, day = index.hour.to_numpy(), index.dayofyear.to_numpy()
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    seasonal = 0.68 + 0.32 * np.sin(2 * np.pi * (day - 80) / 365)
    solar_factor, wind_factor = {"normal": (1.0, 1.0), "low_renewables": (0.55, 0.55), "wind_scarcity": (1.0, 0.30)}[scenario]
    solar = np.clip(assets.solar.capacity_mw * solar_factor * daylight * seasonal + rng.normal(0, assets.solar.capacity_mw * solar_factor * 0.035, len(index)), 0, assets.solar.capacity_mw * solar_factor)
    wind_shape = 0.50 + 0.12 * np.sin(2 * np.pi * (hour + 5) / 24) + 0.10 * np.sin(2 * np.pi * day / 365)
    wind = np.clip(assets.wind.capacity_mw * wind_factor * (wind_shape + rng.normal(0, 0.09, len(index))), 0, assets.wind.capacity_mw * wind_factor)
    return pd.DataFrame({"solar_actual_mw": solar, "wind_actual_mw": wind}, index=index)


def issue_resource_forecast(
    targets: pd.DatetimeIndex, assets: AssetConfig, scenario: str = "normal"
) -> dict[str, np.ndarray]:
    """Issue calendar-pattern forecasts without reading any future actuals."""
    hour, day = targets.hour.to_numpy(), targets.dayofyear.to_numpy()
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    solar_factor, wind_factor = {"normal": (1.0, 1.0), "low_renewables": (0.55, 0.55), "wind_scarcity": (1.0, 0.30)}[scenario]
    solar = assets.solar.capacity_mw * solar_factor * daylight * (0.68 + 0.32 * np.sin(2 * np.pi * (day - 80) / 365))
    wind_shape = 0.50 + 0.12 * np.sin(2 * np.pi * (hour + 5) / 24) + 0.10 * np.sin(2 * np.pi * day / 365)
    wind = assets.wind.capacity_mw * wind_factor * wind_shape
    return {"solar": np.clip(solar, 0, assets.solar.capacity_mw), "wind": np.clip(wind, 0, assets.wind.capacity_mw)}


class IntegratedSimulation:
    """One-hour-at-a-time replay; each issue immediately creates a full plan."""

    def __init__(
        self,
        demand: pd.Series,
        model: DirectMLPForecaster | DirectXGBoostForecaster,
        assets: AssetConfig,
        *,
        config: IntegratedConfig | None = None,
        detector=None,
        start_timestamp: pd.Timestamp | None = None,
        model_family: str | None = None,
        renewable_provider: RenewableResourceProvider | None = None,
        hydro_parameters: ReservoirParameters | None = None,
        hydro_inflow_profile: pd.Series | None = None,
    ) -> None:
        values = pd.to_numeric(demand, errors="coerce").astype(float)
        index = pd.DatetimeIndex(demand.index)
        if index.tz is None:
            raise ValueError("demand index must be timezone-aware UTC")
        index = index.tz_convert("UTC")
        if not index.is_monotonic_increasing or index.has_duplicates or values.isna().any() or (values < 0).any():
            raise ValueError("demand must be finite, non-negative, chronological, and unique")
        if len(index) < 2 or not (index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
            raise ValueError("integrated replay requires continuous hourly demand")
        self.demand = pd.Series(values.to_numpy(), index=index, name="demand_mw")
        self.static_model = model
        inferred_family = "xgboost" if isinstance(model, DirectXGBoostForecaster) else "mlp"
        self.model_family = model_family or inferred_family
        if self.model_family not in {"mlp", "xgboost"} or self.model_family != inferred_family:
            raise ValueError("model_family must match the supplied forecaster")
        self.config = config or IntegratedConfig()
        self.renewable_provider = renewable_provider
        self.hydro_inflow_profile = hydro_inflow_profile
        self.resource_data_mode = "entsoe_pecd" if renewable_provider is not None else "simulated"
        if renewable_provider is not None:
            if hydro_parameters is None or hydro_inflow_profile is None:
                raise ValueError("historical renewable replay requires PECD reservoir parameters and hourly inflow")
            if not isinstance(hydro_parameters, ReservoirParameters):
                raise TypeError("hydro_parameters must be a ReservoirParameters value")
            scale = self.config.demand_scale
            scaled_hydro = HydroConfig(
                capacity_mw=hydro_parameters.turbine_capacity_mw * scale,
                marginal_cost_eur_mwh=assets.hydro.marginal_cost_eur_mwh,
                initial_energy_budget_mwh=hydro_parameters.initial_reservoir_energy_mwh * scale,
                reservoir_capacity_mwh=hydro_parameters.reservoir_capacity_mwh * scale,
                minimum_storage_mwh=hydro_parameters.minimum_reservoir_energy_mwh * scale,
            )
            self.assets = replace(assets, hydro=scaled_hydro)
            inflow_index = pd.DatetimeIndex(hydro_inflow_profile.index)
            inflow_values = pd.to_numeric(hydro_inflow_profile, errors="coerce").astype(float)
            if (
                inflow_index.tz is None
                or not inflow_index.is_monotonic_increasing
                or inflow_index.has_duplicates
                or inflow_values.isna().any()
                or not np.isfinite(inflow_values.to_numpy()).all()
                or (inflow_values < 0).any()
            ):
                raise ValueError("hydro inflow profile must be finite, nonnegative, unique, and UTC-indexed")
            inflow_index = inflow_index.tz_convert("UTC")
            self.hydro_inflow_profile = pd.Series(
                inflow_values.to_numpy(), index=inflow_index, name="assumed_inflow_mwh_per_hour"
            )
        else:
            if hydro_parameters is not None or hydro_inflow_profile is not None:
                raise ValueError("PECD hydro inputs require a renewable_provider")
            self.assets = assets
        self._detector_template = copy.deepcopy(detector)
        self._resource_actuals = (
            None
            if renewable_provider is not None
            else simulated_resource_actuals(
                index, self.assets, self.config.resource_seed, self.config.resource_scenario
            )
        )
        self.reset(start_timestamp=start_timestamp)

    @property
    def minimum_replay_position(self) -> int:
        """First position after the frozen model's training targets are observable."""
        lookback_position = self.static_model.feature_spec.lookback_hours - 1
        after_training = int(self.demand.index.searchsorted(
            self.static_model.train_end_issue_time + pd.Timedelta(hours=24), side="left"
        ))
        return max(lookback_position, after_training)

    def reset(self, start_timestamp: pd.Timestamp | None = None) -> None:
        """Reset the replay, optionally starting at a requested observed UTC time.

        The MLP and replay detector are reset as before. A short, leakage-safe
        forecast warm-up is registered so each newly revealed hour can be scored
        against a forecast issued 24 hours earlier from then-available history.
        """
        if self.model_family == "mlp":
            self.replay = StreamingReplay(self.demand, self.static_model, config=self.config.adaptation,
                                          detector=copy.deepcopy(self._detector_template))
            self.adaptive_registry = self.replay.adaptive_registry
            self.static_registry = self.replay.static_registry
            self.detector = self.replay.detector
            self.adaptation_events = self.replay.events
        else:
            self.replay = XGBoostStreamingReplay(
                self.static_model,
                config=self.config.xgboost_adaptation,
                detector=copy.deepcopy(self._detector_template),
            )
            self.adaptive_registry = self.replay.adaptive_registry
            self.static_registry = self.replay.static_registry
            self.detector = self.replay.detector
            self.adaptation_events = self.replay.events
        self.current_position = self.minimum_replay_position
        if start_timestamp is not None:
            requested = pd.Timestamp(start_timestamp)
            requested = requested.tz_localize("UTC") if requested.tz is None else requested.tz_convert("UTC")
            self.current_position = max(
                self.current_position,
                int(self.demand.index.searchsorted(requested, side="left")),
            )
        if self.current_position >= len(self.demand):
            raise ValueError("requested replay start is after the last available actual observation")
        self.state = DispatchState(
            self.assets.battery.initial_soc_mwh,
            self.assets.hydro.initial_energy_budget_mwh,
            0.0,
        )
        self.actual_history = self.demand.iloc[: self.current_position + 1].copy()
        self.errors: list[dict] = []
        self.drift_events: list[dict] = []
        self.realized: list[dict] = []
        self.accumulated_realized_cost_eur = 0.0
        self.last_drift = False
        self._seed_fixed_horizon_forecasts()
        self._issue_plan()

    def _seed_fixed_horizon_forecasts(self) -> None:
        """Register recent past issues without exposing their future targets.

        Starting 24 hours after the latest training issue means the first
        eligible backfill issue is strictly later than the training cutoff.
        Each prediction receives history only through its own issue timestamp.
        """
        first_eligible_issue = int(self.demand.index.searchsorted(
            self.static_model.train_end_issue_time, side="right"
        ))
        first_issue_position = max(
            self.static_model.feature_spec.lookback_hours - 1,
            first_eligible_issue,
            self.current_position - 23,
        )
        for position in range(first_issue_position, self.current_position):
            issue = self.demand.index[position]
            if issue <= self.static_model.train_end_issue_time:
                continue
            history = self.demand.iloc[: position + 1]
            static_forecast = self._predict_horizon(self.static_model, history, issue)
            adaptive_forecast = self._predict_horizon(self.replay.adaptive_model, history, issue)
            self.static_registry.record_horizon(issue, static_forecast, self.static_model.model_version)
            self.adaptive_registry.record_horizon(
                issue, adaptive_forecast, self.replay.adaptive_model.model_version
            )

    @property
    def timestamp(self) -> pd.Timestamp:
        return self.demand.index[self.current_position]

    @property
    def status(self) -> str:
        return "complete" if self.current_position >= len(self.demand) - 1 else "paused"

    @property
    def active_model(self) -> DirectMLPForecaster | DirectXGBoostForecaster:
        return self.replay.adaptive_model if self.config.policy == "adaptive" else self.static_model

    def _predict_horizon(self, model, history: pd.Series, issue: pd.Timestamp) -> pd.Series:
        """Require an exact, ordered UTC +1h..+24h prediction horizon."""
        forecast = model.predict_horizon(history, issue)
        expected = pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="1h")
        if not isinstance(forecast, pd.Series) or not forecast.index.equals(expected):
            raise ValueError("forecaster must return exactly 24 UTC targets from issue time +1h through +24h")
        values = pd.to_numeric(forecast, errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all() or (values < 0).any():
            raise ValueError("forecast values must be finite and non-negative")
        return values.astype(float).rename("predicted_demand_mw")

    def _issue_plan(self) -> None:
        issue = self.timestamp
        history = self.actual_history.loc[:issue]
        self.static_forecast = self._predict_horizon(self.static_model, history, issue)
        self.adaptive_forecast = self._predict_horizon(self.replay.adaptive_model, history, issue)
        self.static_registry.record_horizon(issue, self.static_forecast, self.static_model.model_version)
        self.adaptive_registry.record_horizon(issue, self.adaptive_forecast, self.replay.adaptive_model.model_version)
        forecast = self.static_forecast if self.config.policy == "static" else self.adaptive_forecast
        self.forecast = forecast
        if self.renewable_provider is None:
            self.resource_forecast = issue_resource_forecast(
                pd.DatetimeIndex(forecast.index), self.assets, self.config.resource_scenario
            )
            self.resource_forecast_details = pd.DataFrame(index=forecast.index)
            self.resource_forecast_details["issue_time_policy"] = "simulated calendar profile"
        else:
            self.resource_forecast_details = self.renewable_provider.forecast_at(
                issue,
                pd.DatetimeIndex(forecast.index),
                scale=self.config.demand_scale,
            )
            self.resource_forecast = {
                "solar": self.resource_forecast_details.solar_available_mw.to_numpy(dtype=float),
                "wind": self.resource_forecast_details.wind_available_mw.to_numpy(dtype=float),
            }
        self.hydro_inflow_forecast_mw = self._hydro_inflow_rate(pd.DatetimeIndex(forecast.index))
        dispatch_assets = self._assets_for_resource_availability(self.resource_forecast)
        self.plan = optimize_dispatch(
            forecast.to_numpy() * self.config.demand_scale,
            self.resource_forecast,
            dispatch_assets,
            self.state,
            timestamps=forecast.index,
            hydro_inflow_mw=self.hydro_inflow_forecast_mw,
            **self._renewable_costs(forecast.index),
            terminal_soc_policy=self.config.terminal_soc_policy,
            battery_charging_policy=self.config.battery_charging_policy,
            allow_hydro_battery_charging=self.config.allow_hydro_battery_charging,
        )
        self.planned_cost_eur = self.plan.total_cost_eur

    def set_hydro_battery_charging(self, enabled: bool) -> bool:
        """Apply the hydro charging option and re-optimize the current plan.

        This does not advance replay time, reveal an observation, update ADWIN,
        or append duplicate forecast-registry entries.
        """
        enabled = bool(enabled)
        if enabled == self.config.allow_hydro_battery_charging:
            return False
        updated_config = replace(self.config, allow_hydro_battery_charging=enabled)
        forecast = self.forecast
        dispatch_assets = self._assets_for_resource_availability(self.resource_forecast)
        updated_plan = optimize_dispatch(
            forecast.to_numpy() * updated_config.demand_scale,
            self.resource_forecast,
            dispatch_assets,
            self.state,
            timestamps=forecast.index,
            hydro_inflow_mw=self.hydro_inflow_forecast_mw,
            **self._renewable_costs(forecast.index),
            terminal_soc_policy=updated_config.terminal_soc_policy,
            battery_charging_policy=updated_config.battery_charging_policy,
            allow_hydro_battery_charging=updated_config.allow_hydro_battery_charging,
        )
        self.config = updated_config
        self.plan = updated_plan
        self.planned_cost_eur = updated_plan.total_cost_eur
        return True

    def _renewable_costs(self, timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        blocks = pd.DatetimeIndex(timestamps).tz_convert("UTC").hour.to_numpy() // 4
        return {
            "solar_costs_eur_mwh": np.asarray(self.config.solar_cost_blocks_eur_mwh, dtype=float)[blocks],
            "wind_costs_eur_mwh": np.asarray(self.config.wind_cost_blocks_eur_mwh, dtype=float)[blocks],
        }

    def _hydro_inflow_rate(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        if self.hydro_inflow_profile is None:
            return np.zeros(len(timestamps), dtype=float)
        targets = pd.DatetimeIndex(timestamps)
        missing = targets.difference(self.hydro_inflow_profile.index)
        if len(missing):
            raise ValueError(f"hydro inflow scenario does not cover target hour {missing[0]}")
        # Input is MWh per hour in national units; optimizer input is MW.
        interval = self.assets.interval_hours
        energy_mwh = self.hydro_inflow_profile.loc[targets].to_numpy(dtype=float) * self.config.demand_scale
        return energy_mwh / interval

    def _assets_for_resource_availability(self, resources: dict[str, np.ndarray]) -> AssetConfig:
        """Use this interval's availability as its renewable envelope.

        The source provides production forecasts/actuals but no installed
        solar/wind capacity. The optimizer already bounds renewable dispatch by
        availability. Expanding this validation-only envelope to the supplied
        horizon prevents unrelated illustrative capacities from clipping it.
        """
        if self.renewable_provider is None:
            return self.assets
        solar_cap = max(self.assets.solar.capacity_mw, float(np.max(resources["solar"])))
        wind_cap = max(self.assets.wind.capacity_mw, float(np.max(resources["wind"])))
        return replace(
            self.assets,
            solar=replace(self.assets.solar, capacity_mw=solar_cap),
            wind=replace(self.assets.wind, capacity_mw=wind_cap),
        )

    def step(self) -> dict:
        """Reveal the next hour, score old forecasts, realize one interval, replan."""
        if self.current_position >= len(self.demand) - 1:
            return {"status": "complete", "timestamp": self.timestamp}
        position = self.current_position + 1
        timestamp = self.demand.index[position]
        actual = float(self.demand.iloc[position])
        self.actual_history.loc[timestamp] = actual
        self.last_drift = False
        adaptive_match = self.adaptive_registry.fixed_horizon_errors(timestamp, actual, 24)
        static_match = self.static_registry.fixed_horizon_errors(timestamp, actual, 24)
        if not static_match.empty:
            self.errors.append({**static_match.iloc[0].to_dict(), "model_policy": "static"})
        if not adaptive_match.empty:
            adaptive_row = adaptive_match.iloc[0].to_dict()
            self.errors.append({**adaptive_row, "model_policy": "adaptive"})
            if self.model_family == "xgboost":
                self.last_drift = self.replay.observe_error(
                    timestamp, float(adaptive_row["absolute_error_mw"]), self.actual_history
                )
            else:
                self.last_drift = bool(self.detector.update(float(adaptive_row["absolute_error_mw"])))
            if self.last_drift:
                change = getattr(self.detector, "last_change", None)
                self.drift_events.append({
                    "drift_timestamp": timestamp,
                    "error_mean_before_mw": getattr(change, "error_mean_before_mw", None),
                    "error_mean_after_mw": getattr(change, "error_mean_after_mw", None),
                    "direction": getattr(change, "direction", "detected"),
                })
                # Keep the adaptive comparison model learning in parallel,
                # regardless of which forecast policy drives dispatch. The
                # static model remains frozen; policy only selects the plan input.
                if self.model_family == "mlp":
                    self.replay._adapt(timestamp, change)

        if self.renewable_provider is None:
            actual_resource = self._resource_actuals.iloc[position]
            actual_solar_mw = float(actual_resource.solar_actual_mw)
            actual_wind_mw = float(actual_resource.wind_actual_mw)
            resource_source = "simulated seeded resource outcomes"
        else:
            actual_resource = self.renewable_provider.actual_at(
                timestamp,
                scale=self.config.demand_scale,
            )
            actual_solar_mw = float(actual_resource["solar_actual_mw"])
            actual_wind_mw = float(actual_resource["wind_actual_mw"])
            resource_source = str(actual_resource["source"])
        actual_resources = {"solar": np.array([actual_solar_mw]), "wind": np.array([actual_wind_mw])}
        realized_assets = self._assets_for_resource_availability(actual_resources)
        actual_hydro_inflow_mw = float(self._hydro_inflow_rate(pd.DatetimeIndex([timestamp]))[0])
        realized = optimize_dispatch(
            [actual * self.config.demand_scale],
            actual_resources,
            realized_assets,
            self.state,
            timestamps=[timestamp],
            hydro_inflow_mw=[actual_hydro_inflow_mw],
            **self._renewable_costs(pd.DatetimeIndex([timestamp])),
            terminal_soc_policy="none",
            battery_charging_policy=self.config.battery_charging_policy,
            allow_hydro_battery_charging=self.config.allow_hydro_battery_charging,
        )
        row = realized.schedule.iloc[0].to_dict()
        row.update({"timestamp": timestamp, "actual_demand_unscaled_mw": actual,
                    "resource_data_source": resource_source,
                    "solar_actual_source_scaled_mw": actual_solar_mw,
                    "wind_actual_source_scaled_mw": actual_wind_mw,
                    "hydro_inflow_source_mwh": actual_hydro_inflow_mw * self.assets.interval_hours,
                    "planned_first_hour_cost_eur": float(self.plan.schedule.cost_total_eur.iloc[0]),
                    "planned_first_hour_hydro_to_demand_mw": float(self.plan.schedule.hydro_to_demand_mw.iloc[0]),
                    "planned_first_hour_hydro_to_battery_mw": float(self.plan.schedule.hydro_to_battery_mw.iloc[0]),
                    "planned_first_hour_hydro_spill_mw": float(self.plan.schedule.hydro_spill_mw.iloc[0]),
                    "ending_battery_soc_mwh": realized.first_hour_state.battery_soc_mwh,
                    "ending_hydro_remaining_mwh": realized.first_hour_state.hydro_remaining_mwh,
                    "realized_cost_eur": realized.total_cost_eur})
        self.realized.append(row)
        self.accumulated_realized_cost_eur += realized.total_cost_eur
        self.state = realized.first_hour_state
        self.current_position = position
        self._issue_plan()
        return {"status": self.status, "timestamp": timestamp, "drift": self.last_drift,
                "actual_demand_mw": actual, "realized_cost_eur": realized.total_cost_eur}

    def step_n(self, count: int) -> list[dict]:
        if count < 1:
            raise ValueError("step count must be positive")
        return [self.step() for _ in range(count) if self.status != "complete"]

    def history_frame(self) -> pd.DataFrame:
        """Only actual observations through simulated time are returned."""
        return self.actual_history.loc[: self.timestamp].to_frame()

    def forecast_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"predicted_demand_mw": self.forecast})

    def metrics(self) -> dict[str, float]:
        errors = pd.DataFrame(self.errors)
        result = {}
        for policy in ("static", "adaptive"):
            part = errors.loc[errors.model_policy == policy] if not errors.empty else errors
            e = part.absolute_error_mw.to_numpy() if not part.empty else np.array([])
            result[f"{policy}_mae_mw"] = float(e.mean()) if len(e) else float("nan")
            result[f"{policy}_rmse_mw"] = float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")
        return result
