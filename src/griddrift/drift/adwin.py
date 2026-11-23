"""A stateful River ADWIN wrapper for sequential forecast errors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ADWINConfig:
    delta: float = 0.002
    clock: int = 32
    min_window_length: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.delta < 1:
            raise ValueError("ADWIN delta must be between 0 and 1")
        if self.clock < 1 or self.min_window_length < 1:
            raise ValueError("ADWIN clock and min_window_length must be >= 1")


@dataclass(frozen=True)
class ErrorDistributionChange:
    """ADWIN's estimated error level immediately around an accepted cut."""

    error_mean_before_mw: float
    error_mean_after_mw: float
    window_width_before: int
    window_width_after: int

    @property
    def direction(self) -> str:
        if self.error_mean_after_mw > self.error_mean_before_mw:
            return "error_increased"
        if self.error_mean_after_mw < self.error_mean_before_mw:
            return "error_decreased"
        return "error_unchanged"


class PersistentADWIN:
    """One detector instance, updated once for each observed 24-hour error."""

    def __init__(self, config: ADWINConfig = ADWINConfig()) -> None:
        try:
            from river import drift
        except ImportError as exc:
            raise RuntimeError("River is required for ADWIN. Install the [drift] project extra.") from exc
        self.config = config
        self._detector = drift.ADWIN(
            delta=config.delta,
            clock=config.clock,
            min_window_length=config.min_window_length,
        )
        self.observations = 0
        self.last_change: ErrorDistributionChange | None = None

    def update(self, absolute_error_mw: float) -> bool:
        if absolute_error_mw < 0:
            raise ValueError("ADWIN input must be a non-negative absolute error")
        error_mean_before = float(self._detector.estimation)
        window_width_before = int(self._detector.width)
        self._detector.update(float(absolute_error_mw))
        self.observations += 1
        drift_detected = bool(self._detector.drift_detected)
        self.last_change = (
            ErrorDistributionChange(
                error_mean_before_mw=error_mean_before,
                error_mean_after_mw=float(self._detector.estimation),
                window_width_before=window_width_before,
                window_width_after=int(self._detector.width),
            )
            if drift_detected
            else None
        )
        return drift_detected
