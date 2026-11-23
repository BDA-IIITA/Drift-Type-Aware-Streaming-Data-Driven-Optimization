"""Cached ENTSO-E actual-load downloads.

The API token is read only from ``ENTSOE_API_TOKEN``. It is intentionally never
included in filenames, exceptions, metadata, or logs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import pandas as pd

from griddrift.data.preprocessing import normalize_demand


class EntsoeLoadError(RuntimeError):
    """An ENTSO-E request or response could not produce usable actual load."""


class LoadClient(Protocol):
    def query_load(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series | pd.DataFrame: ...


def _utc_timestamp(value: str | pd.Timestamp, *, end: bool = False) -> pd.Timestamp:
    """Parse a UTC bound; date-only end values represent an inclusive calendar day."""
    text = str(value)
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    if end and len(text) == 10:
        stamp += pd.Timedelta(days=1)
    return stamp


def _cache_path(
    cache_dir: Path, bidding_zone: str, start: pd.Timestamp, end: pd.Timestamp, resolution: str
) -> Path:
    safe_zone = bidding_zone.replace("-", "_").upper()
    return cache_dir / (
        f"entsoe_actual_load_{safe_zone}_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}_{resolution}.csv"
    )


def _load_cache(path: Path, resolution: str) -> pd.DataFrame:
    cached = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    cached.index = pd.DatetimeIndex(cached.index)
    if cached.index.tz is None:
        cached.index = cached.index.tz_localize("UTC")
    else:
        cached.index = cached.index.tz_convert("UTC")
    cached.index.name = "timestamp"
    return normalize_demand(cached.reset_index(), resolution=resolution)


def _series_from_response(response: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(response, pd.Series):
        return response
    if not isinstance(response, pd.DataFrame):
        raise EntsoeLoadError("ENTSO-E client returned neither a pandas Series nor DataFrame")
    numeric = response.select_dtypes(include="number")
    if numeric.shape[1] != 1:
        raise EntsoeLoadError(
            "ENTSO-E actual-load response has an ambiguous number of numeric columns; "
            "select a single actual-load series before retrying"
        )
    return numeric.iloc[:, 0]


def _default_client(api_token: str) -> LoadClient:
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise EntsoeLoadError(
            "ENTSO-E support is not installed. Install the project with the [data] extra."
        ) from exc
    return EntsoePandasClient(api_key=api_token)


def fetch_entsoe_actual_load(
    *,
    bidding_zone: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    resolution: str = "1h",
    cache_dir: str | Path = "data/raw/entsoe_cache",
    force_refresh: bool = False,
    api_token: str | None = None,
    client: LoadClient | None = None,
) -> tuple[pd.DataFrame, Path, bool]:
    """Fetch actual demand, normalize it to hourly UTC MW, and cache success.

    ``end`` is exclusive when it includes a time. A date-only end value is
    inclusive, so ``2024-12-31`` requests through 2025-01-01T00:00Z. A failed
    request raises :class:`EntsoeLoadError`; synthetic data is never substituted.
    The returned boolean is true only when an existing cache file was used.
    """
    interval = pd.Timedelta(resolution)
    if interval <= pd.Timedelta(0):
        raise ValueError("resolution must be positive")
    if not bidding_zone:
        raise ValueError("bidding_zone is required")
    start_utc = _utc_timestamp(start)
    end_utc = _utc_timestamp(end, end=True)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")

    path = _cache_path(Path(cache_dir), bidding_zone, start_utc, end_utc, resolution)
    if path.exists() and not force_refresh:
        return _load_cache(path, resolution), path, True

    token = api_token or os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise EntsoeLoadError(
            "ENTSOE_API_TOKEN is not set and no cached ENTSO-E file is available. "
            "Set the environment variable before requesting real data."
        )

    query_client = client or _default_client(token)
    code = bidding_zone.replace("-", "_").upper()
    try:
        response = query_client.query_load(country_code=code, start=start_utc, end=end_utc)
    except Exception:  # Suppress the client exception; its URL may contain the API token.
        raise EntsoeLoadError(
            f"ENTSO-E actual-load request failed for {bidding_zone} between "
            f"{start_utc.isoformat()} and {end_utc.isoformat()}; no synthetic fallback was used"
        ) from None

    series = _series_from_response(response)
    raw = pd.DataFrame({"timestamp": series.index, "demand_mw": series.to_numpy()})
    try:
        hourly = normalize_demand(raw, resolution=resolution)
    except (KeyError, TypeError, ValueError) as exc:
        raise EntsoeLoadError("ENTSO-E response could not be normalized to complete hourly UTC MW") from exc

    expected = pd.date_range(start_utc, end_utc - interval, freq=resolution)
    hourly = hourly.reindex(expected)
    if hourly["demand_mw"].isna().any():
        missing = hourly.index[hourly["demand_mw"].isna()]
        raise EntsoeLoadError(
            f"ENTSO-E response is missing {len(missing)} requested {resolution} record(s); first={missing[0]}"
        )
    hourly.index.name = "timestamp"
    path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(path, index=True)
    return hourly, path, False
