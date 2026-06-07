"""
weather.py

Open-Meteo Historical Weather Client — Slice 2.

Fetches observed weather for a specified date range and the 30-year WMO
baseline series (1991–2020) for a given LocationResult.

Public interface:
  fetch_weather_observation(location, start_date, end_date) → WeatherObservation
  fetch_baseline_series(location, calendar_month)           → list[dict]

API reference: https://open-meteo.com/en/docs/historical-weather-api
Archive endpoint: https://archive-api.open-meteo.com/v1/archive
"""

import requests
from datetime import date, timedelta
from statistics import mean
from typing import Optional

from src.schema import LocationResult, WeatherObservation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Different timeouts to account for payload size differences
TIMEOUT_OBSERVED_SECS = 30    # ≤ 90 days of data
TIMEOUT_BASELINE_SECS = 120   # 30 years of daily data (~10 950 rows)

BASELINE_START_YEAR = 1991
BASELINE_END_YEAR = 2020

# Open-Meteo archive typically lags 5 days behind the current date
ARCHIVE_LAG_DAYS = 5

# >15% null temperature values → downgrade quality flag (CLAUDE.md non-negotiable)
MISSING_DATA_THRESHOLD_PCT = 15.0

# Daily variables available in the ERA5 archive at daily resolution
_DAILY_VARIABLES = "temperature_2m_mean,precipitation_sum,windspeed_10m_max"

# Hourly dewpoint is requested for the observed period only; not used for the
# 30-year baseline because the hourly payload would be ~262 800 rows.
_HOURLY_DEWPOINT = "dewpoint_2m"


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def fetch_weather_observation(
    location: LocationResult,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> WeatherObservation:
    """
    Fetches observed weather for a given location and date range.

    Defaults to a 30-day window ending ARCHIVE_LAG_DAYS before today,
    which avoids requesting dates the archive has not yet processed.

    Hourly dewpoint_2m values are fetched alongside daily data and
    aggregated to daily means — this is the correct approach because
    the Open-Meteo archive does not expose dewpoint as a native daily
    aggregate variable.

    Args:
        location:   A validated LocationResult from geocoding.py.
        start_date: ISO 8601 "YYYY-MM-DD". Defaults to 35 days ago.
        end_date:   ISO 8601 "YYYY-MM-DD". Defaults to 5 days ago.

    Returns:
        WeatherObservation with observed temperature, precipitation, and
        mean dewpoint for the requested window.

    Raises:
        ValueError:      Date range invalid, or API returned no usable data.
        ConnectionError: Archive API is unreachable or returned an HTTP error.
    """
    start, end = _resolve_date_range(start_date, end_date)
    records = _fetch_daily_with_dewpoint(location, start, end, TIMEOUT_OBSERVED_SECS)
    return _build_weather_observation(location, start, end, records)


def fetch_baseline_series(
    location: LocationResult,
    calendar_month: int,
) -> list[dict]:
    """
    Fetches daily temperature and precipitation records for a given calendar
    month across the full 1991–2020 WMO baseline period.

    A single API call retrieves the entire 30-year range; the results are
    then filtered to the target month in Python. This avoids 30 separate
    yearly requests.

    Each returned dict has keys:
        date            str          — "YYYY-MM-DD"
        temp_mean_c     float | None
        precip_sum_mm   float | None
        dewpoint_mean_c None         — not fetched at baseline scale (see note above)

    For a 30-day month, expect ~900 records (30 years × 30 days).

    This function is intended to be called by climate_stats.py (Slice 3)
    to compute baseline_mean_c, baseline_stddev_c, CDD/HDD baseline, and
    the 10-year trend slope.

    Args:
        location:       A validated LocationResult from geocoding.py.
        calendar_month: Integer 1–12.

    Returns:
        List of daily dicts filtered to the target month, 1991–2020.

    Raises:
        ValueError:      calendar_month outside 1–12, or no data returned.
        ConnectionError: Archive API is unreachable or returned an HTTP error.
    """
    if not 1 <= calendar_month <= 12:
        raise ValueError(
            f"calendar_month must be between 1 and 12, got {calendar_month}."
        )

    start = f"{BASELINE_START_YEAR}-01-01"
    end = f"{BASELINE_END_YEAR}-12-31"

    raw = _fetch_daily_only(location, start, end, TIMEOUT_BASELINE_SECS)

    month_str = f"{calendar_month:02d}"
    records = [r for r in raw if r["date"][5:7] == month_str]

    if not records:
        raise ValueError(
            f"No baseline records found for month {calendar_month} at "
            f"{location.city_name}. The API returned data but the month "
            "filter matched nothing — this is unexpected."
        )

    return records


# ---------------------------------------------------------------------------
# Private: Date Handling
# ---------------------------------------------------------------------------

def _resolve_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[str, str]:
    """
    Resolves optional date strings to a validated (start, end) pair.
    Default window: 30 days ending ARCHIVE_LAG_DAYS before today.

    Raises:
        ValueError: Malformed date strings, or start is after end.
    """
    today = date.today()
    default_end = today - timedelta(days=ARCHIVE_LAG_DAYS)
    default_start = default_end - timedelta(days=29)   # inclusive → 30-day window

    start_str = start_date or default_start.isoformat()
    end_str = end_date or default_end.isoformat()

    try:
        start_d = date.fromisoformat(start_str)
        end_d = date.fromisoformat(end_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid date format — expected YYYY-MM-DD. Detail: {exc}"
        ) from exc

    if start_d > end_d:
        raise ValueError(
            f"start_date ({start_str}) must not be later than end_date ({end_str})."
        )

    # ERA5 reanalysis starts in 1940; earlier dates are unreliable
    earliest_valid = date(1940, 1, 1)
    if start_d < earliest_valid:
        raise ValueError(
            f"start_date ({start_str}) predates the archive's reliable coverage "
            f"(earliest: {earliest_valid.isoformat()})."
        )

    return start_str, end_str


# ---------------------------------------------------------------------------
# Private: API Clients
# ---------------------------------------------------------------------------

def _fetch_daily_with_dewpoint(
    location: LocationResult,
    start_date: str,
    end_date: str,
    timeout: int,
) -> list[dict]:
    """
    Fetches daily temperature/precipitation AND hourly dewpoint in a single
    API call. Hourly dewpoint values are aggregated to daily means.

    Returns a list of dicts with keys:
        date, temp_mean_c, precip_sum_mm, dewpoint_mean_c
    """
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": _DAILY_VARIABLES,
        "hourly": _HOURLY_DEWPOINT,
        "timezone": "UTC",
    }

    data = _call_archive_api(params, timeout, location.city_name)

    daily_section = data.get("daily", {})
    hourly_section = data.get("hourly", {})

    dates = daily_section.get("time", [])
    if not dates:
        raise ValueError(
            f"Archive API returned no daily data for {location.city_name} "
            f"({start_date} to {end_date}). Date range may be outside coverage."
        )

    temps = daily_section.get("temperature_2m_mean", [None] * len(dates))
    precips = daily_section.get("precipitation_sum", [None] * len(dates))
    winds = daily_section.get("windspeed_10m_max", [None] * len(dates))

    # Aggregate hourly dewpoint to one mean value per calendar day
    daily_dewpoints = _aggregate_hourly_to_daily(
        hourly_section.get("time", []),
        hourly_section.get("dewpoint_2m", []),
    )

    return [
        {
            "date": dates[i],
            "temp_mean_c": temps[i] if i < len(temps) else None,
            "precip_sum_mm": precips[i] if i < len(precips) else None,
            "dewpoint_mean_c": daily_dewpoints.get(dates[i]),
            "wind_speed_max_ms": winds[i] if i < len(winds) else None,
        }
        for i in range(len(dates))
    ]


def _fetch_daily_only(
    location: LocationResult,
    start_date: str,
    end_date: str,
    timeout: int,
) -> list[dict]:
    """
    Fetches daily temperature and precipitation only — no hourly data.
    Used for the baseline series where the hourly dewpoint payload would
    be prohibitively large (~262 800 rows for 30 years).

    Returns a list of dicts with keys:
        date, temp_mean_c, precip_sum_mm, dewpoint_mean_c (always None)
    """
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": _DAILY_VARIABLES,
        "timezone": "UTC",
    }

    data = _call_archive_api(params, timeout, location.city_name)
    daily_section = data.get("daily", {})

    dates = daily_section.get("time", [])
    if not dates:
        raise ValueError(
            f"Archive API returned no daily data for {location.city_name} "
            f"({start_date} to {end_date})."
        )

    temps = daily_section.get("temperature_2m_mean", [None] * len(dates))
    precips = daily_section.get("precipitation_sum", [None] * len(dates))
    winds = daily_section.get("windspeed_10m_max", [None] * len(dates))

    return [
        {
            "date": dates[i],
            "temp_mean_c": temps[i] if i < len(temps) else None,
            "precip_sum_mm": precips[i] if i < len(precips) else None,
            "dewpoint_mean_c": None,
            "wind_speed_max_ms": winds[i] if i < len(winds) else None,
        }
        for i in range(len(dates))
    ]


def _call_archive_api(
    params: dict,
    timeout: int,
    city_name: str,
) -> dict:
    """
    Makes a single GET request to the Open-Meteo archive API and returns
    the parsed JSON body.

    Raises:
        ConnectionError: On any network failure or HTTP error status.
        ValueError:      If the API reports an application-level error in
                         the response body (e.g. unsupported variable).
    """
    try:
        response = requests.get(
            ARCHIVE_API_URL,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            "Could not reach the Open-Meteo archive API. "
            f"Check your internet connection. Detail: {exc}"
        ) from exc

    except requests.exceptions.Timeout:
        raise ConnectionError(
            f"Open-Meteo archive API timed out after {timeout}s for "
            f"{city_name}. Try again later."
        )

    except requests.exceptions.HTTPError as exc:
        raise ConnectionError(
            f"Open-Meteo archive API returned an HTTP error: {exc}"
        ) from exc

    data = response.json()

    # Open-Meteo signals application-level errors in the response body
    if data.get("error"):
        raise ValueError(
            f"Open-Meteo archive API error for {city_name}: "
            f"{data.get('reason', 'Unknown error')}"
        )

    return data


# ---------------------------------------------------------------------------
# Private: Data Processing
# ---------------------------------------------------------------------------

def _aggregate_hourly_to_daily(
    hourly_times: list[str],
    hourly_values: list[Optional[float]],
) -> dict[str, Optional[float]]:
    """
    Aggregates hourly time-series data into per-day means.

    Open-Meteo hourly timestamps are ISO 8601 strings of the form
    "YYYY-MM-DDTHH:MM". The date key is the 10-character YYYY-MM-DD prefix.

    Returns:
        dict mapping "YYYY-MM-DD" → mean of non-null hourly values for that day,
        or None if all hourly values for that day are null.
    """
    daily_buckets: dict[str, list[float]] = {}

    for ts, val in zip(hourly_times, hourly_values):
        day = ts[:10]
        if val is not None:
            daily_buckets.setdefault(day, []).append(val)

    return {
        day: round(mean(vals), 2) if vals else None
        for day, vals in daily_buckets.items()
    }


def _build_weather_observation(
    location: LocationResult,
    start_date: str,
    end_date: str,
    records: list[dict],
) -> WeatherObservation:
    """
    Aggregates a list of daily records into a single period-level
    WeatherObservation dataclass.

    Aggregation rules:
      - observed_temp_mean_c   → mean of non-null daily means
      - observed_precip_sum_mm → sum of non-null daily precipitation totals
      - dewpoint_mean_c        → mean of non-null daily dewpoint means; 0.0 if
                                 all dewpoints are unavailable (flagged as partial)

    Missing data rule (non-negotiable per CLAUDE.md):
      > 15% null temperature values → data_quality_flag = "partial"
      100% null temperature         → raises ValueError (no observation possible)

    Raises:
        ValueError: If records is empty or all temperature values are null.
    """
    total = len(records)
    if total == 0:
        raise ValueError(
            f"No daily records available for {location.city_name} "
            f"({start_date} to {end_date}). Cannot build a WeatherObservation."
        )

    temps = [r["temp_mean_c"] for r in records if r["temp_mean_c"] is not None]
    precips = [r["precip_sum_mm"] for r in records if r["precip_sum_mm"] is not None]
    dewpoints = [r["dewpoint_mean_c"] for r in records if r["dewpoint_mean_c"] is not None]
    winds = [r["wind_speed_max_ms"] for r in records if r.get("wind_speed_max_ms") is not None]

    if not temps:
        raise ValueError(
            f"All temperature values are null for {location.city_name} "
            f"({start_date} to {end_date}). Cannot produce a reliable observation."
        )

    missing_pct = ((total - len(temps)) / total) * 100.0
    quality_flag = "partial" if missing_pct > MISSING_DATA_THRESHOLD_PCT else "complete"

    return WeatherObservation(
        location=location,
        date_range_start=start_date,
        date_range_end=end_date,
        observed_temp_mean_c=round(mean(temps), 2),
        observed_precip_sum_mm=round(sum(precips), 2) if precips else 0.0,
        dewpoint_mean_c=round(mean(dewpoints), 2) if dewpoints else 0.0,
        wind_speed_max_ms=round(mean(winds), 2) if winds else 0.0,
        data_source="open-meteo",
        data_quality_flag=quality_flag,
    )
