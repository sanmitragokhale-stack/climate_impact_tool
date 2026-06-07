"""
climate_stats.py

Climate Statistics Engine — Slice 3.

Consumes a WeatherObservation (Layer 2) and the 1991–2020 baseline series
produced by weather.py, and returns a populated ClimateContext dataclass.

Four calculations implemented (all non-negotiable per CLAUDE.md):
  1. Z-score anomaly detection against the WMO 1991–2020 baseline.
  2. Cooling/Heating Degree Days (CDD/HDD) for observed and baseline periods.
  3. Wet-bulb temperature via the Stull (2011) approximation.
  4. 10-year linear regression trend slope (2011–2020) in °C/decade.

Public interface:
  compute_climate_context(observation, baseline_series) → ClimateContext

The LLM synthesis layer (Slice 5) reads the returned ClimateContext.
It must never receive an anomaly classification for Z-scores within ±1.5.
"""

import math
from datetime import date
from statistics import mean, stdev
from typing import Optional

from src.schema import ClimateContext, WeatherObservation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WMO standard base temperature for degree-day calculations (18°C / 65°F)
DEGREE_DAY_BASE_C = 18.0

# The 10-year window within the baseline used for the trend signal
TREND_WINDOW_START_YEAR = 2011
TREND_WINDOW_END_YEAR = 2020

# Z-score thresholds (non-negotiable per CLAUDE.md architecture)
Z_NORMAL_THRESHOLD = 1.5     # |Z| ≤ 1.5 → "normal" — no anomaly narrative
Z_NOTABLE_THRESHOLD = 3.0    # |Z| > 3.0 → "exceptional"; 1.5 < |Z| ≤ 3.0 → "notable"

# Minimum baseline records required to compute mean and sample stddev.
# Production baselines typically have 900+ records; 3 is the hard floor
# (need at least 2 distinct values for a non-zero sample stddev).
MIN_BASELINE_RECORDS = 3


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def compute_climate_context(
    observation: WeatherObservation,
    baseline_series: list[dict],
) -> ClimateContext:
    """
    Derives climate statistics from an observed period and its 1991–2020 baseline.

    Calculations performed:
      • Baseline mean and sample stddev of temperature across all baseline records.
      • Z-score: (observed_mean − baseline_mean) / baseline_stddev.
      • Anomaly classification: "normal" (|Z| ≤ 1.5), "notable" (1.5 < |Z| ≤ 3.0),
        "exceptional" (|Z| > 3.0). Only "notable" and "exceptional" trigger anomaly
        narratives in the LLM synthesis layer.
      • Observed CDD/HDD: period-mean approximation over the observation window.
        Baseline CDD/HDD: computed from actual daily records averaged over 30 years.
      • Wet-bulb temperature via the Stull (2011) formula, requiring both observed
        temperature and dewpoint. Returns 0.0 with a confidence note if dewpoint
        is unavailable (dewpoint_mean_c == 0.0).
      • 10-year trend slope in °C/decade via linear regression on annual means
        for the years 2011–2020 in the baseline series.

    Args:
        observation:     A WeatherObservation from weather.py (Slice 2).
        baseline_series: Daily records for the same calendar month across 1991–2020,
                         as returned by fetch_baseline_series() in weather.py.

    Returns:
        A fully populated ClimateContext dataclass.

    Raises:
        ValueError: If the baseline series is empty, contains fewer than
                    MIN_BASELINE_RECORDS non-null temperature values, or if
                    the baseline stddev is zero (no measurable natural variability).
    """
    # --- Validate and extract baseline temperatures ---
    baseline_temps = _extract_temps(baseline_series)

    if len(baseline_temps) < MIN_BASELINE_RECORDS:
        raise ValueError(
            f"Baseline series has only {len(baseline_temps)} non-null temperature "
            f"records (minimum required: {MIN_BASELINE_RECORDS}). The series may be "
            "filtered to the wrong calendar month or the API returned sparse data."
        )

    null_count = len(baseline_series) - len(baseline_temps)
    null_fraction = null_count / len(baseline_series) if baseline_series else 1.0

    # --- Baseline statistics ---
    b_mean = mean(baseline_temps)
    b_stddev = stdev(baseline_temps)   # sample stddev (n−1 denominator)

    if b_stddev == 0.0:
        raise ValueError(
            f"Baseline temperature stddev is zero — all {len(baseline_temps)} "
            "baseline records have identical temperature values. "
            "Z-score is undefined. Check that the baseline series has realistic variation."
        )

    # --- Z-score and anomaly classification ---
    z = _z_score(observation.observed_temp_mean_c, b_mean, b_stddev)
    classification = _classify_anomaly(z)

    # --- Degree days ---
    n_days = _observation_day_count(observation)
    cdd_obs, hdd_obs = _degree_days_from_mean(
        observation.observed_temp_mean_c, n_days
    )
    cdd_base, hdd_base = _baseline_degree_days(baseline_series)

    # --- Wet-bulb temperature (Stull 2011) ---
    wet_bulb = _wet_bulb_from_observation(observation)

    # --- 10-year trend slope ---
    trend = _trend_slope(baseline_series)

    # --- Precipitation anomaly ---
    precip_z, precip_base_mm, precip_cls = _precip_anomaly(
        observation.observed_precip_sum_mm, baseline_series
    )

    # --- Wind anomaly ---
    wind_z, wind_base_ms, wind_cls = _wind_anomaly(
        observation.wind_speed_max_ms, baseline_series
    )

    # --- Drought indicator ---
    drought = _drought_indicator(precip_z)

    # --- Confidence and notes ---
    confidence, note = _assess_confidence(observation, null_fraction, z)

    return ClimateContext(
        observation=observation,
        baseline_period="1991-2020",
        baseline_mean_c=round(b_mean, 2),
        baseline_stddev_c=round(b_stddev, 2),
        z_score=round(z, 2),
        anomaly_classification=classification,
        cdd_observed=round(cdd_obs, 1),
        cdd_baseline=round(cdd_base, 1),
        cdd_delta=round(cdd_obs - cdd_base, 1),
        hdd_observed=round(hdd_obs, 1),
        hdd_baseline=round(hdd_base, 1),
        hdd_delta=round(hdd_obs - hdd_base, 1),
        wet_bulb_temp_c=round(wet_bulb, 2),
        trend_slope_c_per_decade=round(trend, 3),
        precip_observed_mm=round(observation.observed_precip_sum_mm, 1),
        precip_baseline_mm=round(precip_base_mm, 1),
        precip_z_score=round(precip_z, 2),
        precip_anomaly_classification=precip_cls,
        wind_speed_max_ms=round(observation.wind_speed_max_ms, 1),
        wind_baseline_ms=round(wind_base_ms, 1),
        wind_z_score=round(wind_z, 2),
        wind_anomaly_classification=wind_cls,
        drought_indicator=drought,
        confidence=confidence,
        confidence_note=note,
    )


# ---------------------------------------------------------------------------
# Private: Baseline helpers
# ---------------------------------------------------------------------------

def _extract_temps(baseline_series: list[dict]) -> list[float]:
    """Returns non-null temperature values from the baseline series."""
    return [r["temp_mean_c"] for r in baseline_series if r["temp_mean_c"] is not None]


# ---------------------------------------------------------------------------
# Private: Z-score
# ---------------------------------------------------------------------------

def _z_score(observed: float, b_mean: float, b_stddev: float) -> float:
    """Z = (observed − baseline_mean) / baseline_stddev."""
    return (observed - b_mean) / b_stddev


def _classify_anomaly(z: float) -> str:
    """
    Three-tier classification per CLAUDE.md:
      "normal"      — |Z| ≤ 1.5  → no anomaly narrative (CLAUDE.md non-negotiable)
      "notable"     — 1.5 < |Z| ≤ 3.0
      "exceptional" — |Z| > 3.0
    """
    abs_z = abs(z)
    if abs_z > Z_NOTABLE_THRESHOLD:
        return "exceptional"
    if abs_z > Z_NORMAL_THRESHOLD:
        return "notable"
    return "normal"


# ---------------------------------------------------------------------------
# Private: Degree Days
# ---------------------------------------------------------------------------

def _observation_day_count(observation: WeatherObservation) -> int:
    """Number of calendar days in the observation window (inclusive)."""
    start = date.fromisoformat(observation.date_range_start)
    end = date.fromisoformat(observation.date_range_end)
    return (end - start).days + 1


def _degree_days_from_mean(temp_mean_c: float, n_days: int) -> tuple[float, float]:
    """
    Period-mean approximation for CDD and HDD over an observation window.

    Because WeatherObservation stores only the period mean (not individual
    daily values), this approximation treats every day as having the same
    temperature. It is accurate enough for monthly summaries.

    Returns: (CDD, HDD) — both in degree-days (base 18°C).
    """
    cdd = max(0.0, temp_mean_c - DEGREE_DAY_BASE_C) * n_days
    hdd = max(0.0, DEGREE_DAY_BASE_C - temp_mean_c) * n_days
    return cdd, hdd


def _baseline_degree_days(baseline_series: list[dict]) -> tuple[float, float]:
    """
    Computes mean annual CDD and HDD from the baseline daily records.

    For each year present in the baseline series, sums the degree days
    from actual daily temperatures, then returns the 30-year mean.
    This uses real daily values (more accurate than a mean approximation).

    Returns: (mean_CDD, mean_HDD) across all baseline years.
    """
    # Group non-null daily temperatures by calendar year
    years: dict[str, list[float]] = {}
    for r in baseline_series:
        if r["temp_mean_c"] is not None:
            year = r["date"][:4]
            years.setdefault(year, []).append(r["temp_mean_c"])

    if not years:
        return 0.0, 0.0

    annual_cdds = [
        sum(max(0.0, t - DEGREE_DAY_BASE_C) for t in temps)
        for temps in years.values()
    ]
    annual_hdds = [
        sum(max(0.0, DEGREE_DAY_BASE_C - t) for t in temps)
        for temps in years.values()
    ]

    return mean(annual_cdds), mean(annual_hdds)


# ---------------------------------------------------------------------------
# Private: Wet-bulb Temperature (Stull 2011)
# ---------------------------------------------------------------------------

def _rh_from_dewpoint(temp_c: float, dewpoint_c: float) -> float:
    """
    Derives relative humidity from dry-bulb temperature and dewpoint using
    the August-Roche-Magnus formula.

    Valid for typical atmospheric conditions (−40°C to 60°C).
    Result is clamped to [1.0, 100.0] to keep the Stull formula stable.
    """
    a, b = 17.625, 243.04
    rh = 100.0 * math.exp((a * dewpoint_c) / (b + dewpoint_c)) / math.exp((a * temp_c) / (b + temp_c))
    return min(100.0, max(1.0, rh))


def _stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """
    Stull (2011) empirical wet-bulb approximation.

    Reference: Stull, R. (2011). "Wet-Bulb Temperature from Relative Humidity
    and Air Temperature." Journal of Applied Meteorology and Climatology, 50(11),
    2267–2269.

    Valid for: 5°C ≤ T ≤ 40°C, 5% ≤ RH ≤ 99%.
    Accuracy: typically within ±1°C in the valid range.

    Args:
        temp_c:  Air (dry-bulb) temperature in °C.
        rh_pct:  Relative humidity in percent (0–100).

    Returns:
        Wet-bulb temperature in °C.
    """
    return (
        temp_c * math.atan(0.151977 * (rh_pct + 8.313659) ** 0.5)
        + math.atan(temp_c + rh_pct)
        - math.atan(rh_pct - 1.676331)
        + 0.00391838 * rh_pct ** 1.5 * math.atan(0.023101 * rh_pct)
        - 4.686035
    )


def _wet_bulb_from_observation(observation: WeatherObservation) -> float:
    """
    Computes wet-bulb temperature from the observed period mean values.

    Returns 0.0 if dewpoint is unavailable (dewpoint_mean_c == 0.0), which
    occurs when the archive API did not supply humidity data. This case is
    surfaced in the confidence_note by _assess_confidence.
    """
    if observation.dewpoint_mean_c == 0.0:
        return 0.0

    rh = _rh_from_dewpoint(
        observation.observed_temp_mean_c,
        observation.dewpoint_mean_c,
    )
    return _stull_wet_bulb(observation.observed_temp_mean_c, rh)


# ---------------------------------------------------------------------------
# Private: Trend Slope
# ---------------------------------------------------------------------------

def _trend_slope(baseline_series: list[dict]) -> float:
    """
    10-year linear regression trend slope on annual mean temperatures.

    Uses only records from TREND_WINDOW_START_YEAR to TREND_WINDOW_END_YEAR
    (2011–2020 within the 1991–2020 baseline). Returns the slope in °C/decade.

    This is the climate shift signal: a positive value means warming within
    the baseline period. It is distinct from the Z-score anomaly signal.

    Returns 0.0 if fewer than 2 years of data are available (graceful
    degradation for unusual or very sparse baseline inputs).
    """
    # Group non-null daily temps by year, restricted to the 10-year window
    years: dict[int, list[float]] = {}
    for r in baseline_series:
        year = int(r["date"][:4])
        if TREND_WINDOW_START_YEAR <= year <= TREND_WINDOW_END_YEAR:
            if r["temp_mean_c"] is not None:
                years.setdefault(year, []).append(r["temp_mean_c"])

    if len(years) < 2:
        return 0.0

    x = sorted(years.keys())
    y = [mean(years[yr]) for yr in x]

    n = len(x)
    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0.0:
        return 0.0

    return (numerator / denominator) * 10.0   # °C/year → °C/decade


# ---------------------------------------------------------------------------
# Private: Precipitation anomaly
# ---------------------------------------------------------------------------

def _precip_anomaly(
    observed_mm: float,
    baseline_series: list[dict],
) -> tuple[float, float, str]:
    """
    Computes precipitation Z-score from the observed period total and the
    1991–2020 baseline monthly precip distribution.

    Returns (z_score, baseline_mean_mm, anomaly_classification).
    Returns (0.0, 0.0, "normal") gracefully when baseline data is sparse.
    """
    # Sum precip per year for the filtered month records, then take means
    year_totals: dict[str, float] = {}
    for r in baseline_series:
        if r.get("precip_sum_mm") is not None:
            year = r["date"][:4]
            year_totals[year] = year_totals.get(year, 0.0) + r["precip_sum_mm"]

    if len(year_totals) < MIN_BASELINE_RECORDS:
        return 0.0, 0.0, "normal"

    values = list(year_totals.values())
    b_mean = mean(values)
    b_std = stdev(values)

    if b_std == 0.0:
        return 0.0, b_mean, "normal"

    z = (observed_mm - b_mean) / b_std
    return z, b_mean, _classify_anomaly(z)


# ---------------------------------------------------------------------------
# Private: Wind anomaly
# ---------------------------------------------------------------------------

def _wind_anomaly(
    observed_ms: float,
    baseline_series: list[dict],
) -> tuple[float, float, str]:
    """
    Computes wind speed Z-score from observed mean-daily-max and baseline.

    Returns (z_score, baseline_mean_ms, anomaly_classification).
    Returns (0.0, 0.0, "normal") when data is unavailable.
    """
    if observed_ms == 0.0:
        return 0.0, 0.0, "normal"

    year_means: dict[str, list[float]] = {}
    for r in baseline_series:
        v = r.get("wind_speed_max_ms")
        if v is not None:
            year = r["date"][:4]
            year_means.setdefault(year, []).append(v)

    if len(year_means) < MIN_BASELINE_RECORDS:
        return 0.0, 0.0, "normal"

    annual_means = [mean(vals) for vals in year_means.values()]
    b_mean = mean(annual_means)
    b_std = stdev(annual_means)

    if b_std == 0.0:
        return 0.0, b_mean, "normal"

    z = (observed_ms - b_mean) / b_std
    return z, b_mean, _classify_anomaly(z)


# ---------------------------------------------------------------------------
# Private: Drought indicator
# ---------------------------------------------------------------------------

def _drought_indicator(precip_z: float) -> str:
    """
    Derives a drought risk label from the precipitation Z-score.
    A strongly negative Z indicates prolonged dryness relative to baseline.
    """
    if precip_z <= -3.0:
        return "severe"
    if precip_z <= -1.5:
        return "moderate"
    return "none"


# ---------------------------------------------------------------------------
# Private: Confidence Assessment
# ---------------------------------------------------------------------------

def _assess_confidence(
    observation: WeatherObservation,
    null_fraction: float,
    z: float,
) -> tuple[str, str]:
    """
    Determines the overall confidence tier and constructs a human-readable note.

    Confidence rules (highest priority first):
      "low"    — >15% null values in the baseline (CLAUDE.md non-negotiable).
      "medium" — Observed data quality is "partial" (observation period had
                 >15% missing daily values).
      "high"   — All data present and within quality thresholds.

    An additional note is always appended when Z is within ±1.5, instructing
    downstream consumers not to generate an anomaly narrative.
    """
    notes: list[str] = []

    if null_fraction > 0.15:
        confidence = "low"
        notes.append(
            f"Baseline data has {null_fraction * 100:.0f}% missing values "
            "(threshold: 15%); statistics may be unreliable."
        )
    elif observation.data_quality_flag == "partial":
        confidence = "medium"
        notes.append(
            "Observed period has partial data quality (>15% missing daily values)."
        )
    else:
        confidence = "high"

    # Graceful degradation: Z ≤ ±1.5 must not trigger anomaly narrative (CLAUDE.md)
    if abs(z) <= Z_NORMAL_THRESHOLD:
        notes.append(
            f"Z-score ({z:.2f}) is within ±{Z_NORMAL_THRESHOLD} σ — "
            "this is normal climate variability. "
            "Do not generate an anomaly narrative; return a variability note instead."
        )

    if observation.dewpoint_mean_c == 0.0:
        notes.append(
            "Dewpoint unavailable; wet-bulb temperature could not be computed (returned 0.0)."
        )

    return confidence, " ".join(notes)
