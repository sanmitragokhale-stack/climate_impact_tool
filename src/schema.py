"""
schema.py

Shared dataclass definitions for the three-layer data contract.
Each slice populates one layer; no layer imports from a higher one.

Layer 1 — LocationResult        populated by: geocoding.py
Layer 2 — WeatherObservation    populated by: weather.py
          ClimateContext         populated by: climate_stats.py
Layer 3 — EconomicImpact        populated by: economic_impact.py
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Layer 1: Location Result
# ---------------------------------------------------------------------------

@dataclass
class LocationResult:
    """
    Output of the Location Engine (geocoding.py).
    Represents a validated, resolved city with coordinates.
    All downstream layers receive this as their location reference.
    """
    city_name: str          # Canonical name returned by geocoding API
    country: str            # ISO country name (e.g. "France")
    country_code: str       # ISO 3166-1 alpha-2 (e.g. "FR")
    latitude: float
    longitude: float
    admin_region: str       # State / province / region if available

    population: Optional[int] = None  # Helps disambiguate same-name cities

    # Data quality flags — surfaced in the UI and passed to LLM payload
    match_confidence: str = "high"    # "high" | "medium" | "low"
    match_note: str = ""              # Human-readable note on any ambiguity


# ---------------------------------------------------------------------------
# Layer 2: Weather Observation (populated by weather.py — Slice 2)
# ---------------------------------------------------------------------------

@dataclass
class WeatherObservation:
    """
    Raw observed weather data for the current/recent period.
    Populated by weather.py in Slice 2.
    """
    location: Optional[LocationResult] = None

    # Date range of the observation window (ISO 8601)
    date_range_start: str = ""
    date_range_end: str = ""

    # Core meteorological fields
    observed_temp_mean_c: float = 0.0
    observed_precip_sum_mm: float = 0.0
    dewpoint_mean_c: float = 0.0       # Required for wet-bulb calculation

    # Data provenance
    data_source: str = "open-meteo"
    data_quality_flag: str = "complete"  # "complete" | "partial" | "interpolated"


# ---------------------------------------------------------------------------
# Layer 2: Climate Context (populated by climate_stats.py — Slice 3)
# ---------------------------------------------------------------------------

@dataclass
class ClimateContext:
    """
    Statistical climate analysis against the 1991–2020 WMO baseline.
    Populated by climate_stats.py in Slice 3.

    Z-score thresholds (non-negotiable per architecture):
      |Z| <= 1.5  → within normal variability, no anomaly narrative
      |Z| >  2.0  → notable anomaly
      |Z| >  3.0  → exceptional anomaly
    """
    observation: Optional[WeatherObservation] = None

    # Baseline parameters — always 1991–2020
    baseline_period: str = "1991-2020"
    baseline_mean_c: float = 0.0
    baseline_stddev_c: float = 0.0

    # Anomaly signal
    z_score: float = 0.0
    anomaly_classification: str = ""   # "normal" | "notable" | "exceptional"

    # Degree day metrics (base temperature = 18°C / 65°F)
    cdd_observed: float = 0.0          # Cooling Degree Days, observed period
    cdd_baseline: float = 0.0          # Cooling Degree Days, historical mean
    cdd_delta: float = 0.0             # Difference (observed − baseline)

    hdd_observed: float = 0.0          # Heating Degree Days, observed period
    hdd_baseline: float = 0.0          # Heating Degree Days, historical mean
    hdd_delta: float = 0.0             # Difference (observed − baseline)

    # Heat stress metric (Stull 2011 approximation)
    wet_bulb_temp_c: float = 0.0

    # Trend signal — linear regression slope on last 10 years, same calendar period
    trend_slope_c_per_decade: float = 0.0

    # Overall confidence in this layer's output
    confidence: str = "high"           # "high" | "medium" | "low"
    confidence_note: str = ""


# ---------------------------------------------------------------------------
# Layer 3: Economic Impact (populated by economic_impact.py — Slice 4)
# ---------------------------------------------------------------------------

@dataclass
class EconomicImpact:
    """
    Localised financial impact estimate based on CDD/HDD delta and
    regional electricity pricing. Populated by economic_impact.py in Slice 4.

    Pricing tiers (5-tier proxy system):
      Tier 1 — USA, Canada            (EIA, NRCan)         High confidence
      Tier 2 — EU27, UK, Norway       (EUROSTAT)           High confidence
      Tier 3 — Major OECD             (IEA)                Medium confidence
      Tier 4 — Emerging markets       (World Bank)         Medium-Low confidence
      Tier 5 — All others             (Regional proxy)     Low — always flagged

    All cost estimates are per 100m² residential unit per week.
    """
    climate_context: Optional[ClimateContext] = None

    # Pricing inputs
    electricity_price_per_kwh_usd: float = 0.0
    electricity_price_source: str = ""       # e.g. "EIA", "EUROSTAT", "proxy"
    electricity_price_tier: int = 5          # 1–5; lower = more authoritative

    # Cost impact estimate
    delta_energy_cost_usd: float = 0.0       # Estimated additional cost
    per_unit_description: str = "per 100m² residential unit, per week"
    uncertainty_band_pct: float = 0.0        # e.g. 25.0 means ±25%

    # Overall confidence in this layer's output
    confidence: str = "low"                  # "high" | "medium" | "low"
    confidence_note: str = ""
