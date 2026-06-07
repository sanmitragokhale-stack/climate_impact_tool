"""
test_climate_stats.py

Validation suite for src/climate_stats.py — Slice 3: Climate Statistics Engine.

All tests are OFFLINE — no API calls, no network dependency.
Synthetic baseline data is used so that expected values are known exactly.

Synthetic baseline design (used across most tests):
  3 years × 1 record each → temps [18.0, 20.0, 22.0]
  mean  = 20.0°C  (exact)
  stdev = 2.0°C   (exact sample stddev, n−1 denominator)
  This gives clean, predictable Z-scores.

Run:
    python -m pytest tests/test_climate_stats.py -v
    python tests/test_climate_stats.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ClimateContext, LocationResult, WeatherObservation
from src.climate_stats import (
    compute_climate_context,
    # Private helpers exposed for unit testing
    _z_score,
    _classify_anomaly,
    _degree_days_from_mean,
    _baseline_degree_days,
    _rh_from_dewpoint,
    _stull_wet_bulb,
    _wet_bulb_from_observation,
    _trend_slope,
    DEGREE_DAY_BASE_C,
    Z_NORMAL_THRESHOLD,
    Z_NOTABLE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Synthetic Data Builders
# ---------------------------------------------------------------------------

def _make_observation(
    temp_mean_c: float = 20.0,
    dewpoint_c: float = 12.0,
    start: str = "2023-06-01",
    end: str = "2023-06-30",
    quality: str = "complete",
) -> WeatherObservation:
    """Build a minimal WeatherObservation for testing without an API call."""
    loc = LocationResult(
        city_name="TestCity",
        country="Testland",
        country_code="TS",
        latitude=48.0,
        longitude=2.0,
        admin_region="Test Region",
        match_confidence="high",
    )
    return WeatherObservation(
        location=loc,
        date_range_start=start,
        date_range_end=end,
        observed_temp_mean_c=temp_mean_c,
        observed_precip_sum_mm=50.0,
        dewpoint_mean_c=dewpoint_c,
        data_source="open-meteo",
        data_quality_flag=quality,
    )


def _make_baseline(
    temps_by_year: dict[int, float],
    month: int = 6,
    days_per_year: int = 1,
) -> list[dict]:
    """
    Build a synthetic baseline_series for testing.

    temps_by_year: {year: temperature} — single temperature applied to all
                   days_per_year records for that year.
    days_per_year: how many daily records to generate per year (default 1).

    With temps_by_year={1991:18, 1993:20, 1995:22} and days_per_year=1:
      → 3 records, mean=20.0, stdev=2.0 exactly.
    """
    records = []
    for year, temp in sorted(temps_by_year.items()):
        for day in range(1, days_per_year + 1):
            records.append({
                "date": f"{year}-{month:02d}-{day:02d}",
                "temp_mean_c": temp,
                "precip_sum_mm": 2.0,
                "dewpoint_mean_c": None,
            })
    return records


def _canonical_baseline(mean_c: float = 20.0) -> list[dict]:
    """
    The canonical 3-year synthetic baseline used across most tests.
    Temperatures [mean-2, mean, mean+2] → stdev = 2.0 exactly.
    """
    return _make_baseline({1991: mean_c - 2.0, 1993: mean_c, 1995: mean_c + 2.0})


# ---------------------------------------------------------------------------
# Z-score Unit Tests
# ---------------------------------------------------------------------------

def test_z_score_zero_at_baseline_mean():
    """Observed temperature equal to the baseline mean → Z = 0.0."""
    z = _z_score(20.0, 20.0, 2.0)
    assert z == 0.0, f"Expected Z=0.0, got {z}"
    print(f"  ✓ Z-score: observed=baseline_mean → Z={z:.2f}")


def test_z_score_positive_warm_anomaly():
    """Observed 2 stddevs above the baseline mean → Z = +2.0."""
    z = _z_score(24.0, 20.0, 2.0)
    assert abs(z - 2.0) < 1e-9, f"Expected Z=2.0, got {z}"
    print(f"  ✓ Z-score: warm anomaly → Z={z:.2f}")


def test_z_score_negative_cold_anomaly():
    """Observed 2 stddevs below the baseline mean → Z = −2.0."""
    z = _z_score(16.0, 20.0, 2.0)
    assert abs(z - (-2.0)) < 1e-9, f"Expected Z=-2.0, got {z}"
    print(f"  ✓ Z-score: cold anomaly → Z={z:.2f}")


# ---------------------------------------------------------------------------
# Anomaly Classification Tests (CLAUDE.md non-negotiable thresholds)
# ---------------------------------------------------------------------------

def test_classify_z_zero_is_normal():
    """Z = 0.0 → 'normal'."""
    assert _classify_anomaly(0.0) == "normal"
    print("  ✓ Z=0.0 → 'normal'")


def test_classify_z_at_boundary_1_5_is_normal():
    """
    Z = ±1.5 exactly is the boundary — must be 'normal', not 'notable'.
    CLAUDE.md: 'Z-score within ±1.5 → do not generate anomaly narrative.'
    """
    assert _classify_anomaly(1.5) == "normal", "Z=+1.5 must be 'normal'"
    assert _classify_anomaly(-1.5) == "normal", "Z=-1.5 must be 'normal'"
    print("  ✓ Z=±1.5 (boundary) → 'normal'")


def test_classify_z_just_above_1_5_is_notable():
    """Z = 1.501 (just above the normal threshold) → 'notable'."""
    result = _classify_anomaly(1.501)
    assert result == "notable", f"Expected 'notable', got '{result}'"
    result_neg = _classify_anomaly(-1.501)
    assert result_neg == "notable", f"Expected 'notable', got '{result_neg}'"
    print("  ✓ Z=±1.501 (just above normal) → 'notable'")


def test_classify_z_mid_range_notable():
    """Z = ±2.5 (well within the notable band) → 'notable'."""
    assert _classify_anomaly(2.5) == "notable"
    assert _classify_anomaly(-2.5) == "notable"
    print("  ✓ Z=±2.5 → 'notable'")


def test_classify_z_at_boundary_3_0_is_notable():
    """Z = ±3.0 exactly is the upper boundary of 'notable', not 'exceptional'."""
    assert _classify_anomaly(3.0) == "notable", "Z=+3.0 must be 'notable'"
    assert _classify_anomaly(-3.0) == "notable", "Z=-3.0 must be 'notable'"
    print("  ✓ Z=±3.0 (boundary) → 'notable'")


def test_classify_z_just_above_3_0_is_exceptional():
    """Z = 3.001 (just above the notable threshold) → 'exceptional'."""
    result = _classify_anomaly(3.001)
    assert result == "exceptional", f"Expected 'exceptional', got '{result}'"
    result_neg = _classify_anomaly(-3.001)
    assert result_neg == "exceptional", f"Expected 'exceptional', got '{result_neg}'"
    print("  ✓ Z=±3.001 (just above notable) → 'exceptional'")


def test_classify_z_large_positive_exceptional():
    """Z = 5.0 → 'exceptional'."""
    assert _classify_anomaly(5.0) == "exceptional"
    print("  ✓ Z=5.0 → 'exceptional'")


# ---------------------------------------------------------------------------
# Degree Day Unit Tests
# ---------------------------------------------------------------------------

def test_cdd_hot_period():
    """
    All days at 28°C → CDD = (28−18) × 30 = 300, HDD = 0.
    """
    cdd, hdd = _degree_days_from_mean(28.0, 30)
    assert abs(cdd - 300.0) < 1e-9, f"Expected CDD=300, got {cdd}"
    assert hdd == 0.0, f"Expected HDD=0, got {hdd}"
    print(f"  ✓ Hot period (28°C × 30d) → CDD={cdd:.1f}, HDD={hdd:.1f}")


def test_hdd_cold_period():
    """
    All days at 10°C → HDD = (18−10) × 30 = 240, CDD = 0.
    """
    cdd, hdd = _degree_days_from_mean(10.0, 30)
    assert cdd == 0.0, f"Expected CDD=0, got {cdd}"
    assert abs(hdd - 240.0) < 1e-9, f"Expected HDD=240, got {hdd}"
    print(f"  ✓ Cold period (10°C × 30d) → CDD={cdd:.1f}, HDD={hdd:.1f}")


def test_cdd_hdd_zero_at_base_temperature():
    """At exactly 18°C (the base temperature), both CDD and HDD must be zero."""
    cdd, hdd = _degree_days_from_mean(DEGREE_DAY_BASE_C, 30)
    assert cdd == 0.0, f"Expected CDD=0 at base temp, got {cdd}"
    assert hdd == 0.0, f"Expected HDD=0 at base temp, got {hdd}"
    print(f"  ✓ Base temperature (18°C × 30d) → CDD=0, HDD=0")


def test_baseline_cdd_computed_from_daily_records():
    """
    Baseline with 30 daily records at 25°C per year (3 years):
    annual CDD per year = (25−18) × 30 = 210
    expected mean baseline CDD = 210.
    """
    baseline = _make_baseline(
        {1991: 25.0, 1993: 25.0, 1995: 25.0},
        days_per_year=30,
    )
    cdd_base, hdd_base = _baseline_degree_days(baseline)
    assert abs(cdd_base - 210.0) < 1e-9, f"Expected CDD=210, got {cdd_base}"
    assert hdd_base == 0.0, f"Expected HDD=0, got {hdd_base}"
    print(f"  ✓ Baseline CDD from daily records → {cdd_base:.1f} (expected 210)")


def test_baseline_hdd_cold_records():
    """
    Baseline with 10 daily records at 8°C per year (3 years):
    annual HDD = (18−8) × 10 = 100
    expected mean baseline HDD = 100.
    """
    baseline = _make_baseline(
        {1991: 8.0, 1993: 8.0, 1995: 8.0},
        days_per_year=10,
    )
    cdd_base, hdd_base = _baseline_degree_days(baseline)
    assert cdd_base == 0.0, f"Expected CDD=0, got {cdd_base}"
    assert abs(hdd_base - 100.0) < 1e-9, f"Expected HDD=100, got {hdd_base}"
    print(f"  ✓ Baseline HDD from cold daily records → {hdd_base:.1f} (expected 100)")


# ---------------------------------------------------------------------------
# Wet-bulb Temperature Unit Tests (Stull 2011)
# ---------------------------------------------------------------------------

def test_wet_bulb_is_below_dry_bulb():
    """
    Fundamental physical constraint: Tw ≤ T for any air mass below saturation.
    Tested at T=30°C, RH=50% (moderate humidity).
    """
    rh = _rh_from_dewpoint(30.0, 18.0)    # RH ≈ 48.6%
    tw = _stull_wet_bulb(30.0, rh)
    assert tw < 30.0, f"Wet-bulb ({tw:.3f}°C) must be below dry-bulb (30°C)."
    print(f"  ✓ Tw={tw:.2f}°C < T=30°C (Tw < T constraint satisfied)")


def test_wet_bulb_approaches_dry_bulb_at_saturation():
    """
    At near-saturation (Td ≈ T → RH ≈ 100%), Tw must be very close to T.
    """
    T, Td = 25.0, 24.9
    rh = _rh_from_dewpoint(T, Td)
    tw = _stull_wet_bulb(T, rh)
    # At RH ≈ 99%, Tw should be within 0.1°C of T
    assert abs(tw - T) < 0.1, f"At saturation, Tw={tw:.3f}°C should be ≈ T={T}°C"
    print(f"  ✓ Near-saturation (Td=24.9, T=25) → Tw={tw:.3f}°C ≈ T=25°C")


def test_wet_bulb_reference_value():
    """
    Reference check: T=30°C, Td=18°C → Tw ≈ 22.05°C (verified analytically).
    Tolerance ±0.5°C accounts for the Stull (2011) approximation accuracy.
    """
    rh = _rh_from_dewpoint(30.0, 18.0)
    tw = _stull_wet_bulb(30.0, rh)
    expected = 22.05
    assert abs(tw - expected) < 0.5, (
        f"Tw={tw:.3f}°C is outside ±0.5°C of expected {expected}°C "
        "at T=30, Td=18."
    )
    print(f"  ✓ Reference wet-bulb: T=30, Td=18 → Tw={tw:.3f}°C (expected ≈{expected})")


def test_wet_bulb_is_above_dewpoint():
    """
    Secondary physical constraint: Tw ≥ Td (wet-bulb above dewpoint).
    """
    T, Td = 20.0, 15.0
    rh = _rh_from_dewpoint(T, Td)
    tw = _stull_wet_bulb(T, rh)
    assert tw >= Td, f"Tw={tw:.3f}°C must be ≥ Td={Td}°C."
    print(f"  ✓ Wet-bulb is above dewpoint: Tw={tw:.3f}°C ≥ Td={Td}°C")


def test_wet_bulb_unavailable_when_dewpoint_is_zero():
    """
    Graceful degradation: when dewpoint_mean_c == 0.0 (unavailable),
    _wet_bulb_from_observation must return 0.0 without raising an error.
    """
    obs = _make_observation(temp_mean_c=25.0, dewpoint_c=0.0)
    tw = _wet_bulb_from_observation(obs)
    assert tw == 0.0, f"Expected 0.0 for unavailable dewpoint, got {tw}"
    print("  ✓ Unavailable dewpoint (0.0) → wet-bulb returns 0.0 gracefully")


def test_rh_from_dewpoint_known_value():
    """
    RH from T=30°C, Td=18°C using the Magnus formula should be ≈ 48.6%.
    This is the intermediate step used by _wet_bulb_from_observation.
    """
    rh = _rh_from_dewpoint(30.0, 18.0)
    assert abs(rh - 48.6) < 1.0, f"RH={rh:.1f}% is outside 1% tolerance of 48.6%"
    print(f"  ✓ RH from T=30, Td=18 → RH={rh:.1f}% (expected ≈48.6%)")


# ---------------------------------------------------------------------------
# Trend Slope Unit Tests
# ---------------------------------------------------------------------------

def test_warming_trend_positive_slope():
    """
    10 years (2011–2020) with linear warming of 0.2°C/year
    → slope should be exactly 2.0°C/decade.
    """
    temps_by_year = {yr: 20.0 + 0.2 * (yr - 2011) for yr in range(2011, 2021)}
    baseline = _make_baseline(temps_by_year)
    slope = _trend_slope(baseline)
    assert abs(slope - 2.0) < 1e-9, f"Expected slope=2.0°C/decade, got {slope}"
    print(f"  ✓ Warming trend → slope={slope:.3f}°C/decade (expected 2.000)")


def test_cooling_trend_negative_slope():
    """
    10 years with linear cooling of 0.3°C/year → slope = −3.0°C/decade.
    """
    temps_by_year = {yr: 25.0 - 0.3 * (yr - 2011) for yr in range(2011, 2021)}
    baseline = _make_baseline(temps_by_year)
    slope = _trend_slope(baseline)
    assert abs(slope - (-3.0)) < 1e-9, f"Expected slope=-3.0°C/decade, got {slope}"
    print(f"  ✓ Cooling trend → slope={slope:.3f}°C/decade (expected -3.000)")


def test_flat_trend_near_zero_slope():
    """
    All 10 trend-window years at exactly the same temperature → slope = 0.0.
    """
    temps_by_year = {yr: 20.0 for yr in range(2011, 2021)}
    baseline = _make_baseline(temps_by_year)
    slope = _trend_slope(baseline)
    assert slope == 0.0, f"Expected slope=0.0 for flat data, got {slope}"
    print(f"  ✓ Flat trend → slope={slope:.3f}°C/decade (expected 0.000)")


def test_trend_uses_only_last_10_years():
    """
    Records for 1991–2010 have a strong cooling pattern; records for 2011–2020
    have a warming pattern. The trend slope must reflect only 2011–2020.
    """
    # 1991–2010: aggressively cooling (should be ignored)
    old_years = {yr: 30.0 - (yr - 1991) for yr in range(1991, 2011)}
    # 2011–2020: warming at 0.2°C/year (should drive the slope)
    new_years = {yr: 20.0 + 0.2 * (yr - 2011) for yr in range(2011, 2021)}

    baseline = _make_baseline({**old_years, **new_years})
    slope = _trend_slope(baseline)

    assert abs(slope - 2.0) < 1e-9, (
        f"Expected slope=2.0°C/decade (only 2011-2020), got {slope:.6f}. "
        "The function may be including pre-2011 records."
    )
    print(f"  ✓ Trend window isolation → slope={slope:.3f}°C/decade (2011-2020 only)")


def test_trend_with_insufficient_years_returns_zero():
    """
    Fewer than 2 years in the trend window → graceful degradation: return 0.0.
    """
    baseline = _make_baseline({2015: 20.0})   # Only 1 year in 2011–2020
    slope = _trend_slope(baseline)
    assert slope == 0.0, f"Expected 0.0 for <2 trend years, got {slope}"
    print(f"  ✓ Insufficient trend data (1 year) → slope=0.0 (graceful degradation)")


# ---------------------------------------------------------------------------
# compute_climate_context Integration Tests
# ---------------------------------------------------------------------------

def test_full_context_schema_types():
    """
    End-to-end schema contract: verify every field in ClimateContext has the
    correct Python type after a complete computation.
    """
    obs = _make_observation(temp_mean_c=23.0, dewpoint_c=14.0)
    baseline = _canonical_baseline()

    ctx = compute_climate_context(obs, baseline)

    assert isinstance(ctx, ClimateContext),           "result must be ClimateContext"
    assert isinstance(ctx.observation, WeatherObservation), "observation must be WeatherObservation"
    assert isinstance(ctx.baseline_period, str),      "baseline_period must be str"
    assert isinstance(ctx.baseline_mean_c, float),    "baseline_mean_c must be float"
    assert isinstance(ctx.baseline_stddev_c, float),  "baseline_stddev_c must be float"
    assert isinstance(ctx.z_score, float),            "z_score must be float"
    assert isinstance(ctx.anomaly_classification, str),"anomaly_classification must be str"
    assert isinstance(ctx.cdd_observed, float),       "cdd_observed must be float"
    assert isinstance(ctx.cdd_baseline, float),       "cdd_baseline must be float"
    assert isinstance(ctx.cdd_delta, float),          "cdd_delta must be float"
    assert isinstance(ctx.hdd_observed, float),       "hdd_observed must be float"
    assert isinstance(ctx.hdd_baseline, float),       "hdd_baseline must be float"
    assert isinstance(ctx.hdd_delta, float),          "hdd_delta must be float"
    assert isinstance(ctx.wet_bulb_temp_c, float),    "wet_bulb_temp_c must be float"
    assert isinstance(ctx.trend_slope_c_per_decade, float), "trend_slope_c_per_decade must be float"
    assert isinstance(ctx.confidence, str),           "confidence must be str"
    assert isinstance(ctx.confidence_note, str),      "confidence_note must be str"

    print("  ✓ ClimateContext schema contract — all field types correct")


def test_baseline_period_label_is_wmo_standard():
    """The baseline_period field must always be '1991-2020' (WMO standard)."""
    ctx = compute_climate_context(_make_observation(), _canonical_baseline())
    assert ctx.baseline_period == "1991-2020", \
        f"Expected '1991-2020', got '{ctx.baseline_period}'"
    print(f"  ✓ baseline_period = '{ctx.baseline_period}' (WMO standard)")


def test_z_within_normal_range_produces_normal_classification():
    """
    CLAUDE.md non-negotiable: Z within ±1.5 must produce anomaly_classification='normal'.
    Tests observed = baseline_mean (Z=0) and observed = baseline_mean ± 1.5σ.
    Baseline: mean=20.0, stddev=2.0 → ±1.5σ = ±3.0°C.
    """
    for observed_temp, label in [
        (20.0, "Z=0.0"),
        (23.0, "Z=+1.5"),
        (17.0, "Z=-1.5"),
    ]:
        obs = _make_observation(temp_mean_c=observed_temp)
        ctx = compute_climate_context(obs, _canonical_baseline())

        assert ctx.anomaly_classification == "normal", (
            f"[{label}] Expected 'normal', got '{ctx.anomaly_classification}'. "
            "CLAUDE.md requires Z ≤ ±1.5 to yield 'normal'."
        )
        print(f"  ✓ [{label}] anomaly_classification='normal' (graceful degradation rule)")


def test_normal_z_includes_variability_note_in_confidence_note():
    """
    When Z ≤ ±1.5, the confidence_note must contain a variability note instructing
    downstream consumers not to generate an anomaly narrative.
    """
    obs = _make_observation(temp_mean_c=20.0)   # Z = 0.0
    ctx = compute_climate_context(obs, _canonical_baseline())

    assert "normal climate variability" in ctx.confidence_note.lower() or \
           "variability" in ctx.confidence_note.lower(), (
        f"confidence_note must mention variability for Z≤1.5. "
        f"Got: '{ctx.confidence_note}'"
    )
    print(f"  ✓ Z=0.0 → confidence_note contains variability guidance")


def test_notable_anomaly_classification():
    """
    Observed 2.5 stddevs above baseline mean (Z=+2.5) → 'notable'.
    Baseline: mean=20, stddev=2 → observed=25 → Z=2.5.
    """
    obs = _make_observation(temp_mean_c=25.0)   # Z = (25-20)/2 = 2.5
    ctx = compute_climate_context(obs, _canonical_baseline())

    assert ctx.anomaly_classification == "notable", \
        f"Expected 'notable' for Z=2.5, got '{ctx.anomaly_classification}'"
    assert abs(ctx.z_score - 2.5) < 0.01, \
        f"Expected Z≈2.5, got {ctx.z_score}"
    print(f"  ✓ Z=2.5 → 'notable' (z_score={ctx.z_score})")


def test_exceptional_anomaly_classification():
    """
    Observed 3.5 stddevs above baseline mean (Z=+3.5) → 'exceptional'.
    Baseline: mean=20, stddev=2 → observed=27 → Z=3.5.
    """
    obs = _make_observation(temp_mean_c=27.0)   # Z = (27-20)/2 = 3.5
    ctx = compute_climate_context(obs, _canonical_baseline())

    assert ctx.anomaly_classification == "exceptional", \
        f"Expected 'exceptional' for Z=3.5, got '{ctx.anomaly_classification}'"
    print(f"  ✓ Z=3.5 → 'exceptional' (z_score={ctx.z_score})")


def test_cdd_delta_sign_hot_anomaly():
    """
    Hot observed period → CDD_observed > CDD_baseline → CDD_delta > 0.
    Baseline temps [19, 20, 21]°C (30 days each) → mean CDD_base = 60.
    Observed: 30°C × 30 days → CDD_obs = 360.
    """
    obs = _make_observation(temp_mean_c=30.0, start="2023-06-01", end="2023-06-30")
    # Three years with different temps to ensure stdev > 0
    baseline = _make_baseline({1991: 19.0, 1993: 20.0, 1995: 21.0}, days_per_year=30)
    ctx = compute_climate_context(obs, baseline)

    assert ctx.cdd_delta > 0, f"Hot anomaly should have positive CDD_delta, got {ctx.cdd_delta}"
    assert ctx.hdd_observed == 0.0, "No heating needed at 30°C"
    print(f"  ✓ Hot anomaly: CDD_delta={ctx.cdd_delta:.1f} > 0")


def test_hdd_delta_sign_cold_anomaly():
    """
    Cold observed period → HDD_observed > HDD_baseline → HDD_delta > 0.
    Baseline temps [9, 10, 11]°C (31 days each) → mean HDD_base = 248.
    Observed: 5°C × 31 days → HDD_obs = (18-5)*31 = 403.
    """
    obs = _make_observation(temp_mean_c=5.0, start="2023-01-01", end="2023-01-31")
    # Three years with different temps to ensure stdev > 0
    baseline = _make_baseline(
        {1991: 9.0, 1993: 10.0, 1995: 11.0},
        month=1, days_per_year=31,
    )
    ctx = compute_climate_context(obs, baseline)

    assert ctx.hdd_delta > 0, f"Cold anomaly should have positive HDD_delta, got {ctx.hdd_delta}"
    assert ctx.cdd_observed == 0.0, "No cooling needed at 5°C"
    print(f"  ✓ Cold anomaly: HDD_delta={ctx.hdd_delta:.1f} > 0")


def test_cdd_delta_is_observed_minus_baseline():
    """CDD_delta == CDD_observed - CDD_baseline (arithmetic consistency check)."""
    obs = _make_observation(temp_mean_c=25.0, start="2023-06-01", end="2023-06-30")
    baseline = _make_baseline({1991: 19.0, 1993: 20.0, 1995: 21.0}, days_per_year=30)
    ctx = compute_climate_context(obs, baseline)

    expected_delta = round(ctx.cdd_observed - ctx.cdd_baseline, 1)
    assert abs(ctx.cdd_delta - expected_delta) < 1e-6, (
        f"CDD_delta ({ctx.cdd_delta}) ≠ CDD_obs - CDD_base "
        f"({ctx.cdd_observed} - {ctx.cdd_baseline} = {expected_delta})"
    )
    print(f"  ✓ CDD arithmetic: {ctx.cdd_observed:.1f} - {ctx.cdd_baseline:.1f} = {ctx.cdd_delta:.1f}")


def test_wet_bulb_populated_when_dewpoint_available():
    """When dewpoint_mean_c ≠ 0.0, wet_bulb_temp_c must be non-zero and ≤ observed T."""
    obs = _make_observation(temp_mean_c=28.0, dewpoint_c=18.0)
    ctx = compute_climate_context(obs, _canonical_baseline())

    assert ctx.wet_bulb_temp_c != 0.0, "Wet-bulb must be computed when dewpoint is available"
    assert ctx.wet_bulb_temp_c <= obs.observed_temp_mean_c, (
        f"Tw ({ctx.wet_bulb_temp_c}°C) must be ≤ T ({obs.observed_temp_mean_c}°C)"
    )
    print(f"  ✓ Wet-bulb from dewpoint: Tw={ctx.wet_bulb_temp_c}°C ≤ T={obs.observed_temp_mean_c}°C")


def test_confidence_degrades_to_low_with_high_null_fraction():
    """
    CLAUDE.md rule: >15% null baseline values → confidence='low'.
    Build a baseline where 20% of records have None temperature.
    """
    # 10 records total, 2 are None (20% null)
    # Alternate 19/21°C so stdev > 0 (avoids the zero-stdev guard)
    records = [
        {"date": f"199{i}-06-01",
         "temp_mean_c": 19.0 if i % 2 == 0 else 21.0,
         "precip_sum_mm": 2.0,
         "dewpoint_mean_c": None}
        for i in range(1, 9)
    ]
    records += [
        {"date": "1999-06-01", "temp_mean_c": None, "precip_sum_mm": None, "dewpoint_mean_c": None},
        {"date": "2000-06-01", "temp_mean_c": None, "precip_sum_mm": None, "dewpoint_mean_c": None},
    ]

    obs = _make_observation()
    ctx = compute_climate_context(obs, records)

    assert ctx.confidence == "low", \
        f"Expected confidence='low' with 20% null baseline, got '{ctx.confidence}'"
    print(f"  ✓ 20% null baseline → confidence='{ctx.confidence}'")


def test_confidence_degrades_to_medium_with_partial_observation():
    """
    Partial observation quality (data_quality_flag='partial') → confidence='medium'.
    """
    obs = _make_observation(quality="partial")
    ctx = compute_climate_context(obs, _canonical_baseline())

    assert ctx.confidence == "medium", \
        f"Expected confidence='medium' for partial observation, got '{ctx.confidence}'"
    print(f"  ✓ Partial observation → confidence='{ctx.confidence}'")


def test_zero_baseline_stddev_raises_value_error():
    """
    All baseline records at the same temperature → stddev = 0 → ValueError.
    Z-score is undefined when there is no natural variability.
    """
    flat_baseline = _make_baseline({1991: 20.0, 1993: 20.0, 1995: 20.0})
    obs = _make_observation()

    raised = False
    try:
        compute_climate_context(obs, flat_baseline)
    except ValueError as exc:
        raised = True
        print(f"  ✓ Zero stddev → ValueError: '{exc}'")

    assert raised, "Expected ValueError for zero baseline stddev, none raised."


def test_insufficient_baseline_raises_value_error():
    """
    Fewer than MIN_BASELINE_RECORDS non-null records → ValueError.
    """
    tiny_baseline = [
        {"date": "2015-06-01", "temp_mean_c": 20.0, "precip_sum_mm": 2.0, "dewpoint_mean_c": None},
        {"date": "2016-06-01", "temp_mean_c": 21.0, "precip_sum_mm": 2.0, "dewpoint_mean_c": None},
    ]
    raised = False
    try:
        compute_climate_context(_make_observation(), tiny_baseline)
    except ValueError as exc:
        raised = True
        print(f"  ✓ Insufficient baseline → ValueError: '{exc}'")

    assert raised, "Expected ValueError for insufficient baseline records, none raised."


def test_trend_slope_in_context_reflects_warming():
    """
    A warming baseline (2011–2020) must produce a positive trend slope
    in the returned ClimateContext.
    """
    # Mix of flat early years and a clear warming trend in the 10-year window
    flat_early = {yr: 20.0 for yr in range(1991, 2011)}
    warming_window = {yr: 20.0 + 0.2 * (yr - 2011) for yr in range(2011, 2021)}
    baseline = _make_baseline({**flat_early, **warming_window})

    ctx = compute_climate_context(_make_observation(temp_mean_c=25.0), baseline)

    assert ctx.trend_slope_c_per_decade > 0, \
        f"Warming baseline should produce positive trend, got {ctx.trend_slope_c_per_decade}"
    assert abs(ctx.trend_slope_c_per_decade - 2.0) < 0.01, \
        f"Expected slope≈2.0°C/decade, got {ctx.trend_slope_c_per_decade}"
    print(
        f"  ✓ Warming trend in context: slope={ctx.trend_slope_c_per_decade:.3f}°C/decade"
    )


# ---------------------------------------------------------------------------
# Plain-Python Runner (for use without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Z-score unit tests
        test_z_score_zero_at_baseline_mean,
        test_z_score_positive_warm_anomaly,
        test_z_score_negative_cold_anomaly,
        # Anomaly classification (CLAUDE.md thresholds)
        test_classify_z_zero_is_normal,
        test_classify_z_at_boundary_1_5_is_normal,
        test_classify_z_just_above_1_5_is_notable,
        test_classify_z_mid_range_notable,
        test_classify_z_at_boundary_3_0_is_notable,
        test_classify_z_just_above_3_0_is_exceptional,
        test_classify_z_large_positive_exceptional,
        # Degree day unit tests
        test_cdd_hot_period,
        test_hdd_cold_period,
        test_cdd_hdd_zero_at_base_temperature,
        test_baseline_cdd_computed_from_daily_records,
        test_baseline_hdd_cold_records,
        # Wet-bulb unit tests (Stull 2011)
        test_wet_bulb_is_below_dry_bulb,
        test_wet_bulb_approaches_dry_bulb_at_saturation,
        test_wet_bulb_reference_value,
        test_wet_bulb_is_above_dewpoint,
        test_wet_bulb_unavailable_when_dewpoint_is_zero,
        test_rh_from_dewpoint_known_value,
        # Trend slope unit tests
        test_warming_trend_positive_slope,
        test_cooling_trend_negative_slope,
        test_flat_trend_near_zero_slope,
        test_trend_uses_only_last_10_years,
        test_trend_with_insufficient_years_returns_zero,
        # Integration tests
        test_full_context_schema_types,
        test_baseline_period_label_is_wmo_standard,
        test_z_within_normal_range_produces_normal_classification,
        test_normal_z_includes_variability_note_in_confidence_note,
        test_notable_anomaly_classification,
        test_exceptional_anomaly_classification,
        test_cdd_delta_sign_hot_anomaly,
        test_hdd_delta_sign_cold_anomaly,
        test_cdd_delta_is_observed_minus_baseline,
        test_wet_bulb_populated_when_dewpoint_available,
        test_confidence_degrades_to_low_with_high_null_fraction,
        test_confidence_degrades_to_medium_with_partial_observation,
        test_zero_baseline_stddev_raises_value_error,
        test_insufficient_baseline_raises_value_error,
        test_trend_slope_in_context_reflects_warming,
    ]

    print("\n── Climate Statistics Engine Validation ─────────────────\n")
    passed = 0
    failed = 0

    for test_fn in tests:
        label = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {label} FAILED: {exc}")
            failed += 1

    print(f"\n── Results: {passed} passed, {failed} failed ─────────────────\n")

    if failed > 0:
        sys.exit(1)
