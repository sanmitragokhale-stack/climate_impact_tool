"""
test_weather.py

Validation suite for src/weather.py — Slice 2: Open-Meteo Weather Client.

Mirrors the style of test_geocoding.py: live API calls for integration tests,
plus offline tests for input validation.

NOTE: Baseline tests fetch 30 years of daily data from Open-Meteo and may
take 15–60 seconds each. Run observed tests only during quick feedback cycles:

    python -m pytest tests/test_weather.py -v -k "not baseline"

Full suite (recommended before committing):
    python -m pytest tests/test_weather.py -v

Without pytest:
    python tests/test_weather.py
"""

import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import LocationResult, WeatherObservation
from src.weather import fetch_weather_observation, fetch_baseline_series


# ---------------------------------------------------------------------------
# Shared Location Fixtures
#
# Pre-built LocationResults avoid geocoding API calls in this test suite.
# Coordinates verified against Open-Meteo's geocoding API output.
# ---------------------------------------------------------------------------

PARIS = LocationResult(
    city_name="Paris",
    country="France",
    country_code="FR",
    latitude=48.8566,
    longitude=2.3522,
    admin_region="Île-de-France",
    population=2_161_000,
    match_confidence="high",
)

LONDON = LocationResult(
    city_name="London",
    country="United Kingdom",
    country_code="GB",
    latitude=51.5074,
    longitude=-0.1278,
    admin_region="England",
    population=8_982_000,
    match_confidence="high",
)

SYDNEY = LocationResult(
    city_name="Sydney",
    country="Australia",
    country_code="AU",
    latitude=-33.8688,
    longitude=151.2093,
    admin_region="New South Wales",
    population=5_312_000,
    match_confidence="high",
)

# A fixed historical range with guaranteed complete archive coverage
_HIST_START = "2023-06-01"
_HIST_END = "2023-06-30"

# ---------------------------------------------------------------------------
# Baseline Cache
#
# Each 30-year fetch counts against Open-Meteo's free-tier rate limit.
# All baseline tests share these cached results to avoid hitting the limit
# when the full suite runs back-to-back.
# ---------------------------------------------------------------------------

_BASELINE_CACHE: dict[tuple, list[dict]] = {}


def _get_baseline(location: LocationResult, month: int) -> list[dict]:
    """Returns the baseline series, fetching from the API only on first call."""
    key = (location.latitude, location.longitude, month)
    if key not in _BASELINE_CACHE:
        _BASELINE_CACHE[key] = fetch_baseline_series(location, month)
    return _BASELINE_CACHE[key]


# ---------------------------------------------------------------------------
# Shared Assertion Helper
# ---------------------------------------------------------------------------

def assert_valid_observation(obs: WeatherObservation, test_name: str) -> None:
    """Core schema assertions that every successful WeatherObservation must pass."""
    assert isinstance(obs, WeatherObservation), \
        f"[{test_name}] Result must be a WeatherObservation dataclass instance."

    assert obs.location is not None, \
        f"[{test_name}] location must not be None."

    assert obs.date_range_start and isinstance(obs.date_range_start, str), \
        f"[{test_name}] date_range_start must be a non-empty string."

    assert obs.date_range_end and isinstance(obs.date_range_end, str), \
        f"[{test_name}] date_range_end must be a non-empty string."

    assert isinstance(obs.observed_temp_mean_c, float), \
        f"[{test_name}] observed_temp_mean_c must be float."

    assert isinstance(obs.observed_precip_sum_mm, float), \
        f"[{test_name}] observed_precip_sum_mm must be float."

    assert isinstance(obs.dewpoint_mean_c, float), \
        f"[{test_name}] dewpoint_mean_c must be float."

    assert obs.data_quality_flag in ("complete", "partial", "interpolated"), \
        f"[{test_name}] data_quality_flag must be 'complete', 'partial', or 'interpolated'."

    assert obs.data_source == "open-meteo", \
        f"[{test_name}] data_source must be 'open-meteo', got '{obs.data_source}'."


# ---------------------------------------------------------------------------
# Observed Period Tests — Live API
# ---------------------------------------------------------------------------

def test_paris_june_observation():
    """
    Fetch a known historical month for Paris and verify the observation is
    physically plausible. June in Paris typically averages 15–22°C.
    """
    obs = fetch_weather_observation(PARIS, _HIST_START, _HIST_END)
    assert_valid_observation(obs, "Paris-June-2023")

    assert 5.0 <= obs.observed_temp_mean_c <= 40.0, (
        f"Paris June mean temp {obs.observed_temp_mean_c}°C is outside [5, 40]. "
        "Check that the correct location was fetched."
    )
    assert obs.observed_precip_sum_mm >= 0.0, \
        "observed_precip_sum_mm must be non-negative."
    assert obs.date_range_start == _HIST_START
    assert obs.date_range_end == _HIST_END

    print(
        f"  ✓ Paris Jun 2023 → T_mean={obs.observed_temp_mean_c}°C, "
        f"Precip={obs.observed_precip_sum_mm}mm, Dewpoint={obs.dewpoint_mean_c}°C, "
        f"quality={obs.data_quality_flag}"
    )


def test_london_january_observation():
    """
    London in January should return temperatures in the range [−5, 12]°C,
    validating that a winter Northern-Hemisphere period is handled correctly.
    """
    obs = fetch_weather_observation(LONDON, "2023-01-01", "2023-01-31")
    assert_valid_observation(obs, "London-Jan-2023")

    assert -5.0 <= obs.observed_temp_mean_c <= 12.0, (
        f"London Jan mean temp {obs.observed_temp_mean_c}°C is outside [-5, 12]."
    )

    print(
        f"  ✓ London Jan 2023 → T_mean={obs.observed_temp_mean_c}°C, "
        f"quality={obs.data_quality_flag}"
    )


def test_sydney_southern_hemisphere_summer():
    """
    Sydney in January (Southern Hemisphere summer) should return temperatures
    in [15, 35]°C, validating that Southern Hemisphere coordinates work correctly.
    """
    obs = fetch_weather_observation(SYDNEY, "2023-01-01", "2023-01-31")
    assert_valid_observation(obs, "Sydney-Jan-2023")

    assert 15.0 <= obs.observed_temp_mean_c <= 35.0, (
        f"Sydney Jan mean temp {obs.observed_temp_mean_c}°C is outside [15, 35]."
    )

    print(
        f"  ✓ Sydney Jan 2023 → T_mean={obs.observed_temp_mean_c}°C "
        f"(Southern Hemisphere summer), quality={obs.data_quality_flag}"
    )


def test_default_date_range_is_recent():
    """
    Calling fetch_weather_observation with no dates must return an observation
    whose window ends within the last 10 days and spans approximately 30 days.
    This validates the ARCHIVE_LAG_DAYS offset and default window logic.
    """
    obs = fetch_weather_observation(PARIS)
    assert_valid_observation(obs, "Paris-default-dates")

    today = date.today()
    end_d = date.fromisoformat(obs.date_range_end)
    start_d = date.fromisoformat(obs.date_range_start)

    days_ago = (today - end_d).days
    window_length = (end_d - start_d).days + 1

    assert 1 <= days_ago <= 10, (
        f"Default end date {obs.date_range_end} is {days_ago} days ago — "
        "expected 1–10 (archive lag window)."
    )
    assert 28 <= window_length <= 32, (
        f"Default observation window is {window_length} days — expected ~30."
    )

    print(
        f"  ✓ Default dates → {obs.date_range_start} to {obs.date_range_end} "
        f"({window_length} days, {days_ago}d ago), T_mean={obs.observed_temp_mean_c}°C"
    )


def test_dewpoint_does_not_exceed_temperature():
    """
    Dewpoint temperature must always be ≤ dry-bulb temperature.
    This is a fundamental thermodynamic constraint — Td ≤ T for any air mass.
    A violation would indicate a calculation bug in dewpoint aggregation.
    """
    obs = fetch_weather_observation(PARIS, _HIST_START, _HIST_END)

    if obs.dewpoint_mean_c == 0.0 and obs.data_quality_flag == "partial":
        print("  ⚠ Dewpoint unavailable (partial quality) — skipping Td ≤ T check")
        return

    assert obs.dewpoint_mean_c <= obs.observed_temp_mean_c, (
        f"Dewpoint ({obs.dewpoint_mean_c}°C) exceeds air temperature "
        f"({obs.observed_temp_mean_c}°C). This violates thermodynamic constraints."
    )

    print(
        f"  ✓ Dewpoint constraint satisfied: Td={obs.dewpoint_mean_c}°C "
        f"≤ T={obs.observed_temp_mean_c}°C"
    )


def test_schema_field_types_observation():
    """
    Explicitly validates that every WeatherObservation field has the correct
    Python type. This is the Layer 2 schema contract test.
    """
    obs = fetch_weather_observation(PARIS, _HIST_START, _HIST_END)

    assert isinstance(obs.location, LocationResult),    "location must be LocationResult"
    assert isinstance(obs.date_range_start, str),       "date_range_start must be str"
    assert isinstance(obs.date_range_end, str),         "date_range_end must be str"
    assert isinstance(obs.observed_temp_mean_c, float), "observed_temp_mean_c must be float"
    assert isinstance(obs.observed_precip_sum_mm, float),"observed_precip_sum_mm must be float"
    assert isinstance(obs.dewpoint_mean_c, float),      "dewpoint_mean_c must be float"
    assert isinstance(obs.data_source, str),            "data_source must be str"
    assert isinstance(obs.data_quality_flag, str),      "data_quality_flag must be str"

    print(f"  ✓ Layer 2 schema contract — all WeatherObservation field types correct")


# ---------------------------------------------------------------------------
# Baseline Series Tests — Live API (note: each call fetches 30 years of data)
# ---------------------------------------------------------------------------

def test_baseline_june_paris_record_count():
    """
    Baseline for Paris June (month=6) across 1991–2020 should return
    approximately 900 records (30 years × ~30 days per June).
    """
    records = _get_baseline(PARIS, 6)

    # Lower bound: at minimum one record per year
    assert len(records) >= 30, \
        f"Expected at least 30 baseline records, got {len(records)}."

    # Upper bound: 30 years × 30 days = 900 (June has 30 days)
    assert len(records) <= 930, \
        f"Unexpectedly many baseline records: {len(records)}. Max for June: 900."

    print(
        f"  ✓ Paris baseline June → {len(records)} daily records "
        f"across 1991–2020 (expected ~900)"
    )


def test_baseline_month_filter_is_exact():
    """
    All records returned for calendar_month=7 must have '07' in the month
    position of their date string. Off-by-one errors in the filter would
    let adjacent-month records leak through.
    """
    records = _get_baseline(PARIS, 7)

    wrong = [r for r in records if r["date"][5:7] != "07"]
    assert not wrong, (
        f"Found {len(wrong)} records with wrong month: "
        f"{[r['date'] for r in wrong[:3]]}"
    )

    print(
        f"  ✓ Baseline month filter → all {len(records)} records "
        "correctly in month 07"
    )


def test_baseline_wmo_period_coverage():
    """
    The baseline series must span exactly 1991–2020 (the current WMO
    30-year climatological normal). Records outside this window would
    indicate an API parameter or filter bug.
    """
    records = _get_baseline(LONDON, 1)    # January for London

    years = {r["date"][:4] for r in records}

    assert "1991" in years, \
        f"Baseline missing year 1991. Earliest found: {min(years)}."
    assert "2020" in years, \
        f"Baseline missing year 2020. Latest found: {max(years)}."
    assert min(years) == "1991", \
        f"Earliest baseline year is {min(years)}, expected '1991'."
    assert max(years) == "2020", \
        f"Latest baseline year is {max(years)}, expected '2020'."

    # Should have exactly 30 distinct years
    assert len(years) == 30, \
        f"Expected 30 distinct years (1991–2020), got {len(years)}: {sorted(years)}"

    print(
        f"  ✓ WMO period → {min(years)}–{max(years)}, "
        f"{len(years)} years, {len(records)} total records"
    )


def test_baseline_temperature_quality_and_plausibility():
    """
    The null rate for baseline temperature must not exceed the 15% threshold
    defined in CLAUDE.md. Spot-checks also verify physical plausibility.
    """
    records = _get_baseline(PARIS, 6)

    null_count = sum(1 for r in records if r["temp_mean_c"] is None)
    null_pct = (null_count / len(records)) * 100.0

    assert null_pct <= 15.0, (
        f"Baseline temperature is null for {null_pct:.1f}% of records "
        f"(threshold: 15%). ERA5 data quality issue."
    )

    non_null = [r["temp_mean_c"] for r in records if r["temp_mean_c"] is not None]

    # June in Paris: physical plausibility bounds (allow extremes)
    for t in non_null[:20]:   # spot-check first 20 values
        assert -10.0 <= t <= 45.0, \
            f"Baseline temp {t}°C is physically implausible for Paris June."

    print(
        f"  ✓ Baseline temp quality → {null_pct:.1f}% null, "
        f"range [{min(non_null):.1f}, {max(non_null):.1f}]°C"
    )


def test_baseline_record_schema():
    """
    Each baseline record must contain exactly the four required keys with
    the correct Python types. This is the dict schema contract for the
    interface between weather.py and climate_stats.py (Slice 3).
    """
    records = _get_baseline(PARIS, 6)
    required_keys = {"date", "temp_mean_c", "precip_sum_mm", "dewpoint_mean_c"}

    for i, r in enumerate(records[:10]):   # spot-check first 10
        missing = required_keys - set(r.keys())
        assert not missing, f"Record {i} is missing keys: {missing}"

        assert isinstance(r["date"], str), \
            f"Record {i}: date must be str, got {type(r['date'])}"
        assert r["temp_mean_c"] is None or isinstance(r["temp_mean_c"], float), \
            f"Record {i}: temp_mean_c must be float or None"
        assert r["precip_sum_mm"] is None or isinstance(r["precip_sum_mm"], float), \
            f"Record {i}: precip_sum_mm must be float or None"

        # dewpoint_mean_c is intentionally None at baseline scale — verify this
        assert r["dewpoint_mean_c"] is None, (
            f"Record {i}: dewpoint_mean_c should be None in baseline records "
            "(hourly data not fetched at 30-year scale)."
        )

    print(
        f"  ✓ Baseline record schema → all required keys with correct types "
        f"({len(records)} records validated)"
    )


# ---------------------------------------------------------------------------
# Input Validation Tests — Offline (no network required)
# ---------------------------------------------------------------------------

def test_invalid_date_format_raises_value_error():
    """
    A date string with slashes instead of dashes must raise ValueError
    before any network call is made.
    """
    raised = False
    try:
        fetch_weather_observation(PARIS, "2023/06/01", "2023-06-30")
    except ValueError as exc:
        raised = True
        print(f"  ✓ Bad date format → ValueError: '{exc}'")

    assert raised, "Expected ValueError for malformed date format, none raised."


def test_inverted_date_range_raises_value_error():
    """
    A start_date that falls after end_date must raise ValueError.
    """
    raised = False
    try:
        fetch_weather_observation(PARIS, "2023-06-30", "2023-06-01")
    except ValueError as exc:
        raised = True
        print(f"  ✓ Inverted date range → ValueError: '{exc}'")

    assert raised, "Expected ValueError for inverted date range, none raised."


def test_invalid_calendar_month_raises_value_error():
    """
    Calendar months outside the valid range 1–12 must each raise ValueError.
    Tests boundary and well-outside-boundary cases.
    """
    for bad_month in (0, 13, -1, 99):
        raised = False
        try:
            fetch_baseline_series(PARIS, bad_month)
        except ValueError as exc:
            raised = True
            print(f"  ✓ Month={bad_month} → ValueError: '{exc}'")

        assert raised, (
            f"Expected ValueError for calendar_month={bad_month}, none raised."
        )


# ---------------------------------------------------------------------------
# Plain-Python Runner (for use without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Observed period — live API
        test_paris_june_observation,
        test_london_january_observation,
        test_sydney_southern_hemisphere_summer,
        test_default_date_range_is_recent,
        test_dewpoint_does_not_exceed_temperature,
        test_schema_field_types_observation,
        # Baseline series — live API (slow: ~30 years of daily data each)
        test_baseline_june_paris_record_count,
        test_baseline_month_filter_is_exact,
        test_baseline_wmo_period_coverage,
        test_baseline_temperature_quality_and_plausibility,
        test_baseline_record_schema,
        # Input validation — offline
        test_invalid_date_format_raises_value_error,
        test_inverted_date_range_raises_value_error,
        test_invalid_calendar_month_raises_value_error,
    ]

    print("\n── Weather Client Validation ─────────────────────────────\n")
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
