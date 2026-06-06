"""
test_geocoding.py

Standalone validation suite for src/geocoding.py — Slice 1: Location Engine.

Compatible with both pytest and plain Python:

    # Recommended (coloured output, verbose):
    python -m pytest tests/test_geocoding.py -v

    # Without pytest:
    python tests/test_geocoding.py
"""

import sys
import os

# Ensure the project root is on the path so 'src' is importable
# when running from any working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geocoding import geocode_city
from src.schema import LocationResult


# ---------------------------------------------------------------------------
# Shared Assertion Helper
# ---------------------------------------------------------------------------

def assert_valid_location(result: LocationResult, test_name: str) -> None:
    """
    Core schema assertions that every successful LocationResult must pass.
    Called at the start of every positive test case.
    """
    assert isinstance(result, LocationResult), \
        f"[{test_name}] Result must be a LocationResult dataclass instance."

    assert result.city_name and isinstance(result.city_name, str), \
        f"[{test_name}] city_name must be a non-empty string."

    assert isinstance(result.latitude, float), \
        f"[{test_name}] latitude must be a float."

    assert isinstance(result.longitude, float), \
        f"[{test_name}] longitude must be a float."

    assert -90.0 <= result.latitude <= 90.0, \
        f"[{test_name}] Latitude {result.latitude} is outside the valid range [-90, 90]."

    assert -180.0 <= result.longitude <= 180.0, \
        f"[{test_name}] Longitude {result.longitude} is outside the valid range [-180, 180]."

    assert result.country and isinstance(result.country, str), \
        f"[{test_name}] country must be a non-empty string."

    assert result.country_code and isinstance(result.country_code, str), \
        f"[{test_name}] country_code must be a non-empty string."

    assert result.match_confidence in ("high", "medium", "low"), \
        f"[{test_name}] match_confidence must be 'high', 'medium', or 'low'."

    assert isinstance(result.match_note, str), \
        f"[{test_name}] match_note must be a string (can be empty)."


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_paris_france():
    """
    Paris must resolve to France, not Paris, Texas.
    This validates the population-based disambiguation logic.
    """
    result = geocode_city("Paris")
    assert_valid_location(result, "Paris")

    assert result.country_code == "FR", (
        f"Expected country_code 'FR', got '{result.country_code}'. "
        "Population-based disambiguation has failed."
    )
    # Rough bounding box for Paris, France
    assert 48.0 <= result.latitude <= 49.0, \
        f"Paris latitude {result.latitude:.4f} is outside the expected range [48.0, 49.0]."
    assert 1.5 <= result.longitude <= 3.0, \
        f"Paris longitude {result.longitude:.4f} is outside the expected range [1.5, 3.0]."
    assert result.match_confidence == "high"

    print(
        f"  ✓ 'Paris' → {result.city_name}, {result.country} "
        f"({result.latitude:.4f}, {result.longitude:.4f}) "
        f"[confidence: {result.match_confidence}]"
    )


def test_mumbai_india():
    """Mumbai resolves cleanly with coordinates in the correct range."""
    result = geocode_city("Mumbai")
    assert_valid_location(result, "Mumbai")

    assert result.country_code == "IN", \
        f"Expected country_code 'IN', got '{result.country_code}'."
    assert 18.0 <= result.latitude <= 20.0, \
        f"Mumbai latitude {result.latitude:.4f} is outside the expected range [18.0, 20.0]."
    assert 72.0 <= result.longitude <= 73.5, \
        f"Mumbai longitude {result.longitude:.4f} is outside the expected range [72.0, 73.5]."

    print(
        f"  ✓ 'Mumbai' → {result.city_name}, {result.country} "
        f"({result.latitude:.4f}, {result.longitude:.4f}) "
        f"[confidence: {result.match_confidence}]"
    )


def test_tokyo_japan():
    """Tokyo tests a common non-Latin-script city input in English."""
    result = geocode_city("Tokyo")
    assert_valid_location(result, "Tokyo")

    assert result.country_code == "JP", \
        f"Expected country_code 'JP', got '{result.country_code}'."
    assert 35.0 <= result.latitude <= 36.0, \
        f"Tokyo latitude {result.latitude:.4f} is outside the expected range [35.0, 36.0]."
    assert 138.5 <= result.longitude <= 140.5, \
        f"Tokyo longitude {result.longitude:.4f} is outside the expected range [138.5, 140.5]."

    print(
        f"  ✓ 'Tokyo' → {result.city_name}, {result.country} "
        f"({result.latitude:.4f}, {result.longitude:.4f}) "
        f"[confidence: {result.match_confidence}]"
    )


def test_berlin_germany():
    """Berlin validates a major European city and checks admin_region is populated."""
    result = geocode_city("Berlin")
    assert_valid_location(result, "Berlin")

    assert result.country_code == "DE", \
        f"Expected country_code 'DE', got '{result.country_code}'."
    assert 52.0 <= result.latitude <= 53.0, \
        f"Berlin latitude {result.latitude:.4f} is outside the expected range [52.0, 53.0]."

    print(
        f"  ✓ 'Berlin' → {result.city_name}, {result.country} "
        f"({result.latitude:.4f}, {result.longitude:.4f}) "
        f"[admin_region: '{result.admin_region}', confidence: {result.match_confidence}]"
    )


def test_whitespace_input_is_sanitized():
    """
    Inputs with leading/trailing whitespace must resolve identically
    to the clean version. The sanitiser must strip before querying.
    """
    result = geocode_city("  London  ")
    assert_valid_location(result, "London-whitespace")

    assert result.country_code == "GB", \
        f"Expected 'GB', got '{result.country_code}'."

    print(
        f"  ✓ '  London  ' → resolved cleanly to "
        f"{result.city_name}, {result.country} "
        f"({result.latitude:.4f}, {result.longitude:.4f})"
    )


def test_blank_input_raises_value_error():
    """
    A blank or whitespace-only input must raise a ValueError immediately,
    before any network call is made.
    """
    raised = False
    try:
        geocode_city("   ")
    except ValueError as exc:
        raised = True
        print(f"  ✓ Blank input → raised ValueError as expected: '{exc}'")

    assert raised, "Expected ValueError for blank input but none was raised."


def test_nonexistent_city_raises_value_error():
    """
    A completely invented city name must raise ValueError with a
    helpful message. Tests the 'no results' fallback path.
    """
    raised = False
    try:
        geocode_city("Xqzptlwmburg")
    except ValueError as exc:
        raised = True
        print(f"  ✓ Nonexistent city → raised ValueError as expected: '{exc}'")

    assert raised, "Expected ValueError for nonexistent city but none was raised."


def test_schema_field_types():
    """
    Explicitly validate that all Layer 1 schema fields are present
    and have the correct Python types. This is the schema contract test.
    """
    result = geocode_city("Sydney")
    assert_valid_location(result, "Sydney-schema")

    # Type assertions for every field in LocationResult
    assert isinstance(result.city_name, str),       "city_name must be str"
    assert isinstance(result.country, str),          "country must be str"
    assert isinstance(result.country_code, str),     "country_code must be str"
    assert isinstance(result.latitude, float),       "latitude must be float"
    assert isinstance(result.longitude, float),      "longitude must be float"
    assert isinstance(result.admin_region, str),     "admin_region must be str"
    assert isinstance(result.match_confidence, str), "match_confidence must be str"
    assert isinstance(result.match_note, str),       "match_note must be str"

    # Population is Optional[int] — if present, must be int
    if result.population is not None:
        assert isinstance(result.population, int), \
            f"population must be int or None, got {type(result.population)}"

    print(
        f"  ✓ Sydney schema validation → all field types correct "
        f"(population: {result.population})"
    )


# ---------------------------------------------------------------------------
# Plain-Python Runner (for use without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_paris_france,
        test_mumbai_india,
        test_tokyo_japan,
        test_berlin_germany,
        test_whitespace_input_is_sanitized,
        test_blank_input_raises_value_error,
        test_nonexistent_city_raises_value_error,
        test_schema_field_types,
    ]

    print("\n── Geocoding Engine Validation ──────────────────────────\n")
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_fn.__name__} FAILED: {exc}")
            failed += 1

    print(f"\n── Results: {passed} passed, {failed} failed ────────────────\n")

    if failed > 0:
        sys.exit(1)
