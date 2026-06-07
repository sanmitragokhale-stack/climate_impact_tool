"""
geocoding.py

Location Engine for the Climate Impact Portfolio Tool.
Resolves free-text city input to a validated LocationResult using the
Open-Meteo Geocoding API (free, no authentication required).

API docs: https://open-meteo.com/en/docs/geocoding-api

This module is self-contained. It has no dependency on any other
src/ module except schema.py.
"""

import requests
from src.schema import LocationResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_TIMEOUT_SECONDS = 10
MAX_RESULTS = 5   # Fetch top-5 candidates; population-based selection picks best


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def geocode_city(city_input: str) -> LocationResult:
    """
    Takes a free-text city name and returns a validated LocationResult.

    Accepts optional country hint: "London, UK" or "London, Ontario".
    When a country hint is provided results are filtered to that country first.
    The best match (exact name + highest population) is returned.

    Raises:
        ValueError: If the input is blank or no city is found.
        ConnectionError: If the geocoding API is unreachable or returns an error.
    """
    candidates = geocode_city_candidates(city_input)
    if not candidates:
        city_name = city_input.split(",")[0].strip()
        raise ValueError(
            f"No geocoding results found for '{city_name}'. "
            "Try a different spelling or a nearby major city."
        )
    return candidates[0]


def geocode_city_candidates(city_input: str) -> list[LocationResult]:
    """
    Returns up to MAX_RESULTS candidate LocationResults, sorted by:
      1. Exact city-name matches before fuzzy matches.
      2. Highest population within each group.

    Accepts optional country hint ("London, UK", "London, Ontario").
    When provided, results are filtered to match that country or region first;
    if the filter eliminates all candidates the full set is used as fallback.

    Args:
        city_input: Free-text city name, optionally with country hint after a comma.

    Returns:
        Sorted list of LocationResult (may be empty on no match).

    Raises:
        ValueError: If the input is blank or the API returns no results.
        ConnectionError: If the geocoding API is unreachable.
    """
    city_input = _sanitize_input(city_input)
    city_name, country_hint = _parse_city_country(city_input)

    raw_results = _fetch_geocoding_results(city_name)

    # Apply country/region hint filter if provided
    if country_hint:
        hint_lower = country_hint.lower()
        filtered = [
            r for r in raw_results
            if hint_lower in r.get("country", "").lower()
            or hint_lower in r.get("country_code", "").lower()
            or hint_lower in r.get("admin1", "").lower()
        ]
        if filtered:
            raw_results = filtered

    # Sort: exact city name first, then by population descending
    name_lower = city_name.lower()
    exact = [r for r in raw_results if r.get("name", "").lower() == name_lower]
    rest = [r for r in raw_results if r.get("name", "").lower() != name_lower]
    exact.sort(key=lambda r: r.get("population", 0), reverse=True)
    rest.sort(key=lambda r: r.get("population", 0), reverse=True)

    candidates: list[LocationResult] = []
    for match in exact + rest:
        try:
            candidates.append(_build_location_result(match, city_name))
        except ValueError:
            continue

    return candidates


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _parse_city_country(city_input: str) -> tuple[str, str]:
    """
    Splits 'City, Country' into (city_name, country_hint).
    Returns (city_input, '') when no comma is present.
    """
    if "," in city_input:
        parts = city_input.split(",", 1)
        return parts[0].strip(), parts[1].strip()
    return city_input, ""


def _sanitize_input(city_input: str) -> str:
    """
    Strip surrounding whitespace and reject blank input early.

    Raises:
        ValueError: If the cleaned string is empty.
    """
    cleaned = city_input.strip()
    if not cleaned:
        raise ValueError("City input cannot be blank.")
    return cleaned


def _fetch_geocoding_results(city_name: str) -> list[dict]:
    """
    Calls the Open-Meteo Geocoding API and returns the raw results list.

    The API returns JSON with a top-level "results" key containing an array.
    If the key is absent, the city was not found.

    Raises:
        ConnectionError: On any network failure or non-200 HTTP response.
        ValueError: If the API returns no matching locations.
    """
    params = {
        "name": city_name,
        "count": MAX_RESULTS,
        "language": "en",
        "format": "json",
    }

    try:
        response = requests.get(
            GEOCODING_API_URL,
            params=params,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Could not reach the geocoding API. Check your internet connection. "
            f"Detail: {exc}"
        ) from exc

    except requests.exceptions.Timeout:
        raise ConnectionError(
            f"Geocoding API timed out after {DEFAULT_TIMEOUT_SECONDS}s. "
            "Try again later."
        )

    except requests.exceptions.HTTPError as exc:
        raise ConnectionError(
            f"Geocoding API returned an HTTP error: {exc}"
        ) from exc

    data = response.json()

    if "results" not in data or not data["results"]:
        raise ValueError(
            f"No geocoding results found for '{city_name}'. "
            "Try a different spelling or a nearby major city."
        )

    return data["results"]


def _select_best_match(results: list[dict], original_query: str) -> dict:
    """
    Selects the single best candidate from a list of geocoding results.

    Selection logic (priority order):
      1. Prefer exact case-insensitive name matches.
      2. Within that set (or the full set if no exact match), sort by
         population descending — highest population = most likely user intent.

    This cleanly handles common disambiguation cases:
      "Paris"       → Paris, France (pop ~2M) not Paris, Texas (pop ~25k)
      "Springfield" → the most populous Springfield in the results

    Args:
        results: Raw list of result dicts from the geocoding API.
        original_query: The sanitised user input string.

    Returns:
        The single best-matching result dict.
    """
    query_lower = original_query.lower()

    exact_matches = [
        r for r in results
        if r.get("name", "").lower() == query_lower
    ]

    # Use exact matches if any found, otherwise fall back to full result set
    candidates = exact_matches if exact_matches else results

    # Sort by population descending; treat missing population as 0
    candidates.sort(key=lambda r: r.get("population", 0), reverse=True)

    return candidates[0]


def _build_location_result(match: dict, original_query: str) -> LocationResult:
    """
    Constructs a validated LocationResult from a raw API response dict.

    Confidence assignment logic:
      "high"   — exact name match AND population data present
      "medium" — exact name match BUT no population data
      "low"    — no exact name match (API returned a fuzzy/nearby result)

    Args:
        match: The selected best-match dict from the geocoding API.
        original_query: The sanitised user input, used for confidence comparison.

    Returns:
        A fully populated LocationResult.

    Raises:
        ValueError: If the result is missing latitude or longitude.
    """
    lat = match.get("latitude")
    lon = match.get("longitude")

    # Coordinates are non-negotiable — every downstream module depends on them
    if lat is None or lon is None:
        raise ValueError(
            f"Geocoding result for '{original_query}' is missing coordinates. "
            "This is an API data quality issue. Try a nearby major city."
        )

    canonical_name = match.get("name", original_query)
    is_exact_match = canonical_name.lower() == original_query.lower()
    has_population = match.get("population") is not None

    # Assign confidence tier
    if is_exact_match and has_population:
        confidence = "high"
        note = ""
    elif is_exact_match:
        confidence = "medium"
        note = "Exact name match, but population data unavailable for disambiguation."
    else:
        confidence = "low"
        note = (
            f"Input '{original_query}' did not exactly match returned name "
            f"'{canonical_name}'. Verify this is the intended location."
        )

    return LocationResult(
        city_name=canonical_name,
        country=match.get("country", "Unknown"),
        country_code=match.get("country_code", "XX"),
        latitude=float(lat),
        longitude=float(lon),
        admin_region=match.get("admin1", ""),   # State / province / region
        population=match.get("population"),
        match_confidence=confidence,
        match_note=note,
    )
