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

    Strategy:
      1. Sanitise and validate the raw input string.
      2. Query Open-Meteo geocoding for up to MAX_RESULTS candidates.
      3. Select best match: exact name + highest population wins.
      4. Validate required coordinate fields are present.
      5. Return a LocationResult with appropriate confidence flags.

    Args:
        city_input: Free-text city name, e.g. "Paris" or "  Mumbai  ".

    Returns:
        A fully populated LocationResult dataclass.

    Raises:
        ValueError: If the input is blank or no city is found.
        ConnectionError: If the geocoding API is unreachable or returns an error.
    """
    city_input = _sanitize_input(city_input)
    raw_results = _fetch_geocoding_results(city_input)
    best_match = _select_best_match(raw_results, city_input)
    return _build_location_result(best_match, city_input)


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

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
