"""
test_llm_synthesis.py

Validation suite for src/llm_synthesis.py — Slice 5: LLM Narrative Synthesis.

All standard tests are OFFLINE — no API calls, no network dependency.
_fallback_narrative is tested directly.
synthesize_narrative is exercised with unittest.mock to avoid API calls.

OPTIONAL INTEGRATION TEST:
  test_integration_real_api_call makes a real call to claude-haiku-4-5.
  It runs ONLY when ANTHROPIC_API_KEY is present in the environment.

Run:
    python -m pytest tests/test_llm_synthesis.py -v
    python -m pytest tests/test_llm_synthesis.py -v -k "not integration"
    python tests/test_llm_synthesis.py
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ClimateContext, EconomicImpact, LocationResult, WeatherObservation
from src.llm_synthesis import (
    _build_payload,
    _fallback_narrative,
    _season_from_dates,
    synthesize_narrative,
    Z_NORMAL_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Synthetic Fixture Builder
# ---------------------------------------------------------------------------

def _make_impact(
    city_name: str = "Phoenix",
    country: str = "United States",
    country_code: str = "US",
    admin_region: str = "Arizona",
    date_range_start: str = "2023-07-01",
    date_range_end: str = "2023-07-31",
    observed_temp_mean_c: float = 34.0,
    baseline_mean_c: float = 30.0,
    baseline_stddev_c: float = 1.6,
    z_score: float = 2.5,
    anomaly_classification: str = "notable",
    cdd_observed: float = 95.0,
    cdd_baseline: float = 80.0,
    cdd_delta: float = 15.0,
    hdd_observed: float = 0.0,
    hdd_baseline: float = 0.0,
    hdd_delta: float = 0.0,
    wet_bulb_temp_c: float = 24.5,
    trend_slope_c_per_decade: float = 0.25,
    climate_confidence: str = "high",
    electricity_price_tier: int = 1,
    electricity_price_per_kwh_usd: float = 0.16,
    electricity_price_source: str = "EIA / NRCan",
    delta_energy_cost_usd: float = 14.40,
    uncertainty_band_pct: float = 15.0,
    econ_confidence: str = "high",
) -> EconomicImpact:
    """
    Builds a fully populated EconomicImpact chain for testing.
    No API calls are made — all values are synthetic and deterministic.
    """
    location = LocationResult(
        city_name=city_name,
        country=country,
        country_code=country_code,
        latitude=33.4,
        longitude=-112.0,
        admin_region=admin_region,
        match_confidence="high",
    )
    observation = WeatherObservation(
        location=location,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        observed_temp_mean_c=observed_temp_mean_c,
        observed_precip_sum_mm=2.1,
        dewpoint_mean_c=16.5,
        data_source="open-meteo",
        data_quality_flag="complete",
    )
    ctx = ClimateContext(
        observation=observation,
        baseline_period="1991-2020",
        baseline_mean_c=baseline_mean_c,
        baseline_stddev_c=baseline_stddev_c,
        z_score=z_score,
        anomaly_classification=anomaly_classification,
        cdd_observed=cdd_observed,
        cdd_baseline=cdd_baseline,
        cdd_delta=cdd_delta,
        hdd_observed=hdd_observed,
        hdd_baseline=hdd_baseline,
        hdd_delta=hdd_delta,
        wet_bulb_temp_c=wet_bulb_temp_c,
        trend_slope_c_per_decade=trend_slope_c_per_decade,
        confidence=climate_confidence,
        confidence_note="",
    )
    return EconomicImpact(
        climate_context=ctx,
        electricity_price_per_kwh_usd=electricity_price_per_kwh_usd,
        electricity_price_source=electricity_price_source,
        electricity_price_tier=electricity_price_tier,
        delta_energy_cost_usd=delta_energy_cost_usd,
        per_unit_description="per 100m² residential unit, observation period",
        uncertainty_band_pct=uncertainty_band_pct,
        confidence=econ_confidence,
        confidence_note="",
    )


# ---------------------------------------------------------------------------
# _season_from_dates Tests
# ---------------------------------------------------------------------------

def test_season_july_is_summer():
    assert _season_from_dates("2023-07-01", "2023-07-31") == "summer"
    print("  ✓ July → summer")


def test_season_january_is_winter():
    assert _season_from_dates("2023-01-15", "2023-01-31") == "winter"
    print("  ✓ January → winter")


def test_season_april_is_spring():
    assert _season_from_dates("2023-04-01", "2023-04-30") == "spring"
    print("  ✓ April → spring")


def test_season_october_is_autumn():
    assert _season_from_dates("2023-10-15", "2023-10-31") == "autumn"
    print("  ✓ October → autumn")


def test_season_december_is_winter():
    assert _season_from_dates("2023-12-01", "2023-12-31") == "winter"
    print("  ✓ December → winter")


def test_season_invalid_date_returns_period_string():
    result = _season_from_dates("not-a-date", "2023-07-31")
    assert result == "the observation period"
    print("  ✓ Invalid date → 'the observation period'")


def test_season_empty_string_returns_period_string():
    result = _season_from_dates("", "")
    assert result == "the observation period"
    print("  ✓ Empty string → 'the observation period'")


# ---------------------------------------------------------------------------
# _build_payload Tests
# ---------------------------------------------------------------------------

def test_build_payload_returns_dict():
    result = _build_payload(_make_impact())
    assert isinstance(result, dict)
    print("  ✓ _build_payload returns dict")


def test_build_payload_contains_city_name():
    result = _build_payload(_make_impact(city_name="Helsinki"))
    assert result["city"] == "Helsinki"
    print("  ✓ _build_payload extracts city_name correctly")


def test_build_payload_contains_z_score():
    result = _build_payload(_make_impact(z_score=2.75))
    assert result["z_score"] == 2.75
    print("  ✓ _build_payload extracts z_score correctly")


def test_build_payload_contains_all_required_keys():
    """All keys the LLM is authorised to cite must be present in the payload."""
    required_keys = {
        "city", "country", "admin_region", "latitude", "longitude",
        "date_range_start", "date_range_end", "observed_temp_mean_c",
        "baseline_period", "baseline_mean_c", "baseline_stddev_c",
        "z_score", "anomaly_classification",
        "cdd_observed", "cdd_baseline", "cdd_delta",
        "hdd_observed", "hdd_baseline", "hdd_delta",
        "wet_bulb_temp_c", "trend_slope_c_per_decade", "climate_confidence",
        "electricity_price_per_kwh_usd", "electricity_price_source",
        "electricity_price_tier", "delta_energy_cost_usd",
        "per_unit_description", "uncertainty_band_pct", "economic_confidence",
    }
    result = _build_payload(_make_impact())
    missing = required_keys - result.keys()
    assert not missing, f"Missing payload keys: {missing}"
    print(f"  ✓ All {len(required_keys)} required keys present in payload")


def test_build_payload_anomaly_classification_extracted():
    result = _build_payload(_make_impact(anomaly_classification="exceptional", z_score=3.5))
    assert result["anomaly_classification"] == "exceptional"
    print("  ✓ _build_payload extracts anomaly_classification correctly")


def test_build_payload_economic_tier_and_cost_extracted():
    result = _build_payload(_make_impact(electricity_price_tier=2, delta_energy_cost_usd=28.50))
    assert result["electricity_price_tier"] == 2
    assert result["delta_energy_cost_usd"] == 28.50
    print("  ✓ _build_payload extracts economic tier and cost correctly")


# ---------------------------------------------------------------------------
# _fallback_narrative Tests — LPCA Structure
# ---------------------------------------------------------------------------

def test_fallback_returns_non_empty_string():
    impact = _make_impact()
    result = _fallback_narrative(impact, _build_payload(impact))
    assert isinstance(result, str)
    assert len(result) > 100
    print(f"  ✓ _fallback_narrative returns non-empty string ({len(result)} chars)")


def test_fallback_contains_all_four_lpca_section_headers():
    """All four LPCA section headers must be present in every fallback output."""
    impact = _make_impact()
    result = _fallback_narrative(impact, _build_payload(impact))
    for header in [
        "**Local Anchor**",
        "**Present Consequence**",
        "**Trend Context**",
        "**Actionable Framing**",
    ]:
        assert header in result, f"Missing LPCA header: {header}"
    print("  ✓ All four LPCA section headers present in fallback narrative")


def test_fallback_city_name_appears_in_output():
    impact = _make_impact(city_name="Nairobi")
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "Nairobi" in result
    print("  ✓ City name 'Nairobi' appears in fallback narrative")


def test_fallback_z_score_appears_in_output():
    """The formatted Z-score value must appear in the narrative."""
    impact = _make_impact(z_score=2.50)
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "+2.50" in result or "2.50" in result
    print("  ✓ Z-score value (+2.50) appears in fallback narrative")


# ---------------------------------------------------------------------------
# _fallback_narrative Tests — Normal Classification (variability note)
# ---------------------------------------------------------------------------

def test_fallback_normal_classification_no_anomaly_language():
    """
    Z ≤ ±1.5σ (classification='normal') must not produce anomaly language.
    CLAUDE.md mandates: return a variability note, not an anomaly narrative.
    """
    impact = _make_impact(
        z_score=0.8,
        anomaly_classification="normal",
        cdd_delta=2.0,
        delta_energy_cost_usd=0.0,
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    result_lower = result.lower()
    assert "notable" not in result_lower, "'notable' must not appear in normal classification output"
    assert "exceptional" not in result_lower, "'exceptional' must not appear in normal classification output"
    assert any(word in result_lower for word in ["normal", "variability", "within"])
    print("  ✓ Normal Z-score: no anomaly language, variability note returned")


def test_fallback_normal_classification_contains_all_lpca_headers():
    """Normal classification must still produce all four LPCA section headers."""
    impact = _make_impact(z_score=0.5, anomaly_classification="normal", cdd_delta=-3.0)
    result = _fallback_narrative(impact, _build_payload(impact))
    for header in [
        "**Local Anchor**",
        "**Present Consequence**",
        "**Trend Context**",
        "**Actionable Framing**",
    ]:
        assert header in result, f"Missing header in normal-class output: {header}"
    print("  ✓ Normal classification still produces complete four-section LPCA structure")


# ---------------------------------------------------------------------------
# _fallback_narrative Tests — Anomaly Severity Language
# ---------------------------------------------------------------------------

def test_fallback_notable_anomaly_uses_notable_language():
    impact = _make_impact(z_score=2.5, anomaly_classification="notable")
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "notable" in result.lower()
    print("  ✓ 'notable' anomaly classification → 'notable' in narrative")


def test_fallback_exceptional_anomaly_uses_exceptional_language():
    impact = _make_impact(z_score=3.5, anomaly_classification="exceptional")
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "exceptional" in result.lower()
    print("  ✓ 'exceptional' anomaly classification → 'exceptional' in narrative")


# ---------------------------------------------------------------------------
# _fallback_narrative Tests — CDD Delta and Cost
# ---------------------------------------------------------------------------

def test_fallback_positive_cdd_delta_includes_cost_estimate():
    """When cdd_delta > 0 and delta_energy_cost_usd > 0, the $ figure must appear."""
    impact = _make_impact(
        z_score=2.5,
        anomaly_classification="notable",
        cdd_delta=15.0,
        delta_energy_cost_usd=14.40,
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "$14.40" in result
    print("  ✓ Positive CDD delta: cost estimate ($14.40) cited in narrative")


def test_fallback_negative_cdd_delta_no_excess_cooling_cost():
    """cdd_delta ≤ 0 → no excess cooling cost statement generated."""
    impact = _make_impact(
        z_score=-2.0,
        anomaly_classification="notable",
        cdd_delta=-8.0,
        hdd_delta=0.0,
        delta_energy_cost_usd=0.0,
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    result_lower = result.lower()
    assert "no additional cooling cost" in result_lower or "did not produce" in result_lower
    print("  ✓ Negative CDD delta: no excess cooling cost in narrative")


def test_fallback_negative_cdd_positive_hdd_mentions_heating():
    """When hdd_delta > 0 alongside negative cdd_delta, HDD increase should appear."""
    impact = _make_impact(
        z_score=-2.0,
        anomaly_classification="notable",
        cdd_delta=-8.0,
        hdd_delta=12.0,
        hdd_observed=55.0,
        hdd_baseline=43.0,
        delta_energy_cost_usd=0.0,
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "12.0" in result or "HDD" in result
    print("  ✓ Negative CDD delta with positive HDD delta: heating increase mentioned")


# ---------------------------------------------------------------------------
# _fallback_narrative Tests — Wet-Bulb Temperature
# ---------------------------------------------------------------------------

def test_fallback_wet_bulb_cited_when_nonzero():
    """Wet-bulb temperature must appear when non-zero for an anomaly."""
    impact = _make_impact(
        z_score=2.5,
        anomaly_classification="notable",
        wet_bulb_temp_c=27.3,
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "27.3" in result
    print("  ✓ Non-zero wet-bulb temperature (27.3°C) cited in anomaly narrative")


def test_fallback_wet_bulb_omitted_when_zero():
    """Wet-bulb = 0.0 (unavailable sentinel) must not be cited as a heat stress value."""
    impact = _make_impact(
        z_score=2.5,
        anomaly_classification="notable",
        wet_bulb_temp_c=0.0,
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "Wet-bulb temperature reached 0.0" not in result
    print("  ✓ Wet-bulb = 0.0: sentinel value not cited as a heat stress indicator")


# ---------------------------------------------------------------------------
# _fallback_narrative Tests — Tier and Regional Proxy Badge
# ---------------------------------------------------------------------------

def test_fallback_tier5_includes_regional_proxy_badge():
    """Tier 5 pricing must display the 'Regional Proxy' badge."""
    impact = _make_impact(
        z_score=2.5,
        anomaly_classification="notable",
        electricity_price_tier=5,
        electricity_price_source="Regional proxy (IEA Tier 3 median)",
        electricity_price_per_kwh_usd=0.15,
        uncertainty_band_pct=60.0,
        econ_confidence="low",
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "Regional Proxy" in result
    print("  ✓ Tier 5 pricing → 'Regional Proxy' badge appears in narrative")


def test_fallback_tier1_no_proxy_badge():
    """Tier 1 (authoritative source) must NOT show the Regional Proxy badge."""
    impact = _make_impact(electricity_price_tier=1)
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "Regional Proxy" not in result
    print("  ✓ Tier 1 pricing → no 'Regional Proxy' badge in narrative")


def test_fallback_tier4_includes_proxy_badge():
    """Tier 4 (emerging markets proxy) must also show the 'Regional Proxy' badge."""
    impact = _make_impact(
        z_score=2.5,
        anomaly_classification="notable",
        electricity_price_tier=4,
        electricity_price_source="World Bank Energy Data",
        electricity_price_per_kwh_usd=0.12,
        uncertainty_band_pct=40.0,
        econ_confidence="low",
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "Regional Proxy" in result
    print("  ✓ Tier 4 pricing → 'Regional Proxy' badge appears in narrative")


# ---------------------------------------------------------------------------
# _fallback_narrative Tests — Confidence and Trend
# ---------------------------------------------------------------------------

def test_fallback_low_climate_confidence_adds_caution_note():
    """Low climate confidence must produce a visible caution note in the output."""
    impact = _make_impact(climate_confidence="low")
    result = _fallback_narrative(impact, _build_payload(impact))
    result_lower = result.lower()
    assert any(word in result_lower for word in ["low", "caution", "quality", "interpret"])
    print("  ✓ Low climate confidence → caution note appended to narrative")


def test_fallback_warming_trend_referenced_in_narrative():
    """A trend slope > 0.1°C/decade should appear in the narrative."""
    impact = _make_impact(
        anomaly_classification="notable",
        trend_slope_c_per_decade=0.35,
    )
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "0.350" in result or "warming trend" in result.lower()
    print("  ✓ Warming trend of +0.350°C/decade referenced in narrative")


def test_fallback_near_zero_trend_uses_neutral_language():
    """|slope| < 0.05 → 'no statistically meaningful temperature trend' language."""
    impact = _make_impact(trend_slope_c_per_decade=0.02)
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "no statistically meaningful" in result.lower()
    print("  ✓ Near-zero trend slope → neutral 'no statistically meaningful' language")


def test_fallback_cooling_trend_uses_cooling_language():
    """A negative trend slope → 'cooling trend' language in the narrative."""
    impact = _make_impact(trend_slope_c_per_decade=-0.18)
    result = _fallback_narrative(impact, _build_payload(impact))
    assert "cooling trend" in result.lower()
    print("  ✓ Negative trend slope → 'cooling trend' language in narrative")


# ---------------------------------------------------------------------------
# synthesize_narrative Tests — Mocked _call_llm
# ---------------------------------------------------------------------------

@patch("src.llm_synthesis._call_llm")
def test_synthesize_falls_back_when_llm_returns_none(mock_call):
    """When _call_llm returns None, synthesize_narrative must return the fallback template."""
    mock_call.return_value = None
    impact = _make_impact()
    result = synthesize_narrative(impact)
    assert isinstance(result, str)
    assert len(result) > 50
    assert "**Local Anchor**" in result  # Confirms the fallback template was used
    mock_call.assert_called_once()
    print("  ✓ _call_llm returns None → fallback template returned by synthesize_narrative")


@patch("src.llm_synthesis._call_llm")
def test_synthesize_returns_llm_text_when_available(mock_call):
    """When _call_llm returns a string, synthesize_narrative must return it unchanged."""
    llm_text = (
        "**Local Anchor** — Phoenix, Arizona recorded Z=+2.50σ.\n\n"
        "**Present Consequence** — Grid stress elevated.\n\n"
        "**Trend Context** — Warming at +0.25°C/decade.\n\n"
        "**Actionable Framing** — Review energy procurement."
    )
    mock_call.return_value = llm_text
    result = synthesize_narrative(_make_impact())
    assert result == llm_text
    print("  ✓ _call_llm returns text → LLM text returned without modification")


@patch("src.llm_synthesis._call_llm")
def test_synthesize_always_returns_str(mock_call):
    """synthesize_narrative must always return a str for any anomaly classification."""
    mock_call.return_value = None
    for classification in ["normal", "notable", "exceptional"]:
        impact = _make_impact(anomaly_classification=classification)
        result = synthesize_narrative(impact)
        assert isinstance(result, str), f"Expected str for classification='{classification}'"
    print("  ✓ synthesize_narrative always returns str for all anomaly classifications")


@patch("src.llm_synthesis._call_llm")
def test_synthesize_passes_correct_payload_to_call_llm(mock_call):
    """synthesize_narrative must pass a dict payload with the correct city to _call_llm."""
    mock_call.return_value = None
    impact = _make_impact(city_name="Copenhagen")
    synthesize_narrative(impact)
    mock_call.assert_called_once()
    payload = mock_call.call_args[0][0]
    assert isinstance(payload, dict)
    assert payload["city"] == "Copenhagen"
    print("  ✓ synthesize_narrative passes correct payload dict to _call_llm")


# ---------------------------------------------------------------------------
# OPTIONAL INTEGRATION TEST — requires ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason=(
        "OPTIONAL INTEGRATION TEST — set ANTHROPIC_API_KEY to run. "
        "Makes a real API call to claude-haiku-4-5."
    ),
)
def test_integration_real_api_call_returns_narrative():
    """
    OPTIONAL INTEGRATION TEST.

    Makes one real API call to claude-haiku-4-5. Verifies the response
    is a non-empty string of substantive length. Does not assert on exact
    content since LLM output is non-deterministic.

    Skipped automatically when ANTHROPIC_API_KEY is not set.

    To run:
        ANTHROPIC_API_KEY=<key> python -m pytest tests/test_llm_synthesis.py \
            -v -k "integration"
    """
    impact = _make_impact(
        city_name="Singapore",
        country="Singapore",
        country_code="SG",
        admin_region="",
        z_score=2.8,
        anomaly_classification="notable",
        cdd_delta=12.0,
        delta_energy_cost_usd=9.90,
        electricity_price_tier=3,
        electricity_price_source="IEA World Energy Prices",
        electricity_price_per_kwh_usd=0.22,
        uncertainty_band_pct=25.0,
        econ_confidence="medium",
    )
    result = synthesize_narrative(impact)
    assert isinstance(result, str), "synthesize_narrative must return str"
    assert len(result) > 80, f"Expected substantive narrative, got only: '{result[:100]}'"
    print(f"  ✓ [INTEGRATION] Real API call returned {len(result)}-char narrative")
    print(f"    First 200 chars:\n    {result[:200]}")


# ---------------------------------------------------------------------------
# Plain-Python Runner (for use without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    offline_tests = [
        # _season_from_dates
        test_season_july_is_summer,
        test_season_january_is_winter,
        test_season_april_is_spring,
        test_season_october_is_autumn,
        test_season_december_is_winter,
        test_season_invalid_date_returns_period_string,
        test_season_empty_string_returns_period_string,
        # _build_payload
        test_build_payload_returns_dict,
        test_build_payload_contains_city_name,
        test_build_payload_contains_z_score,
        test_build_payload_contains_all_required_keys,
        test_build_payload_anomaly_classification_extracted,
        test_build_payload_economic_tier_and_cost_extracted,
        # _fallback_narrative: LPCA structure
        test_fallback_returns_non_empty_string,
        test_fallback_contains_all_four_lpca_section_headers,
        test_fallback_city_name_appears_in_output,
        test_fallback_z_score_appears_in_output,
        # _fallback_narrative: normal classification (variability note)
        test_fallback_normal_classification_no_anomaly_language,
        test_fallback_normal_classification_contains_all_lpca_headers,
        # _fallback_narrative: anomaly severity
        test_fallback_notable_anomaly_uses_notable_language,
        test_fallback_exceptional_anomaly_uses_exceptional_language,
        # _fallback_narrative: CDD delta and cost
        test_fallback_positive_cdd_delta_includes_cost_estimate,
        test_fallback_negative_cdd_delta_no_excess_cooling_cost,
        test_fallback_negative_cdd_positive_hdd_mentions_heating,
        # _fallback_narrative: wet-bulb
        test_fallback_wet_bulb_cited_when_nonzero,
        test_fallback_wet_bulb_omitted_when_zero,
        # _fallback_narrative: tier & proxy badge
        test_fallback_tier5_includes_regional_proxy_badge,
        test_fallback_tier1_no_proxy_badge,
        test_fallback_tier4_includes_proxy_badge,
        # _fallback_narrative: confidence & trend
        test_fallback_low_climate_confidence_adds_caution_note,
        test_fallback_warming_trend_referenced_in_narrative,
        test_fallback_near_zero_trend_uses_neutral_language,
        test_fallback_cooling_trend_uses_cooling_language,
        # synthesize_narrative (mocked — @patch decorator handles injection)
        test_synthesize_falls_back_when_llm_returns_none,
        test_synthesize_returns_llm_text_when_available,
        test_synthesize_always_returns_str,
        test_synthesize_passes_correct_payload_to_call_llm,
    ]

    print("\n── LLM Synthesis Engine Validation ──────────────────────\n")
    passed = 0
    failed = 0

    for test_fn in offline_tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_fn.__name__} FAILED: {exc}")
            failed += 1

    # Optional integration test
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n  [Integration test] ANTHROPIC_API_KEY found — running live API call...\n")
        try:
            test_integration_real_api_call_returns_narrative()
            passed += 1
        except Exception as exc:
            print(f"  ✗ Integration test FAILED: {exc}")
            failed += 1
    else:
        print("\n  [Integration test] SKIPPED — ANTHROPIC_API_KEY not set.")

    print(f"\n── Results: {passed} passed, {failed} failed ─────────────────\n")
    if failed > 0:
        sys.exit(1)
