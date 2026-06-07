"""
test_economic_impact.py

Validation suite for src/economic_impact.py — Slice 4: Economic Impact Engine.

All tests are OFFLINE — no API calls, no network dependency.
Synthetic ClimateContext objects are built with controlled country_code and
cdd_delta values so that every assertion is exact and deterministic.

Formula reference (CLAUDE.md):
  ΔCost = cdd_delta × 100 × 0.06 × price_per_kwh

Selected reference values (exact):
  cdd=10, Tier 1 ($0.16): 10 × 100 × 0.06 × 0.16 = $9.60
  cdd=10, Tier 2 ($0.28): 10 × 100 × 0.06 × 0.28 = $16.80
  cdd=10, Tier 3 ($0.22): 10 × 100 × 0.06 × 0.22 = $13.20
  cdd=10, Tier 4 ($0.12): 10 × 100 × 0.06 × 0.12 = $7.20
  cdd=10, Tier 5 ($0.15): 10 × 100 × 0.06 × 0.15 = $9.00
  cdd=30, Tier 1 ($0.16): 30 × 100 × 0.06 × 0.16 = $28.80
  cdd=30, Tier 2 ($0.28): 30 × 100 × 0.06 × 0.28 = $50.40

Run:
    python -m pytest tests/test_economic_impact.py -v
    python tests/test_economic_impact.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ClimateContext, EconomicImpact, LocationResult, WeatherObservation
from src.economic_impact import (
    compute_economic_impact,
    # Private helpers exposed for unit testing
    _assign_tier,
    _compute_delta_cost,
    _final_confidence,
    FLOOR_AREA_M2,
    EFFICIENCY_KWH_PER_M2_PER_DD,
)


# ---------------------------------------------------------------------------
# Synthetic Fixture Builders
# ---------------------------------------------------------------------------

def _make_context(
    country_code: str = "US",
    cdd_delta: float = 20.0,
    hdd_delta: float = 0.0,
    ctx_confidence: str = "high",
) -> ClimateContext:
    """
    Builds a minimal ClimateContext for testing without any API calls.
    Only the fields read by economic_impact.py are populated meaningfully;
    all others carry neutral defaults.
    """
    location = LocationResult(
        city_name="TestCity",
        country="TestCountry",
        country_code=country_code,
        latitude=0.0,
        longitude=0.0,
        admin_region="",
        match_confidence="high",
    )
    observation = WeatherObservation(
        location=location,
        date_range_start="2023-06-01",
        date_range_end="2023-06-30",
        observed_temp_mean_c=22.0,
        observed_precip_sum_mm=50.0,
        dewpoint_mean_c=14.0,
        data_source="open-meteo",
        data_quality_flag="complete",
    )
    return ClimateContext(
        observation=observation,
        baseline_period="1991-2020",
        baseline_mean_c=20.0,
        baseline_stddev_c=2.0,
        z_score=1.0,
        anomaly_classification="normal",
        cdd_observed=60.0,
        cdd_baseline=60.0 - cdd_delta,
        cdd_delta=cdd_delta,
        hdd_observed=0.0,
        hdd_baseline=0.0,
        hdd_delta=hdd_delta,
        wet_bulb_temp_c=18.0,
        trend_slope_c_per_decade=0.3,
        confidence=ctx_confidence,
        confidence_note="",
    )


# ---------------------------------------------------------------------------
# Tier Assignment Unit Tests
# ---------------------------------------------------------------------------

def test_assign_tier_usa():
    assert _assign_tier("US") == 1
    print("  ✓ US → Tier 1")


def test_assign_tier_canada():
    assert _assign_tier("CA") == 1
    print("  ✓ CA → Tier 1")


def test_assign_tier_france():
    assert _assign_tier("FR") == 2
    print("  ✓ FR → Tier 2")


def test_assign_tier_germany():
    assert _assign_tier("DE") == 2
    print("  ✓ DE → Tier 2")


def test_assign_tier_uk():
    assert _assign_tier("GB") == 2
    print("  ✓ GB → Tier 2")


def test_assign_tier_norway():
    assert _assign_tier("NO") == 2
    print("  ✓ NO → Tier 2")


def test_assign_tier_japan():
    assert _assign_tier("JP") == 3
    print("  ✓ JP → Tier 3")


def test_assign_tier_australia():
    assert _assign_tier("AU") == 3
    print("  ✓ AU → Tier 3")


def test_assign_tier_south_korea():
    assert _assign_tier("KR") == 3
    print("  ✓ KR → Tier 3")


def test_assign_tier_india():
    assert _assign_tier("IN") == 4
    print("  ✓ IN → Tier 4")


def test_assign_tier_brazil():
    assert _assign_tier("BR") == 4
    print("  ✓ BR → Tier 4")


def test_assign_tier_south_africa():
    assert _assign_tier("ZA") == 4
    print("  ✓ ZA → Tier 4")


def test_assign_tier_unknown_country_code():
    """Any country code not in Tiers 1–4 must default to Tier 5."""
    assert _assign_tier("ZZ") == 5
    print("  ✓ ZZ (unlisted) → Tier 5")


def test_assign_tier_empty_string_is_tier5():
    assert _assign_tier("") == 5
    print("  ✓ '' (empty) → Tier 5")


def test_assign_tier_xx_is_tier5():
    """'XX' is the internal sentinel for unknown country code → Tier 5."""
    assert _assign_tier("XX") == 5
    print("  ✓ XX (unknown sentinel) → Tier 5")


# ---------------------------------------------------------------------------
# Cost Formula Unit Tests
# ---------------------------------------------------------------------------

def test_delta_cost_formula_exact_tier1():
    """cdd=10, price=0.16: 10 × 100 × 0.06 × 0.16 = $9.60 exactly."""
    cost = _compute_delta_cost(10.0, 0.16)
    assert abs(cost - 9.60) < 1e-9, f"Expected $9.60, got ${cost:.6f}"
    print(f"  ✓ cdd=10, Tier 1 → ${cost:.2f} (expected $9.60)")


def test_delta_cost_formula_exact_tier2():
    """cdd=10, price=0.28: 10 × 100 × 0.06 × 0.28 = $16.80 exactly."""
    cost = _compute_delta_cost(10.0, 0.28)
    assert abs(cost - 16.80) < 1e-9, f"Expected $16.80, got ${cost:.6f}"
    print(f"  ✓ cdd=10, Tier 2 → ${cost:.2f} (expected $16.80)")


def test_delta_cost_formula_exact_tier3():
    """cdd=10, price=0.22: 10 × 100 × 0.06 × 0.22 = $13.20 exactly."""
    cost = _compute_delta_cost(10.0, 0.22)
    assert abs(cost - 13.20) < 1e-9, f"Expected $13.20, got ${cost:.6f}"
    print(f"  ✓ cdd=10, Tier 3 → ${cost:.2f} (expected $13.20)")


def test_delta_cost_formula_exact_tier4():
    """cdd=10, price=0.12: 10 × 100 × 0.06 × 0.12 = $7.20 exactly."""
    cost = _compute_delta_cost(10.0, 0.12)
    assert abs(cost - 7.20) < 1e-9, f"Expected $7.20, got ${cost:.6f}"
    print(f"  ✓ cdd=10, Tier 4 → ${cost:.2f} (expected $7.20)")


def test_delta_cost_formula_exact_tier5():
    """cdd=10, price=0.15: 10 × 100 × 0.06 × 0.15 = $9.00 exactly."""
    cost = _compute_delta_cost(10.0, 0.15)
    assert abs(cost - 9.00) < 1e-9, f"Expected $9.00, got ${cost:.6f}"
    print(f"  ✓ cdd=10, Tier 5 → ${cost:.2f} (expected $9.00)")


def test_delta_cost_larger_cdd_tier1():
    """cdd=30, price=0.16: 30 × 100 × 0.06 × 0.16 = $28.80 exactly."""
    cost = _compute_delta_cost(30.0, 0.16)
    assert abs(cost - 28.80) < 1e-9, f"Expected $28.80, got ${cost:.6f}"
    print(f"  ✓ cdd=30, Tier 1 → ${cost:.2f} (expected $28.80)")


def test_delta_cost_larger_cdd_tier2():
    """cdd=30, price=0.28: 30 × 100 × 0.06 × 0.28 = $50.40 exactly."""
    cost = _compute_delta_cost(30.0, 0.28)
    assert abs(cost - 50.40) < 1e-9, f"Expected $50.40, got ${cost:.6f}"
    print(f"  ✓ cdd=30, Tier 2 → ${cost:.2f} (expected $50.40)")


def test_delta_cost_zero_returns_zero():
    """cdd_delta = 0.0 → cost = 0.0."""
    assert _compute_delta_cost(0.0, 0.28) == 0.0
    print("  ✓ cdd_delta=0.0 → cost=$0.00")


def test_delta_cost_negative_returns_zero():
    """cdd_delta < 0 → cost = 0.0 (cool anomaly, no excess cooling)."""
    assert _compute_delta_cost(-15.0, 0.28) == 0.0
    print("  ✓ cdd_delta=-15.0 → cost=$0.00")


def test_formula_uses_correct_constants():
    """Verify FLOOR_AREA and EFFICIENCY constants match CLAUDE.md spec."""
    assert FLOOR_AREA_M2 == 100.0, f"Floor area must be 100m², got {FLOOR_AREA_M2}"
    assert EFFICIENCY_KWH_PER_M2_PER_DD == 0.06, \
        f"Efficiency must be 0.06 kWh/m²/DD, got {EFFICIENCY_KWH_PER_M2_PER_DD}"
    print(f"  ✓ Formula constants: floor={FLOOR_AREA_M2}m², efficiency={EFFICIENCY_KWH_PER_M2_PER_DD} kWh/m²/DD")


# ---------------------------------------------------------------------------
# Integration Tests — compute_economic_impact()
# ---------------------------------------------------------------------------

def test_schema_field_types():
    """All EconomicImpact fields must have the correct Python types."""
    impact = compute_economic_impact(_make_context("US", cdd_delta=20.0))

    assert isinstance(impact, EconomicImpact),                  "result must be EconomicImpact"
    assert isinstance(impact.electricity_price_per_kwh_usd, float), "price must be float"
    assert isinstance(impact.electricity_price_source, str),    "source must be str"
    assert isinstance(impact.electricity_price_tier, int),      "tier must be int"
    assert isinstance(impact.delta_energy_cost_usd, float),     "cost must be float"
    assert isinstance(impact.per_unit_description, str),        "per_unit must be str"
    assert isinstance(impact.uncertainty_band_pct, float),      "uncertainty must be float"
    assert isinstance(impact.confidence, str),                  "confidence must be str"
    assert isinstance(impact.confidence_note, str),             "note must be str"

    print("  ✓ EconomicImpact schema — all field types correct")


def test_tier1_price_and_source_in_output():
    """US produces Tier 1: price=$0.16, source contains 'EIA'."""
    impact = compute_economic_impact(_make_context("US"))
    assert impact.electricity_price_tier == 1
    assert impact.electricity_price_per_kwh_usd == 0.16
    assert "EIA" in impact.electricity_price_source
    print(f"  ✓ US → Tier 1, ${impact.electricity_price_per_kwh_usd}/kWh ({impact.electricity_price_source})")


def test_tier2_price_and_source_in_output():
    """FR produces Tier 2: price=$0.28, source contains 'EUROSTAT'."""
    impact = compute_economic_impact(_make_context("FR"))
    assert impact.electricity_price_tier == 2
    assert impact.electricity_price_per_kwh_usd == 0.28
    assert "EUROSTAT" in impact.electricity_price_source
    print(f"  ✓ FR → Tier 2, ${impact.electricity_price_per_kwh_usd}/kWh ({impact.electricity_price_source})")


def test_tier3_price_and_source_in_output():
    """JP produces Tier 3: price=$0.22, source contains 'IEA'."""
    impact = compute_economic_impact(_make_context("JP"))
    assert impact.electricity_price_tier == 3
    assert impact.electricity_price_per_kwh_usd == 0.22
    assert "IEA" in impact.electricity_price_source
    print(f"  ✓ JP → Tier 3, ${impact.electricity_price_per_kwh_usd}/kWh ({impact.electricity_price_source})")


def test_tier4_price_and_source_in_output():
    """IN produces Tier 4: price=$0.12, source contains 'World Bank'."""
    impact = compute_economic_impact(_make_context("IN"))
    assert impact.electricity_price_tier == 4
    assert impact.electricity_price_per_kwh_usd == 0.12
    assert "World Bank" in impact.electricity_price_source
    print(f"  ✓ IN → Tier 4, ${impact.electricity_price_per_kwh_usd}/kWh ({impact.electricity_price_source})")


def test_tier5_price_and_source_in_output():
    """ZZ produces Tier 5: price=$0.15, source contains 'proxy'."""
    impact = compute_economic_impact(_make_context("ZZ"))
    assert impact.electricity_price_tier == 5
    assert impact.electricity_price_per_kwh_usd == 0.15
    assert "proxy" in impact.electricity_price_source.lower()
    print(f"  ✓ ZZ → Tier 5, ${impact.electricity_price_per_kwh_usd}/kWh ({impact.electricity_price_source})")


# ---------------------------------------------------------------------------
# Uncertainty Band Tests
# ---------------------------------------------------------------------------

def test_uncertainty_band_tier1_and_2_is_15_pct():
    """Tier 1 and Tier 2 both carry ±15% uncertainty."""
    for code, label in [("US", "Tier 1"), ("FR", "Tier 2")]:
        impact = compute_economic_impact(_make_context(code))
        assert impact.uncertainty_band_pct == 15.0, \
            f"{label} ({code}): expected 15.0%, got {impact.uncertainty_band_pct}%"
    print("  ✓ Tier 1 (US) and Tier 2 (FR) → ±15% uncertainty")


def test_uncertainty_band_tier3_is_25_pct():
    impact = compute_economic_impact(_make_context("JP"))
    assert impact.uncertainty_band_pct == 25.0, \
        f"Expected 25.0%, got {impact.uncertainty_band_pct}%"
    print(f"  ✓ Tier 3 (JP) → ±{impact.uncertainty_band_pct:.0f}% uncertainty")


def test_uncertainty_band_tier4_is_40_pct():
    impact = compute_economic_impact(_make_context("IN"))
    assert impact.uncertainty_band_pct == 40.0, \
        f"Expected 40.0%, got {impact.uncertainty_band_pct}%"
    print(f"  ✓ Tier 4 (IN) → ±{impact.uncertainty_band_pct:.0f}% uncertainty")


def test_uncertainty_band_tier5_is_60_pct():
    impact = compute_economic_impact(_make_context("ZZ"))
    assert impact.uncertainty_band_pct == 60.0, \
        f"Expected 60.0%, got {impact.uncertainty_band_pct}%"
    print(f"  ✓ Tier 5 (ZZ) → ±{impact.uncertainty_band_pct:.0f}% uncertainty")


# ---------------------------------------------------------------------------
# Confidence Note Content Tests (CLAUDE.md: always include tier, source, explanation)
# ---------------------------------------------------------------------------

def test_confidence_note_contains_tier_number():
    """Tier number must appear explicitly in the confidence_note."""
    for code, expected_tier in [("US", 1), ("FR", 2), ("JP", 3), ("IN", 4), ("ZZ", 5)]:
        impact = compute_economic_impact(_make_context(code))
        assert f"Tier {expected_tier}" in impact.confidence_note, (
            f"'Tier {expected_tier}' not found in confidence_note for {code}. "
            f"Got: '{impact.confidence_note[:80]}...'"
        )
    print("  ✓ Tier number present in confidence_note for all 5 tiers")


def test_confidence_note_contains_source_name():
    """The source name must appear in the confidence_note for each tier."""
    cases = [
        ("US", "EIA"),
        ("FR", "EUROSTAT"),
        ("JP", "IEA"),
        ("IN", "World Bank"),
        ("ZZ", "proxy"),
    ]
    for code, expected_source_fragment in cases:
        impact = compute_economic_impact(_make_context(code))
        assert expected_source_fragment in impact.confidence_note, (
            f"'{expected_source_fragment}' not found in confidence_note for {code}. "
            f"Got: '{impact.confidence_note[:80]}...'"
        )
    print("  ✓ Source name present in confidence_note for all 5 tiers")


def test_confidence_note_contains_uncertainty_percentage():
    """The uncertainty percentage must be stated explicitly in the note."""
    cases = [("US", "15%"), ("JP", "25%"), ("IN", "40%"), ("ZZ", "60%")]
    for code, expected_pct in cases:
        impact = compute_economic_impact(_make_context(code))
        assert expected_pct in impact.confidence_note, (
            f"'{expected_pct}' not found in confidence_note for {code}."
        )
    print("  ✓ Uncertainty percentage explicit in confidence_note for all tiers")


def test_confidence_note_explains_uncertainty_in_plain_english():
    """
    The note must contain a plain-English explanation of the uncertainty band
    (not just the number). Verified by checking for 'reflects' or 'covers' or
    'variation', which are required words in the explanation templates.
    """
    for code in ["US", "FR", "JP", "IN", "ZZ"]:
        impact = compute_economic_impact(_make_context(code))
        note_lower = impact.confidence_note.lower()
        has_explanation = any(
            word in note_lower
            for word in ["reflects", "variation", "coverage", "data", "estimated"]
        )
        assert has_explanation, (
            f"confidence_note for {code} lacks a plain-English explanation. "
            f"Got: '{impact.confidence_note[:100]}...'"
        )
    print("  ✓ Plain-English explanation present in confidence_note for all tiers")


def test_confidence_note_tier5_flags_proxy_status():
    """Tier 5 note must explicitly flag that data is estimated from neighbours."""
    impact = compute_economic_impact(_make_context("ZZ"))
    note_lower = impact.confidence_note.lower()
    assert "proxy" in note_lower or "no direct" in note_lower or "estimated" in note_lower, (
        f"Tier 5 note must mention proxy/estimated status. Got: '{impact.confidence_note}'"
    )
    print(f"  ✓ Tier 5 note flags proxy status")


# ---------------------------------------------------------------------------
# Confidence Field Tests
# ---------------------------------------------------------------------------

def test_tier1_confidence_is_high():
    impact = compute_economic_impact(_make_context("US"))
    assert impact.confidence == "high", \
        f"Expected 'high' for Tier 1, got '{impact.confidence}'"
    print(f"  ✓ Tier 1 (US) → confidence='high'")


def test_tier2_confidence_is_high():
    impact = compute_economic_impact(_make_context("DE"))
    assert impact.confidence == "high", \
        f"Expected 'high' for Tier 2, got '{impact.confidence}'"
    print(f"  ✓ Tier 2 (DE) → confidence='high'")


def test_tier3_confidence_is_medium():
    impact = compute_economic_impact(_make_context("JP"))
    assert impact.confidence == "medium", \
        f"Expected 'medium' for Tier 3, got '{impact.confidence}'"
    print(f"  ✓ Tier 3 (JP) → confidence='medium'")


def test_tier4_confidence_is_low():
    impact = compute_economic_impact(_make_context("IN"))
    assert impact.confidence == "low", \
        f"Expected 'low' for Tier 4, got '{impact.confidence}'"
    print(f"  ✓ Tier 4 (IN) → confidence='low'")


def test_tier5_confidence_is_low():
    impact = compute_economic_impact(_make_context("ZZ"))
    assert impact.confidence == "low", \
        f"Expected 'low' for Tier 5, got '{impact.confidence}'"
    print(f"  ✓ Tier 5 (ZZ) → confidence='low'")


def test_low_context_confidence_degrades_tier1_to_low():
    """
    A Tier 1 country with a low-quality ClimateContext must produce
    confidence='low', not 'high'. The upstream signal quality caps the output.
    """
    impact = compute_economic_impact(_make_context("US", ctx_confidence="low"))
    assert impact.confidence == "low", (
        f"Expected 'low' when context confidence is 'low', got '{impact.confidence}'"
    )
    print("  ✓ Low context confidence overrides Tier 1 confidence → 'low'")


def test_medium_context_confidence_degrades_tier1_to_medium():
    """A medium-quality ClimateContext caps Tier 1 confidence at 'medium'."""
    impact = compute_economic_impact(_make_context("US", ctx_confidence="medium"))
    assert impact.confidence == "medium", (
        f"Expected 'medium' when context confidence is 'medium', got '{impact.confidence}'"
    )
    print("  ✓ Medium context confidence caps Tier 1 confidence → 'medium'")


def test_final_confidence_unit_low_wins():
    """_final_confidence always returns the lower of the two inputs."""
    assert _final_confidence("high", "low")    == "low"
    assert _final_confidence("low", "high")    == "low"
    assert _final_confidence("high", "medium") == "medium"
    assert _final_confidence("medium", "high") == "medium"
    assert _final_confidence("high", "high")   == "high"
    assert _final_confidence("low", "low")     == "low"
    print("  ✓ _final_confidence: lower tier always wins")


# ---------------------------------------------------------------------------
# Edge Case Tests — cdd_delta ≤ 0
# ---------------------------------------------------------------------------

def test_zero_cdd_delta_cost_is_zero():
    """
    cdd_delta = 0.0: observed period exactly matches baseline cooling demand.
    Must return delta_energy_cost_usd = 0.0 without error.
    """
    impact = compute_economic_impact(_make_context("US", cdd_delta=0.0))
    assert impact.delta_energy_cost_usd == 0.0, \
        f"Expected $0.00 for cdd_delta=0, got ${impact.delta_energy_cost_usd}"
    print("  ✓ cdd_delta=0.0 → delta_energy_cost_usd=$0.00 (no error)")


def test_negative_cdd_delta_cost_is_zero():
    """
    cdd_delta < 0: observed period is cooler than baseline — no excess cooling cost.
    Must return 0.0 without error for any negative value.
    """
    for cdd in [-0.1, -15.0, -100.0]:
        impact = compute_economic_impact(_make_context("US", cdd_delta=cdd))
        assert impact.delta_energy_cost_usd == 0.0, \
            f"Expected $0.00 for cdd_delta={cdd}, got ${impact.delta_energy_cost_usd}"
    print("  ✓ Negative cdd_delta → delta_energy_cost_usd=$0.00 (no error)")


def test_zero_cdd_confidence_note_explains_no_cost():
    """
    When cdd_delta ≤ 0, the confidence_note must explain why cost is zero.
    Must still include the tier number and source name.
    """
    impact = compute_economic_impact(_make_context("US", cdd_delta=0.0))
    note_lower = impact.confidence_note.lower()

    assert "tier 1" in impact.confidence_note.lower(), \
        "Tier number missing from zero-cost note"
    assert "eia" in note_lower, \
        "Source name missing from zero-cost note"
    assert (
        "no excess" in note_lower
        or "not warmer" in note_lower
        or "0.0" in impact.confidence_note
    ), f"Zero-cost note must explain why cost is zero. Got: '{impact.confidence_note}'"

    print(f"  ✓ Zero cdd_delta note includes tier, source, and zero-cost explanation")


def test_negative_cdd_note_includes_tier_and_source():
    """Even for cool anomalies, the confidence_note must include tier and source."""
    impact = compute_economic_impact(_make_context("JP", cdd_delta=-20.0))
    assert "Tier 3" in impact.confidence_note
    assert "IEA" in impact.confidence_note
    print("  ✓ Negative cdd_delta note still includes tier and source")


# ---------------------------------------------------------------------------
# Per-Unit Description Test
# ---------------------------------------------------------------------------

def test_per_unit_description_mentions_floor_area():
    """The per_unit_description must reference 100m²."""
    impact = compute_economic_impact(_make_context("US"))
    assert "100m²" in impact.per_unit_description, \
        f"'100m²' not found in per_unit_description: '{impact.per_unit_description}'"
    print(f"  ✓ per_unit_description: '{impact.per_unit_description}'")


def test_climate_context_reference_preserved():
    """The returned EconomicImpact must hold a reference to the input ClimateContext."""
    ctx = _make_context("US", cdd_delta=20.0)
    impact = compute_economic_impact(ctx)
    assert impact.climate_context is ctx, \
        "climate_context must be the same object passed in"
    print("  ✓ climate_context reference preserved in EconomicImpact")


# ---------------------------------------------------------------------------
# Plain-Python Runner (for use without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Tier assignment unit tests
        test_assign_tier_usa,
        test_assign_tier_canada,
        test_assign_tier_france,
        test_assign_tier_germany,
        test_assign_tier_uk,
        test_assign_tier_norway,
        test_assign_tier_japan,
        test_assign_tier_australia,
        test_assign_tier_south_korea,
        test_assign_tier_india,
        test_assign_tier_brazil,
        test_assign_tier_south_africa,
        test_assign_tier_unknown_country_code,
        test_assign_tier_empty_string_is_tier5,
        test_assign_tier_xx_is_tier5,
        # Cost formula unit tests
        test_delta_cost_formula_exact_tier1,
        test_delta_cost_formula_exact_tier2,
        test_delta_cost_formula_exact_tier3,
        test_delta_cost_formula_exact_tier4,
        test_delta_cost_formula_exact_tier5,
        test_delta_cost_larger_cdd_tier1,
        test_delta_cost_larger_cdd_tier2,
        test_delta_cost_zero_returns_zero,
        test_delta_cost_negative_returns_zero,
        test_formula_uses_correct_constants,
        # Integration tests
        test_schema_field_types,
        test_tier1_price_and_source_in_output,
        test_tier2_price_and_source_in_output,
        test_tier3_price_and_source_in_output,
        test_tier4_price_and_source_in_output,
        test_tier5_price_and_source_in_output,
        # Uncertainty band tests
        test_uncertainty_band_tier1_and_2_is_15_pct,
        test_uncertainty_band_tier3_is_25_pct,
        test_uncertainty_band_tier4_is_40_pct,
        test_uncertainty_band_tier5_is_60_pct,
        # Confidence note content tests
        test_confidence_note_contains_tier_number,
        test_confidence_note_contains_source_name,
        test_confidence_note_contains_uncertainty_percentage,
        test_confidence_note_explains_uncertainty_in_plain_english,
        test_confidence_note_tier5_flags_proxy_status,
        # Confidence field tests
        test_tier1_confidence_is_high,
        test_tier2_confidence_is_high,
        test_tier3_confidence_is_medium,
        test_tier4_confidence_is_low,
        test_tier5_confidence_is_low,
        test_low_context_confidence_degrades_tier1_to_low,
        test_medium_context_confidence_degrades_tier1_to_medium,
        test_final_confidence_unit_low_wins,
        # Edge case tests
        test_zero_cdd_delta_cost_is_zero,
        test_negative_cdd_delta_cost_is_zero,
        test_zero_cdd_confidence_note_explains_no_cost,
        test_negative_cdd_note_includes_tier_and_source,
        # Per-unit and reference tests
        test_per_unit_description_mentions_floor_area,
        test_climate_context_reference_preserved,
    ]

    print("\n── Economic Impact Engine Validation ────────────────────\n")
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_fn.__name__} FAILED: {exc}")
            failed += 1

    print(f"\n── Results: {passed} passed, {failed} failed ─────────────────\n")

    if failed > 0:
        sys.exit(1)
