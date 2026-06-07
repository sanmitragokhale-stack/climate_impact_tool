"""
economic_impact.py

Economic Impact Engine — Slice 4.

Consumes a ClimateContext (Layer 2) and returns a populated EconomicImpact
dataclass with localised electricity pricing and energy cost estimates.

Public interface:
  compute_economic_impact(climate_context: ClimateContext) → EconomicImpact

5-tier electricity pricing proxy system (CLAUDE.md — non-negotiable):
  Tier 1 — USA, Canada               EIA / NRCan              $0.16/kWh  ±15%
  Tier 2 — EU27, UK, Norway          EUROSTAT                 $0.28/kWh  ±15%
  Tier 3 — Major OECD                IEA World Energy Prices  $0.22/kWh  ±25%
  Tier 4 — Emerging markets          World Bank Energy Data   $0.12/kWh  ±40%
  Tier 5 — All others                Regional proxy           $0.15/kWh  ±60%

Energy cost formula (CLAUDE.md):
  ΔCost = CDD_delta × floor_area_m2 × efficiency_kwh_per_m2_per_dd × price_per_kwh

  floor_area_m2             = 100   (reference residential unit)
  efficiency_kwh_per_m2_dd  = 0.06  (IEA conservative residential baseline)

ΔCost is always 0.0 when cdd_delta ≤ 0. A cool anomaly produces no excess
cooling cost — the function returns cleanly without error.

Prices sourced from: EIA Electric Power Monthly (2023), EUROSTAT nrg_pc_204
(Q1 2024, household, including taxes), IEA World Energy Prices (2023 edition),
World Bank Energy Sector data (2022–2023).
"""

from src.schema import ClimateContext, EconomicImpact


# ---------------------------------------------------------------------------
# Energy cost formula parameters
# ---------------------------------------------------------------------------

FLOOR_AREA_M2 = 100.0
EFFICIENCY_KWH_PER_M2_PER_DD = 0.06   # kWh per m² per degree-day, IEA baseline

PER_UNIT_DESCRIPTION = "per 100m² residential unit, observation period"


# ---------------------------------------------------------------------------
# 5-Tier Pricing Data
# ---------------------------------------------------------------------------
# Each entry: (price_usd_per_kwh, source_label, coverage_description)

_TIER_DATA: dict[int, tuple[float, str, str]] = {
    1: (
        0.16,
        "EIA / NRCan",
        "USA and Canada",
    ),
    2: (
        0.28,
        "EUROSTAT",
        "EU27, United Kingdom, and Norway",
    ),
    3: (
        0.22,
        "IEA World Energy Prices",
        "Major OECD economies (Japan, Australia, South Korea, and others)",
    ),
    4: (
        0.12,
        "World Bank Energy Data",
        "Emerging markets (India, Brazil, South Africa, and others)",
    ),
    5: (
        0.15,
        "Regional proxy (IEA Tier 3 median)",
        "All other countries — no direct electricity price data available",
    ),
}

# Uncertainty bands (±%) — widen as data authority decreases
_TIER_UNCERTAINTY_PCT: dict[int, float] = {
    1: 15.0,
    2: 15.0,
    3: 25.0,
    4: 40.0,
    5: 60.0,
}

# Overall confidence label per tier (feeds EconomicImpact.confidence)
_TIER_CONFIDENCE: dict[int, str] = {
    1: "high",
    2: "high",
    3: "medium",
    4: "low",
    5: "low",
}

# Plain-English explanations for the uncertainty band (required in confidence_note)
_TIER_UNCERTAINTY_EXPLANATION: dict[int, str] = {
    1: (
        "regional price variation within the country "
        "(authoritative government source)"
    ),
    2: (
        "price variation across EU member states and included non-EU countries "
        "(authoritative Eurostat source)"
    ),
    3: (
        "limited IEA data resolution and grid-mix diversity across these OECD economies"
    ),
    4: (
        "sparse World Bank coverage and high in-country price variability "
        "in emerging markets"
    ),
    5: (
        "this country has no direct electricity price data; "
        "price is estimated from the median of regional Tier 3 neighbours"
    ),
}


# ---------------------------------------------------------------------------
# Country Code → Tier Mapping
# Ordered sets; first match wins. Everything not matched → Tier 5.
# ---------------------------------------------------------------------------

_TIER1_CODES: frozenset[str] = frozenset({
    "US", "CA",
})

_TIER2_CODES: frozenset[str] = frozenset({
    # EU-27 member states
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
    # Non-EU Tier 2 countries with EUROSTAT-quality data
    "GB", "NO",
})

_TIER3_CODES: frozenset[str] = frozenset({
    # Major OECD economies with IEA World Energy Prices coverage
    "JP", "AU", "KR", "NZ", "CH", "IS", "MX", "CL", "TR", "IL",
    "CO", "CR",
})

_TIER4_CODES: frozenset[str] = frozenset({
    # Emerging markets with World Bank Energy Data coverage
    "IN", "BR", "ZA", "CN", "RU", "AR", "TH", "ID", "MY", "NG",
    "PK", "BD", "EG", "VN", "PH", "UA", "PE", "VE", "KE", "ET",
    "TZ", "GH", "DZ", "MA", "TN", "SA", "AE", "QA", "KW", "IQ",
    "IR", "MM", "LK", "NP", "KH", "LA", "PG", "SN", "CI",
})

# Tier 5 is the implicit fallback — no set required.


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def compute_economic_impact(climate_context: ClimateContext) -> EconomicImpact:
    """
    Derives a localised energy cost impact estimate from a ClimateContext.

    The cost estimate represents the additional cooling energy expenditure
    attributable to the CDD anomaly (cdd_delta) in the observation period.

    When cdd_delta ≤ 0 (the observed period is not warmer than the baseline),
    delta_energy_cost_usd is set to 0.0 and the confidence_note explains this.
    No error is raised.

    The electricity price, uncertainty band, and confidence tier are assigned
    using the 5-tier proxy system keyed on the country_code from the location
    embedded in the ClimateContext. Unknown country codes default to Tier 5.

    Args:
        climate_context: A ClimateContext produced by climate_stats.py (Slice 3).

    Returns:
        EconomicImpact with tier, price, cost estimate, uncertainty, and notes.
    """
    country_code = _extract_country_code(climate_context)
    tier = _assign_tier(country_code)

    price, source, coverage = _TIER_DATA[tier]
    uncertainty_pct = _TIER_UNCERTAINTY_PCT[tier]

    cdd_delta = climate_context.cdd_delta
    cost = _compute_delta_cost(cdd_delta, price)

    confidence = _final_confidence(
        _TIER_CONFIDENCE[tier],
        climate_context.confidence,
    )
    note = _build_confidence_note(
        tier, source, coverage, price, uncertainty_pct, cdd_delta, cost
    )

    return EconomicImpact(
        climate_context=climate_context,
        electricity_price_per_kwh_usd=price,
        electricity_price_source=source,
        electricity_price_tier=tier,
        delta_energy_cost_usd=round(cost, 2),
        per_unit_description=PER_UNIT_DESCRIPTION,
        uncertainty_band_pct=uncertainty_pct,
        confidence=confidence,
        confidence_note=note,
    )


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _extract_country_code(ctx: ClimateContext) -> str:
    """
    Extracts the ISO 3166-1 alpha-2 country code from the ClimateContext
    observation chain. Returns "XX" (triggers Tier 5) if any part is None.
    """
    try:
        code = ctx.observation.location.country_code
        return code.upper() if code else "XX"
    except AttributeError:
        return "XX"


def _assign_tier(country_code: str) -> int:
    """
    Maps a country code to a pricing tier (1–5).
    Tier 5 is the catch-all for any country not in Tiers 1–4.
    """
    if country_code in _TIER1_CODES:
        return 1
    if country_code in _TIER2_CODES:
        return 2
    if country_code in _TIER3_CODES:
        return 3
    if country_code in _TIER4_CODES:
        return 4
    return 5


def _compute_delta_cost(cdd_delta: float, price_per_kwh: float) -> float:
    """
    Applies the CLAUDE.md energy cost formula:
      ΔCost = CDD_delta × FLOOR_AREA_M2 × EFFICIENCY_KWH_PER_M2_PER_DD × price

    Returns 0.0 when cdd_delta ≤ 0 (no excess cooling cost from a cool anomaly).
    """
    if cdd_delta <= 0.0:
        return 0.0
    return cdd_delta * FLOOR_AREA_M2 * EFFICIENCY_KWH_PER_M2_PER_DD * price_per_kwh


def _final_confidence(tier_confidence: str, context_confidence: str) -> str:
    """
    Returns the lower of the tier confidence and the upstream ClimateContext
    confidence. A low-quality climate signal must not produce a high-confidence
    economic estimate.
    """
    rank = {"high": 2, "medium": 1, "low": 0}
    return min(
        tier_confidence, context_confidence,
        key=lambda c: rank.get(c, 0),
    )


def _build_confidence_note(
    tier: int,
    source: str,
    coverage: str,
    price: float,
    uncertainty_pct: float,
    cdd_delta: float,
    cost: float,
) -> str:
    """
    Builds the plain-English confidence note (required by CLAUDE.md).

    Always includes:
      • The tier number and source name.
      • Coverage area described in plain English.
      • A plain-English explanation of what the uncertainty band means.
      • The cost estimate with its uncertainty range when cdd_delta > 0.

    For cdd_delta ≤ 0, includes tier and source but notes zero excess cost.
    """
    if cdd_delta <= 0.0:
        return (
            f"Tier {tier} ({source}): ${price:.2f}/kWh ({coverage}). "
            f"CDD delta is {cdd_delta:.1f} — the observed period is not warmer "
            "than the 1991–2020 baseline. No excess cooling cost estimated."
        )

    explanation = _TIER_UNCERTAINTY_EXPLANATION[tier]
    margin = cost * uncertainty_pct / 100.0
    low = max(0.0, cost - margin)
    high = cost + margin

    return (
        f"Tier {tier} ({source}): ${price:.2f}/kWh ({coverage}). "
        f"The ±{uncertainty_pct:.0f}% uncertainty band reflects {explanation}. "
        f"Estimated additional cooling cost: ${cost:.2f} USD "
        f"(range: ${low:.2f}–${high:.2f}) {PER_UNIT_DESCRIPTION}."
    )
