"""
llm_synthesis.py

LLM Narrative Synthesis Engine — Slice 5.

Consumes a fully populated EconomicImpact (which carries the complete chain:
LocationResult → WeatherObservation → ClimateContext → EconomicImpact) and
returns a structured plain-English narrative using the Anthropic Claude API.

Model:  claude-haiku-4-5  (cost-minimised per CLAUDE.md)
Key:    ANTHROPIC_API_KEY from .env via python-dotenv

Narrative structure follows the LPCA framework (CLAUDE.md, non-negotiable):
  L — Local anchor        : city, season, Z-score deviation
  P — Present consequence : tangible current impact (CDD delta, grid load)
  C — Trend context       : 10-year trend slope — no overstating certainty
  A — Actionable framing  : risk-management lens, cost estimates, adaptation options

If the Claude API fails for any reason the function falls back to a
deterministic template-based narrative assembled from structured data.
The app is never non-functional when the LLM is down.

Public interface:
  synthesize_narrative(impact: EconomicImpact) → str
"""

import json
import os
from datetime import datetime
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from dotenv import load_dotenv

from src.schema import ClimateContext, EconomicImpact, LocationResult, WeatherObservation

load_dotenv()


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODEL_ID = "claude-haiku-4-5"
MAX_TOKENS = 1024

# Z-score thresholds — must match climate_stats.py
Z_NORMAL_THRESHOLD = 1.5
Z_NOTABLE_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# System prompt — LPCA framework (CLAUDE.md non-negotiable)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert climate risk analyst writing a structured briefing for a \
sustainability professional. You will receive a JSON payload with observed \
weather data, multi-hazard climate statistics (temperature, precipitation, wind), \
and economic impact estimates for a specific location and period.

Produce a professional narrative in exactly four clearly labelled paragraphs \
using the LPCA framework:

**Local Anchor**
State the city, approximate season or date range, and the temperature Z-score \
deviation from the 1991–2020 WMO baseline. If precipitation or wind anomalies \
are also notable (|Z| > 1.5), mention them briefly here.

**Present Consequence**
Translate the anomalies into tangible, current impacts. Reference CDD/HDD delta \
for heat; precipitation delta for flood or drought risk; wind anomaly if present. \
Where delta_energy_cost_usd > 0, include the estimated cost with its uncertainty \
band. If anomaly_classification is "normal" for ALL hazards, state that conditions \
are within normal seasonal variability — do NOT produce an anomaly narrative.

**Trend Context**
Place the observation within the 10-year trend slope (trend_slope_c_per_decade). \
Use measured language. Explicitly distinguish the short-term weather anomaly \
(Z-score) from the decadal climate trend. Do not overstate certainty.

**Actionable Framing**
Close with a risk-management perspective covering all relevant hazards: energy \
cost, flooding or drought exposure, wind risk. Reference the pricing tier and \
uncertainty band where relevant. Frame as cost and risk management — never as \
guilt or activist language.

---

MANDATORY RULES:
1. CITE ONLY the exact numbers present in the JSON payload.
2. When ALL hazard anomaly_classifications are "normal", produce a variability \
   note only.
3. DISTINGUISH short-term weather anomaly (Z-score) from long-term trend.
4. USE risk-management language only.
5. Keep the response under 420 words.
"""


# ---------------------------------------------------------------------------
# Payload extraction
# ---------------------------------------------------------------------------

def _build_payload(impact: EconomicImpact) -> dict:
    """
    Flattens the full EconomicImpact chain into a single dict for the LLM.
    Every value in the returned dict is a number the LLM is authorised to cite.
    """
    ctx: Optional[ClimateContext] = impact.climate_context
    obs: Optional[WeatherObservation] = ctx.observation if ctx else None
    loc: Optional[LocationResult] = obs.location if obs else None

    return {
        # Location
        "city": loc.city_name if loc else "Unknown",
        "country": loc.country if loc else "Unknown",
        "admin_region": loc.admin_region if loc else "",
        "latitude": loc.latitude if loc else 0.0,
        "longitude": loc.longitude if loc else 0.0,
        # Observation window
        "date_range_start": obs.date_range_start if obs else "",
        "date_range_end": obs.date_range_end if obs else "",
        "observed_temp_mean_c": obs.observed_temp_mean_c if obs else 0.0,
        "observed_precip_sum_mm": obs.observed_precip_sum_mm if obs else 0.0,
        # Climate statistics
        "baseline_period": ctx.baseline_period if ctx else "1991-2020",
        "baseline_mean_c": ctx.baseline_mean_c if ctx else 0.0,
        "baseline_stddev_c": ctx.baseline_stddev_c if ctx else 0.0,
        "z_score": ctx.z_score if ctx else 0.0,
        "anomaly_classification": ctx.anomaly_classification if ctx else "normal",
        "cdd_observed": ctx.cdd_observed if ctx else 0.0,
        "cdd_baseline": ctx.cdd_baseline if ctx else 0.0,
        "cdd_delta": ctx.cdd_delta if ctx else 0.0,
        "hdd_observed": ctx.hdd_observed if ctx else 0.0,
        "hdd_baseline": ctx.hdd_baseline if ctx else 0.0,
        "hdd_delta": ctx.hdd_delta if ctx else 0.0,
        "wet_bulb_temp_c": ctx.wet_bulb_temp_c if ctx else 0.0,
        "trend_slope_c_per_decade": ctx.trend_slope_c_per_decade if ctx else 0.0,
        "climate_confidence": ctx.confidence if ctx else "low",
        # Precipitation hazard
        "precip_observed_mm": ctx.precip_observed_mm if ctx else 0.0,
        "precip_baseline_mm": ctx.precip_baseline_mm if ctx else 0.0,
        "precip_z_score": ctx.precip_z_score if ctx else 0.0,
        "precip_anomaly_classification": ctx.precip_anomaly_classification if ctx else "normal",
        "drought_indicator": ctx.drought_indicator if ctx else "none",
        # Wind hazard
        "wind_speed_max_ms": ctx.wind_speed_max_ms if ctx else 0.0,
        "wind_baseline_ms": ctx.wind_baseline_ms if ctx else 0.0,
        "wind_z_score": ctx.wind_z_score if ctx else 0.0,
        "wind_anomaly_classification": ctx.wind_anomaly_classification if ctx else "normal",
        # Economic impact
        "electricity_price_per_kwh_usd": impact.electricity_price_per_kwh_usd,
        "electricity_price_source": impact.electricity_price_source,
        "electricity_price_tier": impact.electricity_price_tier,
        "delta_energy_cost_usd": impact.delta_energy_cost_usd,
        "per_unit_description": impact.per_unit_description,
        "uncertainty_band_pct": impact.uncertainty_band_pct,
        "economic_confidence": impact.confidence,
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(payload: dict) -> Optional[str]:
    """
    Makes one Claude API call. Returns the text response, or None on any failure.
    Failure modes caught: missing package, missing key, network error, rate limit.
    """
    if not _ANTHROPIC_AVAILABLE:
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        user_content = (
            "Produce a structured LPCA climate narrative for the following "
            "observation data. Adhere strictly to the system prompt rules.\n\n"
            + json.dumps(payload, indent=2)
        )
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return next(
            (block.text for block in response.content if block.type == "text"),
            None,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback template
# ---------------------------------------------------------------------------

def _season_from_dates(start: str, end: str) -> str:
    """Derives an approximate season label from an ISO date range start."""
    try:
        month = datetime.fromisoformat(start).month
    except (ValueError, TypeError):
        return "the observation period"
    seasons = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    return seasons.get(month, "the observation period")


def _fallback_narrative(impact: EconomicImpact, payload: dict) -> str:
    """
    Builds a deterministic LPCA narrative from the pre-extracted payload dict.
    Called whenever the Claude API is unavailable. No LLM call is made here.
    """
    city = payload["city"]
    country = payload["country"]
    admin = payload["admin_region"]
    location_str = f"{city}, {admin}, {country}" if admin else f"{city}, {country}"

    start = payload["date_range_start"]
    end = payload["date_range_end"]
    season = _season_from_dates(start, end)
    date_str = f"{start} to {end}" if (start and end) else "the observation period"

    z = payload["z_score"]
    classification = payload["anomaly_classification"]
    baseline_mean = payload["baseline_mean_c"]
    observed_temp = payload["observed_temp_mean_c"]
    cdd_delta = payload["cdd_delta"]
    hdd_delta = payload["hdd_delta"]
    trend = payload["trend_slope_c_per_decade"]
    wet_bulb = payload["wet_bulb_temp_c"]
    baseline_period = payload["baseline_period"]

    cost = payload["delta_energy_cost_usd"]
    price = payload["electricity_price_per_kwh_usd"]
    tier = payload["electricity_price_tier"]
    source = payload["electricity_price_source"]
    uncertainty = payload["uncertainty_band_pct"]
    per_unit = payload["per_unit_description"]
    econ_conf = payload["economic_confidence"]
    climate_conf = payload["climate_confidence"]

    tier_badge = " [Estimated — Regional Proxy]" if tier >= 4 else ""
    abs_z = abs(z)
    direction = "above" if z > 0 else "below"

    # ── L: Local Anchor ─────────────────────────────────────────────────────

    if classification == "normal":
        local_anchor = (
            f"**Local Anchor** — During {season} ({date_str}), {location_str} "
            f"recorded a mean temperature of {observed_temp:.1f}°C against a "
            f"{baseline_period} WMO baseline mean of {baseline_mean:.1f}°C. "
            f"The Z-score of {z:+.2f}σ falls within ±{Z_NORMAL_THRESHOLD}σ, "
            f"indicating normal seasonal variability rather than a statistically "
            f"significant anomaly."
        )
    else:
        severity = "exceptional" if classification == "exceptional" else "notable"
        local_anchor = (
            f"**Local Anchor** — During {season} ({date_str}), {location_str} "
            f"experienced a {severity} climate-context weather anomaly. The mean "
            f"temperature of {observed_temp:.1f}°C was {abs_z:.2f} standard "
            f"deviations {direction} the {baseline_period} WMO baseline mean of "
            f"{baseline_mean:.1f}°C (Z-score: {z:+.2f}σ)."
        )
        if wet_bulb > 0.0:
            local_anchor += (
                f" Wet-bulb temperature reached {wet_bulb:.1f}°C — the primary "
                f"heat stress indicator."
            )

    # ── P: Present Consequence ───────────────────────────────────────────────

    if classification == "normal":
        present_consequence = (
            f"**Present Consequence** — Cooling and heating degree day values "
            f"remain close to climatological norms (CDD observed: "
            f"{payload['cdd_observed']:.0f}, CDD baseline: "
            f"{payload['cdd_baseline']:.0f}; HDD observed: "
            f"{payload['hdd_observed']:.0f}, HDD baseline: "
            f"{payload['hdd_baseline']:.0f}). No significant deviation in energy "
            f"demand is attributable to the observed conditions."
        )
    elif cdd_delta > 0:
        cost_str = ""
        if cost > 0:
            margin = cost * uncertainty / 100.0
            low_c = max(0.0, cost - margin)
            high_c = cost + margin
            cost_str = (
                f" The estimated additional cooling energy expenditure is "
                f"${cost:.2f} USD (range: ${low_c:.2f}–${high_c:.2f}) "
                f"{per_unit}, based on a ${price:.2f}/kWh electricity rate "
                f"({source}){tier_badge}."
            )
        present_consequence = (
            f"**Present Consequence** — The positive temperature anomaly "
            f"produced {cdd_delta:.1f} additional Cooling Degree Days above "
            f"baseline, driving elevated grid load and increased cooling demand. "
            f"Observed CDD: {payload['cdd_observed']:.0f} vs. baseline "
            f"{payload['cdd_baseline']:.0f}.{cost_str}"
        )
    else:
        hdd_str = ""
        if hdd_delta > 0:
            hdd_str = (
                f" Heating demand increased by {hdd_delta:.1f} HDD above baseline "
                f"(observed: {payload['hdd_observed']:.0f}, baseline: "
                f"{payload['hdd_baseline']:.0f})."
            )
        present_consequence = (
            f"**Present Consequence** — The observed conditions did not produce "
            f"excess cooling demand relative to the baseline "
            f"(CDD delta: {cdd_delta:.1f}).{hdd_str} No additional cooling cost "
            f"is attributed to this period."
        )

    # ── C: Trend Context ─────────────────────────────────────────────────────

    if abs(trend) < 0.05:
        trend_desc = "no statistically meaningful temperature trend"
    elif trend > 0:
        trend_desc = f"a warming trend of +{trend:.3f}°C/decade"
    else:
        trend_desc = f"a cooling trend of {trend:.3f}°C/decade"

    if classification == "normal":
        trend_context = (
            f"**Trend Context** — Analysis of the 2011–2020 window within the "
            f"{baseline_period} reference period shows {trend_desc} for this "
            f"location and calendar period. The current observation is consistent "
            f"with natural interannual variability and does not indicate a "
            f"departure from the long-term climate signal."
        )
    else:
        trend_context = (
            f"**Trend Context** — Analysis of the 2011–2020 window within the "
            f"{baseline_period} reference period shows {trend_desc}. The "
            f"current Z-score anomaly ({z:+.2f}σ) is a short-term weather "
            f"departure and should be interpreted separately from the decadal "
            f"trend signal. Climate statistics confidence: {climate_conf}."
        )

    # ── A: Actionable Framing ────────────────────────────────────────────────

    if classification == "normal":
        actionable = (
            f"**Actionable Framing** — Conditions fall within normal operational "
            f"parameters. Routine energy budgets and cooling or heating "
            f"infrastructure plans based on historical norms remain appropriate. "
            f"Continue monitoring for persistent deviations in subsequent periods."
        )
    else:
        parts = [
            "**Actionable Framing** — From a risk-management standpoint, this "
            "event warrants review of short-term energy procurement and cooling "
            "infrastructure capacity."
        ]
        if cost > 0:
            parts.append(
                f"The estimated cost impact of ${cost:.2f} USD {per_unit} carries "
                f"a ±{uncertainty:.0f}% uncertainty band "
                f"(Tier {tier} pricing, economic confidence: {econ_conf}){tier_badge}."
            )
        if trend > 0.1:
            parts.append(
                "The positive decadal trend suggests events of this magnitude may "
                "recur more frequently, supporting the case for proactive adaptation."
            )
        parts.append(
            "Adaptation options include demand-side management programmes, "
            "cool-roof and building insulation upgrades, and review of utility "
            "hedging strategies."
        )
        actionable = " ".join(parts)

    # ── Assemble ─────────────────────────────────────────────────────────────

    narrative = "\n\n".join([local_anchor, present_consequence, trend_context, actionable])

    # Surface data-quality warnings where relevant
    if climate_conf == "low":
        narrative += (
            "\n\n*Note: Climate confidence is rated 'low' due to data quality "
            "issues in the baseline series. Interpret statistics with caution.*"
        )
    elif tier >= 4 and classification != "normal":
        narrative += (
            f"\n\n*Note: Electricity pricing uses a regional proxy estimate "
            f"(Tier {tier} — {source}){tier_badge}. Cost estimates carry high uncertainty.*"
        )

    return narrative


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def synthesize_narrative(impact: EconomicImpact) -> str:
    """
    Generates a structured LPCA climate narrative from a fully populated
    EconomicImpact dataclass.

    Attempts the Claude API (claude-haiku-4-5) first. Falls back to a
    deterministic template-based narrative on any failure — missing package,
    missing API key, network error, or rate limit. The app is never
    non-functional when the LLM is down.

    Args:
        impact: A fully populated EconomicImpact produced by economic_impact.py.

    Returns:
        A plain-English LPCA narrative string. Always a non-empty str.
    """
    payload = _build_payload(impact)
    llm_text = _call_llm(payload)
    if llm_text:
        return llm_text
    return _fallback_narrative(impact, payload)
