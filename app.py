"""
app.py

Streamlit entrypoint for the Climate Impact Portfolio Tool.

Runs the five-slice pipeline in order and displays structured results:
  Slice 1 — geocoding.py      → LocationResult
  Slice 2 — weather.py        → WeatherObservation + baseline series
  Slice 3 — climate_stats.py  → ClimateContext
  Slice 4 — economic_impact.py → EconomicImpact
  Slice 5 — llm_synthesis.py  → LPCA narrative string

All errors are caught and shown as clean user-facing messages.
"""

import streamlit as st
from datetime import datetime

from src.geocoding import geocode_city
from src.weather import fetch_weather_observation, fetch_baseline_series
from src.climate_stats import compute_climate_context
from src.economic_impact import compute_economic_impact
from src.llm_synthesis import synthesize_narrative


# ---------------------------------------------------------------------------
# Constants — must match climate_stats.py thresholds
# ---------------------------------------------------------------------------

Z_NORMAL_THRESHOLD = 1.5
Z_NOTABLE_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Climate Impact Portfolio Tool",
    page_icon="🌍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🌍 Climate Impact Portfolio Tool")
st.markdown(
    "Analyse observed weather conditions against the **1991–2020 WMO 30-year "
    "climatological baseline** for any city worldwide. Computes Z-score anomalies, "
    "Cooling/Heating Degree Day deviations, wet-bulb heat stress, and localised "
    "energy cost impacts — then synthesises a structured LPCA narrative."
)
st.divider()

# ---------------------------------------------------------------------------
# City input form
# ---------------------------------------------------------------------------

with st.form("city_form"):
    city_input = st.text_input(
        "Enter a city name",
        placeholder="e.g. Mumbai, London, Chicago, Singapore, Nairobi",
        help="The tool resolves your input via the Open-Meteo Geocoding API.",
    )
    submitted = st.form_submit_button(
        "Analyse Climate Impact", use_container_width=True
    )

# ---------------------------------------------------------------------------
# Pipeline — only runs when the form is submitted
# ---------------------------------------------------------------------------

if not submitted:
    st.stop()

if not city_input.strip():
    st.error("Please enter a city name before submitting.")
    st.stop()

try:

    # ── Stage 1: Geocode ─────────────────────────────────────────────────────

    with st.spinner(f"Locating **{city_input.strip()}** on the map…"):
        try:
            location = geocode_city(city_input)
        except ValueError as exc:
            st.error(
                f"**City not found.** {exc}\n\n"
                "Try a different spelling, add the country name (e.g. 'Springfield, Illinois'), "
                "or use a nearby major city."
            )
            st.stop()
        except ConnectionError as exc:
            st.error(
                f"**Geocoding service unavailable.** Check your internet connection and try again.\n\n"
                f"Detail: {exc}"
            )
            st.stop()

    # ── Stage 2: Weather observation + 30-year baseline ──────────────────────

    with st.spinner(
        f"Fetching recent weather and 30-year baseline for **{location.city_name}**… "
        "(30–60 seconds — downloading ERA5 reanalysis data)"
    ):
        try:
            observation = fetch_weather_observation(location)
        except ValueError as exc:
            st.error(f"**Weather data error:** {exc}")
            st.stop()
        except ConnectionError as exc:
            st.error(
                f"**Open-Meteo archive API unavailable.** Try again in a moment.\n\nDetail: {exc}"
            )
            st.stop()

        try:
            obs_month = datetime.fromisoformat(observation.date_range_end).month
        except (ValueError, TypeError):
            obs_month = datetime.now().month

        try:
            baseline_series = fetch_baseline_series(location, obs_month)
        except ValueError as exc:
            st.error(f"**Baseline data error:** {exc}")
            st.stop()
        except ConnectionError as exc:
            st.error(
                f"**Open-Meteo archive API unavailable (baseline fetch).** "
                f"Try again in a moment.\n\nDetail: {exc}"
            )
            st.stop()

    # ── Stage 3: Climate statistics ──────────────────────────────────────────

    with st.spinner(
        "Computing Z-score anomaly and degree-day statistics "
        "against the 1991–2020 WMO baseline…"
    ):
        try:
            climate_ctx = compute_climate_context(observation, baseline_series)
        except ValueError as exc:
            st.error(f"**Climate statistics error:** {exc}")
            st.stop()

    # ── Stage 4: Economic impact ──────────────────────────────────────────────

    with st.spinner("Estimating localised energy cost impact…"):
        economic = compute_economic_impact(climate_ctx)

    # ── Stage 5: Narrative synthesis ─────────────────────────────────────────

    with st.spinner(
        "Generating LPCA narrative (Claude Haiku) — "
        "falling back to template if API unavailable…"
    ):
        narrative = synthesize_narrative(economic)

except Exception as exc:
    st.error(
        "**An unexpected error occurred.** The pipeline hit an unhandled condition.\n\n"
        f"Detail: `{type(exc).__name__}: {exc}`\n\n"
        "Please try a different city, or report the issue."
    )
    st.stop()


# ===========================================================================
# Results display
# ===========================================================================

st.success("✅ Analysis complete")
st.divider()

# ---------------------------------------------------------------------------
# 1. City header + coordinates
# ---------------------------------------------------------------------------

st.subheader(f"📍 {location.city_name}, {location.country}")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Region / State", location.admin_region or "—")
with col_b:
    st.metric("Latitude", f"{location.latitude:.4f}°")
with col_c:
    st.metric("Longitude", f"{location.longitude:.4f}°")

meta_parts = [
    f"Observation window: **{observation.date_range_start}** → **{observation.date_range_end}**",
    f"Source: {observation.data_source}",
]
if location.population:
    meta_parts.append(f"City population: {location.population:,}")
st.caption("  ·  ".join(meta_parts))

if observation.data_quality_flag == "partial":
    st.warning(
        "⚠️ **Partial observed data** — More than 15% of daily temperature values "
        "were missing for this period. Results may be less reliable."
    )

# ---------------------------------------------------------------------------
# 2. Geocoding confidence badge
# ---------------------------------------------------------------------------

conf = location.match_confidence
if conf == "high":
    st.success("✅ **High confidence match** — Exact name and population data confirmed.")
elif conf == "medium":
    note = f"\n\n_{location.match_note}_" if location.match_note else ""
    st.warning(f"⚠️ **Medium confidence match** — Exact name match; population data unavailable.{note}")
else:
    note = f"\n\n_{location.match_note}_" if location.match_note else ""
    st.error(
        f"🔴 **Low confidence match** — The returned city name differs from your input. "
        f"Verify this is the intended location before relying on results.{note}"
    )

st.divider()

# ---------------------------------------------------------------------------
# 3. Z-score + anomaly classification
# ---------------------------------------------------------------------------

st.subheader("Climate Anomaly Signal")

z = climate_ctx.z_score
classification = climate_ctx.anomaly_classification
abs_z = abs(z)
direction_word = "warmer" if z > 0 else "cooler"

col_z1, col_z2, col_z3 = st.columns(3)
with col_z1:
    st.metric("Z-score", f"{z:+.2f} σ")
with col_z2:
    st.metric("Observed temp. (period mean)", f"{observation.observed_temp_mean_c:.1f} °C")
with col_z3:
    st.metric(
        "Baseline mean (1991–2020)",
        f"{climate_ctx.baseline_mean_c:.1f} °C",
        delta=f"σ = {climate_ctx.baseline_stddev_c:.2f} °C",
        delta_color="off",
    )

# Anomaly interpretation box
if abs_z <= Z_NORMAL_THRESHOLD:
    st.info(
        f"🔵 **Within Normal Variability** — Z-score of **{z:+.2f}σ** is within "
        f"±{Z_NORMAL_THRESHOLD}σ of the 1991–2020 baseline. This represents normal "
        "seasonal variability, not a statistically significant climate anomaly. "
        "No anomaly narrative is generated for observations within this threshold."
    )
elif abs_z > Z_NOTABLE_THRESHOLD:
    st.error(
        f"🔴 **Exceptional anomaly** — {location.city_name} was **{abs_z:.2f} standard "
        f"deviations {direction_word}** than the 1991–2020 WMO baseline "
        f"(Z = {z:+.2f}σ). This is a statistically exceptional departure (|Z| > 3.0σ)."
    )
else:
    st.warning(
        f"🟡 **Notable anomaly** — {location.city_name} was **{abs_z:.2f} standard "
        f"deviations {direction_word}** than the 1991–2020 WMO baseline "
        f"(Z = {z:+.2f}σ). This is a notable climate-context weather anomaly "
        f"(1.5σ < |Z| ≤ 3.0σ)."
    )

st.divider()

# ---------------------------------------------------------------------------
# 4. CDD / HDD delta
# ---------------------------------------------------------------------------

st.subheader("Degree Day Analysis")
st.caption(
    "Cooling Degree Days (CDD) and Heating Degree Days (HDD) — base temperature "
    "18 °C (WMO standard). Higher CDD = more cooling demand; higher HDD = more "
    "heating demand."
)

col_cdd1, col_cdd2, col_cdd3 = st.columns(3)
with col_cdd1:
    st.metric("CDD observed", f"{climate_ctx.cdd_observed:.0f} °·days")
with col_cdd2:
    st.metric("CDD baseline (1991–2020 mean)", f"{climate_ctx.cdd_baseline:.0f} °·days")
with col_cdd3:
    cdd_delta = climate_ctx.cdd_delta
    delta_sign = "+" if cdd_delta >= 0 else ""
    st.metric("CDD delta", f"{delta_sign}{cdd_delta:.0f} °·days")

col_hdd1, col_hdd2, col_hdd3 = st.columns(3)
with col_hdd1:
    st.metric("HDD observed", f"{climate_ctx.hdd_observed:.0f} °·days")
with col_hdd2:
    st.metric("HDD baseline (1991–2020 mean)", f"{climate_ctx.hdd_baseline:.0f} °·days")
with col_hdd3:
    hdd_delta = climate_ctx.hdd_delta
    delta_sign = "+" if hdd_delta >= 0 else ""
    st.metric("HDD delta", f"{delta_sign}{hdd_delta:.0f} °·days")

# Plain-English interpretations
if cdd_delta > 5:
    st.info(
        f"🌡️ **Elevated cooling demand** — {cdd_delta:.0f} additional Cooling Degree Days "
        "above the historical baseline indicate increased grid load and air conditioning demand "
        "compared to normal conditions for this period."
    )
elif cdd_delta < -5:
    st.info(
        f"❄️ **Reduced cooling demand** — {abs(cdd_delta):.0f} fewer Cooling Degree Days "
        "than the historical baseline suggest below-average cooling requirements for this period."
    )

if hdd_delta > 5:
    st.info(
        f"🔥 **Elevated heating demand** — {hdd_delta:.0f} additional Heating Degree Days "
        "above the historical baseline indicate increased energy demand for space heating."
    )
elif hdd_delta < -5:
    st.info(
        f"🌤️ **Reduced heating demand** — {abs(hdd_delta):.0f} fewer Heating Degree Days "
        "than the historical baseline suggest below-average heating requirements for this period."
    )

st.divider()

# ---------------------------------------------------------------------------
# 5. Wet-bulb temperature
# ---------------------------------------------------------------------------

st.subheader("Heat Stress Indicator")
st.caption("Wet-bulb temperature via the Stull (2011) approximation — the primary heat stress metric.")

wb = climate_ctx.wet_bulb_temp_c

if wb == 0.0:
    st.warning(
        "⚠️ **Wet-bulb temperature unavailable** — Dewpoint data was not returned by the "
        "archive API for this location and period. The Stull (2011) approximation requires "
        "dewpoint to compute wet-bulb temperature."
    )
else:
    col_wb1, col_wb2, col_wb3 = st.columns(3)
    with col_wb1:
        st.metric(
            "Wet-bulb temperature",
            f"{wb:.1f} °C",
            help="Stull (2011) — requires dewpoint. Accurate within ±1 °C for T 5–40 °C, RH 5–99%.",
        )
    with col_wb2:
        st.metric("Dry-bulb temperature", f"{observation.observed_temp_mean_c:.1f} °C")
    with col_wb3:
        st.metric("Dewpoint (period mean)", f"{observation.dewpoint_mean_c:.1f} °C")

    if wb >= 35:
        st.error(
            "🚨 **Extreme heat stress (≥35 °C wet-bulb)** — Physiologically dangerous for most "
            "humans. Risk of hyperthermia even for healthy individuals at rest in shade."
        )
    elif wb >= 31:
        st.error(
            "🔴 **High heat stress (31–34 °C wet-bulb)** — Dangerous for vulnerable populations "
            "and those engaged in outdoor physical activity. Significant cooling demand expected."
        )
    elif wb >= 28:
        st.warning(
            "🟡 **Moderate-high heat stress (28–30 °C wet-bulb)** — Conditions cause thermal "
            "discomfort. Elevated cooling demand and heat-health advisories may apply."
        )
    elif wb >= 24:
        st.info(f"🔵 Wet-bulb of {wb:.1f} °C — mild heat stress range. Generally manageable for most.")
    else:
        st.caption(f"Wet-bulb of {wb:.1f} °C — within comfortable range for most conditions.")

st.divider()

# ---------------------------------------------------------------------------
# 6. Economic impact
# ---------------------------------------------------------------------------

st.subheader("Economic Impact Estimate")
st.caption(
    "Additional cooling energy expenditure attributable to the CDD anomaly — "
    "per 100 m² residential unit for the observation period."
)

tier = economic.electricity_price_tier
cost = economic.delta_energy_cost_usd
uncertainty = economic.uncertainty_band_pct
price = economic.electricity_price_per_kwh_usd
source = economic.electricity_price_source

# Tier 4/5 proxy warning — prominently displayed per CLAUDE.md requirement
if tier >= 4:
    st.warning(
        f"⚠️ **Estimated — Regional Proxy** — {location.city_name} ({location.country}) "
        f"falls into Pricing **Tier {tier}** ({source}). No direct electricity price data "
        f"is available for this country. The estimate carries a **±{uncertainty:.0f}% "
        f"uncertainty band** and should be treated as an indicative order-of-magnitude "
        f"figure only."
    )

tier_icons = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"}
tier_icon = tier_icons.get(tier, "⚪")

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    st.metric(
        f"Electricity price  {tier_icon} Tier {tier}",
        f"${price:.3f} / kWh",
        help=f"Source: {source}",
    )
with col_e2:
    st.metric("Uncertainty band", f"±{uncertainty:.0f}%")
with col_e3:
    st.metric("Economic confidence", economic.confidence.capitalize())

if cost > 0:
    margin = cost * uncertainty / 100.0
    low_c = max(0.0, cost - margin)
    high_c = cost + margin
    st.metric(
        "Est. additional cooling cost",
        f"${cost:.2f} USD",
        delta=f"Range: ${low_c:.2f} – ${high_c:.2f} USD",
        delta_color="inverse",
    )
    st.caption(f"_{economic.per_unit_description}_")
    proxy_note = "  ·  Estimated — Regional Proxy" if tier >= 4 else ""
    st.caption(
        f"Tier {tier} · {source}{proxy_note}  ·  "
        f"Confidence: {economic.confidence}"
    )
else:
    st.info(
        "💡 **No additional cooling cost estimated.** The observed CDD is at or below "
        "the 1991–2020 baseline — no excess cooling expenditure is attributable to this period."
    )

if climate_ctx.confidence_note:
    with st.expander("Data quality notes", expanded=False):
        st.caption(climate_ctx.confidence_note)

st.divider()

# ---------------------------------------------------------------------------
# 7. LPCA narrative
# ---------------------------------------------------------------------------

st.subheader("LPCA Climate Narrative")
st.caption(
    "Structured four-beat analysis: **L**ocal anchor · **P**resent consequence · "
    "**C**limate trend context · **A**ctionable framing"
)

with st.container(border=True):
    st.markdown(narrative)

st.divider()

# ---------------------------------------------------------------------------
# 8. Methodology & Data Sources expander
# ---------------------------------------------------------------------------

with st.expander("📚 Methodology & Data Sources", expanded=False):
    st.markdown("""
### Baseline Period
All statistics use the **1991–2020 WMO 30-year climatological normal** — the current
international standard reference period (World Meteorological Organisation, 2020).
This window captures recent observed climate while providing sufficient length for
robust statistical estimation.

---

### Z-score Anomaly Detection
The anomaly metric is the **Z-score**:

```
Z = (observed period mean − 1991–2020 baseline mean) / baseline standard deviation
```

Thresholds (scientifically non-negotiable):

| Z-score | Classification | Interpretation |
|---------|---------------|----------------|
| abs(Z) ≤ 1.5σ | Normal | Within seasonal variability — no anomaly narrative |
| 1.5σ < abs(Z) ≤ 3.0σ | Notable | Climate-context weather anomaly |
| abs(Z) > 3.0σ | Exceptional | Rare statistical departure |

---

### Stull (2011) Wet-bulb Temperature
Wet-bulb temperature is computed from dry-bulb temperature and relative humidity
using the **Stull (2011)** empirical approximation:

> Stull, R. (2011). "Wet-Bulb Temperature from Relative Humidity and Air Temperature."
> *Journal of Applied Meteorology and Climatology*, 50(11), 2267–2269.

Relative humidity is derived from dry-bulb and dewpoint via the August-Roche-Magnus
formula. Valid range: 5–40 °C, 5–99% RH. Typical accuracy: ±1 °C.

Wet-bulb temperature captures both heat and humidity simultaneously — it is the
primary physiological heat stress metric, not dry-bulb temperature alone.

---

### Cooling/Heating Degree Days (CDD/HDD)
Degree days quantify departure from a **base temperature of 18 °C** (WMO standard):

- **CDD** = max(0, T_mean − 18 °C) per day — proxy for cooling energy demand
- **HDD** = max(0, 18 °C − T_mean) per day — proxy for heating energy demand

Baseline CDD/HDD are computed from actual 1991–2020 daily records; observed
values use the period-mean approximation.

Energy cost formula:
```
ΔCost = CDD_delta × 100 m² × 0.06 kWh/m²/°-day × price_per_kWh
```
(IEA conservative residential efficiency baseline)

---

### 5-Tier Electricity Pricing System

| Tier | Coverage | Source | Uncertainty |
|------|----------|--------|-------------|
| 🟢 1 | USA, Canada | EIA, NRCan | ±15% |
| 🟢 2 | EU27, UK, Norway | EUROSTAT | ±15% |
| 🟡 3 | Major OECD (Japan, Australia, Korea…) | IEA World Energy Prices | ±25% |
| 🟠 4 | Emerging markets (India, Brazil, South Africa…) | World Bank Energy Data | ±40% |
| 🔴 5 | All others | Regional median proxy | ±60% |

Countries in Tier 4 or 5 are labelled **Estimated — Regional Proxy** and carry
high uncertainty. All estimates are per **100 m² residential unit** for the
observation period.

---

### Trend Signal
The 10-year trend slope is computed by linear regression on annual mean temperatures
for the years **2011–2020** within the 1991–2020 baseline series. The result is
expressed in °C/decade. This is a climate shift signal distinct from the Z-score
short-term weather anomaly signal.

---

### Data Sources
| Component | Source |
|-----------|--------|
| Historical weather & geocoding | [Open-Meteo](https://open-meteo.com) (ERA5 reanalysis, free tier) |
| Electricity pricing — Tier 1 | EIA Electric Power Monthly (2023) |
| Electricity pricing — Tier 2 | EUROSTAT nrg_pc_204 (Q1 2024) |
| Electricity pricing — Tier 3 | IEA World Energy Prices (2023) |
| Electricity pricing — Tier 4 | World Bank Energy Sector Data (2022–2023) |
| Narrative synthesis | Anthropic Claude (claude-haiku-4-5) with deterministic fallback |
    """)
