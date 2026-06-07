"""
app.py

Streamlit dashboard entrypoint for the Climate Impact Portfolio Tool.

Pipeline order (five slices):
  1. geocoding.py      → LocationResult
  2. weather.py        → WeatherObservation + baseline series
  3. climate_stats.py  → ClimateContext   (temp + precip + wind + drought)
  4. economic_impact.py → EconomicImpact
  5. llm_synthesis.py  → LPCA narrative string

UI layout:
  - Wide layout, city disambiguation picker
  - Risk summary scorecard at top
  - LPCA narrative immediately below
  - Tabs: Heat | Precipitation & Drought | Wind | Economic | Methodology
  - Plotly charts throughout
"""

import streamlit as st
from datetime import datetime
from statistics import mean

import plotly.graph_objects as go

from src.geocoding import geocode_city_candidates
from src.weather import fetch_weather_observation, fetch_baseline_series
from src.climate_stats import compute_climate_context
from src.economic_impact import compute_economic_impact
from src.llm_synthesis import synthesize_narrative
from src.schema import LocationResult


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Climate Impact Portfolio Tool",
    page_icon="🌍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — tighten metric tiles and section headers
# ---------------------------------------------------------------------------

st.markdown("""
<style>
[data-testid="stMetricLabel"] { font-size: 0.78rem; }
[data-testid="stMetricValue"] { font-size: 1.25rem; font-weight: 600; }
div[data-testid="stTabs"] [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #333;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 4px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Z-score thresholds (must match climate_stats.py)
# ---------------------------------------------------------------------------

Z_NORMAL = 1.5
Z_EXCEPTIONAL = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_badge(z: float, label: str) -> str:
    abs_z = abs(z)
    if abs_z <= Z_NORMAL:
        icon = "🟢"
    elif abs_z <= Z_EXCEPTIONAL:
        icon = "🟡"
    else:
        icon = "🔴"
    direction = f"+{abs_z:.1f}σ" if z >= 0 else f"−{abs_z:.1f}σ"
    return f"{icon} {label}: {direction}"


def _anomaly_color(classification: str, positive_is_bad: bool = True) -> str:
    if classification == "exceptional":
        return "#d62728"
    if classification == "notable":
        return "#ff7f0e"
    return "#2ca02c"


def _year_annual_means(baseline_series: list[dict]) -> tuple[list[int], list[float]]:
    """Returns (years, annual_mean_temps) for the full 1991–2020 baseline."""
    year_data: dict[int, list[float]] = {}
    for r in baseline_series:
        if r["temp_mean_c"] is not None:
            yr = int(r["date"][:4])
            year_data.setdefault(yr, []).append(r["temp_mean_c"])
    years = sorted(year_data.keys())
    means = [mean(year_data[y]) for y in years]
    return years, means


def _year_annual_precip(baseline_series: list[dict]) -> tuple[list[int], list[float]]:
    """Returns (years, annual_precip_totals_mm) for the baseline."""
    year_data: dict[int, float] = {}
    for r in baseline_series:
        if r.get("precip_sum_mm") is not None:
            yr = int(r["date"][:4])
            year_data[yr] = year_data.get(yr, 0.0) + r["precip_sum_mm"]
    years = sorted(year_data.keys())
    totals = [year_data[y] for y in years]
    return years, totals


def _linear_fit(x: list[int], y: list[float]) -> tuple[list[float], float]:
    """Returns (fitted_y_values, slope_per_unit_x)."""
    n = len(x)
    if n < 2:
        return y, 0.0
    x_mean = mean(x)
    y_mean = mean(y)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den == 0:
        return [y_mean] * n, 0.0
    slope = num / den
    intercept = y_mean - slope * x_mean
    fitted = [slope * xi + intercept for xi in x]
    return fitted, slope


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for key in ("candidates", "location", "results"):
    if key not in st.session_state:
        st.session_state[key] = None


# ---------------------------------------------------------------------------
# Sidebar — input form
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/220px-SNice.svg.png",
             width=40) if False else None  # placeholder guard
    st.title("🌍 Climate Impact\nPortfolio Tool")
    st.caption("Powered by ERA5 · Open-Meteo · Claude Haiku")
    st.divider()

    with st.form("city_form"):
        city_input = st.text_input(
            "City (optionally add country)",
            placeholder="e.g. London, UK  ·  Mumbai  ·  Chicago, Illinois",
            help="Type a city name. Add a country or region after a comma to narrow results.",
        )
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    if submitted and city_input.strip():
        with st.spinner("Searching…"):
            try:
                candidates = geocode_city_candidates(city_input.strip())
                st.session_state["candidates"] = candidates
                st.session_state["location"] = None
                st.session_state["results"] = None
            except ValueError as e:
                st.error(str(e))
            except ConnectionError as e:
                st.error(f"Geocoding unavailable: {e}")

    # --- Disambiguation picker ---
    candidates = st.session_state.get("candidates") or []
    if candidates and st.session_state.get("location") is None:
        unique_countries = {c.country_code for c in candidates}
        if len(candidates) > 1 and len(unique_countries) > 1:
            st.divider()
            st.markdown("**Multiple cities found — select one:**")
            options = []
            for c in candidates:
                pop = f"  pop. {c.population:,}" if c.population else ""
                options.append(f"{c.city_name}, {c.country}{pop}")
            with st.form("disambig_form"):
                chosen_label = st.radio("", options, label_visibility="collapsed")
                confirm = st.form_submit_button("Analyse this city", use_container_width=True)
            if confirm:
                idx = options.index(chosen_label)
                st.session_state["location"] = candidates[idx]
        else:
            # Single result or all same country — auto-select best
            st.session_state["location"] = candidates[0]

    st.divider()
    st.caption(
        "**Baseline:** 1991–2020 WMO 30-yr normal  \n"
        "**Anomaly metric:** Z-score  \n"
        "**Heat stress:** Stull (2011) wet-bulb  \n"
        "**Data:** Open-Meteo ERA5 archive"
    )


# ---------------------------------------------------------------------------
# Pipeline — runs once per location selection
# ---------------------------------------------------------------------------

location: LocationResult | None = st.session_state.get("location")

if location is not None and st.session_state.get("results") is None:
    progress = st.empty()
    try:
        with progress.container():
            st.info(f"Running analysis for **{location.city_name}, {location.country}**…")

        with st.spinner("Fetching weather observation…"):
            observation = fetch_weather_observation(location)

        try:
            obs_month = datetime.fromisoformat(observation.date_range_end).month
        except (ValueError, TypeError):
            obs_month = datetime.now().month

        with st.spinner("Downloading 30-year ERA5 baseline (30–60 s)…"):
            baseline_series = fetch_baseline_series(location, obs_month)

        with st.spinner("Computing climate statistics…"):
            climate_ctx = compute_climate_context(observation, baseline_series)

        with st.spinner("Estimating economic impact…"):
            economic = compute_economic_impact(climate_ctx)

        with st.spinner("Generating LPCA narrative…"):
            narrative = synthesize_narrative(economic)

        st.session_state["results"] = {
            "location": location,
            "observation": observation,
            "baseline_series": baseline_series,
            "climate_ctx": climate_ctx,
            "economic": economic,
            "narrative": narrative,
        }
        progress.empty()

    except ValueError as exc:
        progress.empty()
        st.error(f"**Data error:** {exc}")
        st.stop()
    except ConnectionError as exc:
        progress.empty()
        st.error(f"**Connection error:** {exc}")
        st.stop()
    except Exception as exc:
        progress.empty()
        st.error(f"**Unexpected error:** `{type(exc).__name__}: {exc}`")
        st.stop()


# ---------------------------------------------------------------------------
# Landing state — no results yet
# ---------------------------------------------------------------------------

results = st.session_state.get("results")

if results is None:
    st.markdown("## Welcome to the Climate Impact Portfolio Tool")
    st.markdown(
        "Enter a city name in the sidebar to run a full multi-hazard climate risk analysis "
        "against the **1991–2020 WMO 30-year climatological baseline**.\n\n"
        "**What this tool analyses:**\n"
        "- 🌡️ **Heat stress** — temperature Z-score, CDD/HDD anomaly, wet-bulb heat index\n"
        "- 🌧️ **Precipitation & drought** — rainfall anomaly, flood risk signal, drought indicator\n"
        "- 💨 **Wind** — wind speed anomaly vs. 30-year baseline\n"
        "- 💰 **Economic impact** — additional energy cost using 5-tier regional pricing\n"
        "- 📖 **LPCA narrative** — AI-generated risk briefing grounded in the data"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Try:** London, UK")
    with col2:
        st.info("**Try:** Mumbai")
    with col3:
        st.info("**Try:** Chicago, Illinois")
    st.stop()


# ===========================================================================
# Results dashboard
# ===========================================================================

loc = results["location"]
obs = results["observation"]
ctx = results["climate_ctx"]
eco = results["economic"]
narrative = results["narrative"]
baseline_series = results["baseline_series"]


# ---------------------------------------------------------------------------
# Top header row
# ---------------------------------------------------------------------------

hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown(f"## 📍 {loc.city_name}, {loc.country}")
    region_str = f"{loc.admin_region}  ·  " if loc.admin_region else ""
    pop_str = f"Pop. {loc.population:,}  ·  " if loc.population else ""
    st.caption(
        f"{region_str}{pop_str}"
        f"Lat {loc.latitude:.3f}°  Lon {loc.longitude:.3f}°  ·  "
        f"Observation: {obs.date_range_start} → {obs.date_range_end}  ·  "
        f"Source: {obs.data_source}"
    )
with hcol2:
    # Geocoding confidence badge
    conf = loc.match_confidence
    if conf == "high":
        st.success("✅ High confidence match")
    elif conf == "medium":
        st.warning("⚠️ Medium confidence match")
    else:
        st.error("🔴 Low confidence — verify location")

if obs.data_quality_flag == "partial":
    st.warning("⚠️ **Partial observed data** — >15% of daily temperature values missing. Results may be less reliable.")

st.divider()


# ---------------------------------------------------------------------------
# Risk scorecard — one tile per hazard
# ---------------------------------------------------------------------------

st.markdown('<p class="section-header">Risk Scorecard</p>', unsafe_allow_html=True)

sc1, sc2, sc3, sc4, sc5 = st.columns(5)

with sc1:
    z = ctx.z_score
    delta_str = f"{'↑' if z > 0 else '↓'} {abs(z):.2f}σ vs baseline"
    st.metric("🌡️ Heat Z-score", f"{z:+.2f} σ", delta_str,
              delta_color="inverse" if z > 0 else "normal")

with sc2:
    wb = ctx.wet_bulb_temp_c
    wb_val = f"{wb:.1f} °C" if wb > 0 else "N/A"
    wb_label = ("🔴 Extreme" if wb >= 35 else
                "🔴 High" if wb >= 31 else
                "🟡 Moderate" if wb >= 28 else
                "🟢 Low")
    st.metric("💧 Wet-bulb", wb_val, wb_label if wb > 0 else "No dewpoint data",
              delta_color="off")

with sc3:
    pz = ctx.precip_z_score
    pdir = "wetter" if pz > 0 else "drier"
    pcls = ctx.precip_anomaly_classification
    p_icon = "🔴" if pcls == "exceptional" else "🟡" if pcls == "notable" else "🟢"
    st.metric("🌧️ Precip Z-score", f"{pz:+.2f} σ",
              f"{p_icon} {abs(pz):.1f}σ {pdir}",
              delta_color="off")

with sc4:
    wz = ctx.wind_z_score
    wcls = ctx.wind_anomaly_classification
    w_icon = "🔴" if wcls == "exceptional" else "🟡" if wcls == "notable" else "🟢"
    wval = f"{ctx.wind_speed_max_ms:.1f} m/s" if ctx.wind_speed_max_ms > 0 else "N/A"
    st.metric("💨 Wind speed", wval,
              f"{w_icon} {wz:+.2f}σ" if ctx.wind_speed_max_ms > 0 else "No data",
              delta_color="off")

with sc5:
    cost = eco.delta_energy_cost_usd
    if cost > 0:
        st.metric("💰 Extra cooling cost",
                  f"${cost:.2f} USD",
                  f"±{eco.uncertainty_band_pct:.0f}% · Tier {eco.electricity_price_tier}",
                  delta_color="inverse")
    else:
        st.metric("💰 Extra cooling cost", "None", "At or below baseline", delta_color="off")

st.divider()


# ---------------------------------------------------------------------------
# LPCA Narrative — prominent, above all detail tabs
# ---------------------------------------------------------------------------

st.markdown('<p class="section-header">📋 Climate Risk Briefing (LPCA)</p>',
            unsafe_allow_html=True)
st.caption(
    "Structured four-beat analysis: **L**ocal anchor · **P**resent consequence · "
    "**C**limate trend context · **A**ctionable framing"
)
with st.container(border=True):
    st.markdown(narrative)

st.divider()


# ---------------------------------------------------------------------------
# Detail tabs
# ---------------------------------------------------------------------------

tab_heat, tab_precip, tab_wind, tab_econ, tab_method = st.tabs([
    "🌡️  Heat",
    "🌧️  Precipitation & Drought",
    "💨  Wind",
    "💰  Economic Impact",
    "📚  Methodology",
])


# ── Tab: Heat ────────────────────────────────────────────────────────────────

with tab_heat:
    st.markdown("### Temperature Anomaly & Heat Stress")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-header">Anomaly Signal</p>', unsafe_allow_html=True)

        z = ctx.z_score
        abs_z = abs(z)
        direction_word = "warmer" if z > 0 else "cooler"

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Z-score", f"{z:+.2f} σ")
        with mc2:
            st.metric("Observed temp.", f"{obs.observed_temp_mean_c:.1f} °C")
        with mc3:
            st.metric("Baseline mean", f"{ctx.baseline_mean_c:.1f} °C",
                      delta=f"σ = {ctx.baseline_stddev_c:.2f} °C", delta_color="off")

        cls = ctx.anomaly_classification
        if abs_z <= Z_NORMAL:
            st.info(f"🔵 **Within normal variability** — Z = {z:+.2f}σ (threshold ±{Z_NORMAL}σ).")
        elif abs_z > Z_EXCEPTIONAL:
            st.error(f"🔴 **Exceptional anomaly** — {abs_z:.2f}σ {direction_word} than baseline.")
        else:
            st.warning(f"🟡 **Notable anomaly** — {abs_z:.2f}σ {direction_word} than baseline.")

        st.markdown('<p class="section-header" style="margin-top:16px">Degree Days</p>',
                    unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("CDD observed", f"{ctx.cdd_observed:.0f} °·days")
        with dc2:
            st.metric("CDD baseline", f"{ctx.cdd_baseline:.0f} °·days")
        with dc3:
            d = ctx.cdd_delta
            st.metric("CDD delta", f"{d:+.0f} °·days", delta_color="inverse" if d > 0 else "normal")

        dh1, dh2, dh3 = st.columns(3)
        with dh1:
            st.metric("HDD observed", f"{ctx.hdd_observed:.0f} °·days")
        with dh2:
            st.metric("HDD baseline", f"{ctx.hdd_baseline:.0f} °·days")
        with dh3:
            dh = ctx.hdd_delta
            st.metric("HDD delta", f"{dh:+.0f} °·days", delta_color="inverse" if dh > 0 else "normal")

        st.markdown('<p class="section-header" style="margin-top:16px">Wet-bulb Heat Stress</p>',
                    unsafe_allow_html=True)
        wb = ctx.wet_bulb_temp_c
        if wb == 0.0:
            st.warning("⚠️ Wet-bulb unavailable — dewpoint not returned by archive API.")
        else:
            wb1, wb2, wb3 = st.columns(3)
            with wb1:
                st.metric("Wet-bulb (Stull 2011)", f"{wb:.1f} °C")
            with wb2:
                st.metric("Dry-bulb", f"{obs.observed_temp_mean_c:.1f} °C")
            with wb3:
                st.metric("Dewpoint", f"{obs.dewpoint_mean_c:.1f} °C")

            if wb >= 35:
                st.error("🚨 **Extreme heat stress (≥35 °C)** — physiologically dangerous.")
            elif wb >= 31:
                st.error("🔴 **High heat stress (31–34 °C)** — dangerous for vulnerable groups.")
            elif wb >= 28:
                st.warning("🟡 **Moderate-high heat stress (28–30 °C)**.")
            elif wb >= 24:
                st.info(f"🔵 Mild heat stress — {wb:.1f} °C.")
            else:
                st.caption(f"Wet-bulb {wb:.1f} °C — comfortable range.")

    with col_right:
        st.markdown('<p class="section-header">30-Year Temperature Trend (1991–2020)</p>',
                    unsafe_allow_html=True)

        years, ann_means = _year_annual_means(baseline_series)
        fitted, slope = _linear_fit(years, ann_means)

        b_mean = ctx.baseline_mean_c
        b_std = ctx.baseline_stddev_c

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=years, y=ann_means,
            mode="lines+markers",
            name="Annual mean (baseline)",
            line=dict(color="#1f77b4", width=1.5),
            marker=dict(size=5),
        ))
        fig_trend.add_trace(go.Scatter(
            x=years, y=fitted,
            mode="lines",
            name=f"Trend ({slope * 10:+.2f} °C/decade)",
            line=dict(color="#d62728", width=2, dash="dot"),
        ))
        # ±1σ band
        fig_trend.add_hrect(
            y0=b_mean - b_std, y1=b_mean + b_std,
            fillcolor="rgba(31,119,180,0.08)",
            line_width=0,
            annotation_text="±1σ band",
            annotation_position="top left",
        )
        fig_trend.add_hline(
            y=b_mean, line_dash="dash", line_color="grey", line_width=1,
            annotation_text=f"Mean {b_mean:.1f}°C",
        )
        # Observed period mean as a scatter point
        obs_year = datetime.fromisoformat(obs.date_range_end).year
        fig_trend.add_trace(go.Scatter(
            x=[obs_year], y=[obs.observed_temp_mean_c],
            mode="markers",
            name=f"Observed ({obs.observed_temp_mean_c:.1f}°C)",
            marker=dict(color="#ff7f0e", size=12, symbol="star"),
        ))
        fig_trend.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Year", yaxis_title="Temp (°C)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # CDD / HDD grouped bar
        st.markdown('<p class="section-header">Degree Day Comparison</p>',
                    unsafe_allow_html=True)
        fig_dd = go.Figure(data=[
            go.Bar(name="Observed", x=["CDD", "HDD"],
                   y=[ctx.cdd_observed, ctx.hdd_observed],
                   marker_color=["#d62728", "#1f77b4"]),
            go.Bar(name="Baseline mean", x=["CDD", "HDD"],
                   y=[ctx.cdd_baseline, ctx.hdd_baseline],
                   marker_color=["rgba(214,39,40,0.35)", "rgba(31,119,180,0.35)"]),
        ])
        fig_dd.update_layout(
            barmode="group", height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="Degree-days (°·days)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Wet-bulb gauge
        if wb > 0:
            st.markdown('<p class="section-header">Heat Stress Gauge</p>',
                        unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=wb,
                delta={"reference": b_mean, "valueformat": ".1f"},
                title={"text": "Wet-bulb temp (°C)", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 40], "tickwidth": 1},
                    "bar": {"color": "#d62728"},
                    "steps": [
                        {"range": [0, 24], "color": "#d4edda"},
                        {"range": [24, 28], "color": "#fff3cd"},
                        {"range": [28, 31], "color": "#ffc107"},
                        {"range": [31, 35], "color": "#fd7e14"},
                        {"range": [35, 40], "color": "#dc3545"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": wb,
                    },
                },
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Trend slope interpretation
    trend = ctx.trend_slope_c_per_decade
    if abs(trend) >= 0.05:
        direction = "warming" if trend > 0 else "cooling"
        st.info(
            f"📈 **10-year trend signal:** {direction} at {abs(trend):.3f} °C/decade "
            f"(linear regression on 2011–2020 within the 1991–2020 baseline). "
            "This is a climate shift signal, distinct from the Z-score short-term anomaly."
        )


# ── Tab: Precipitation & Drought ─────────────────────────────────────────────

with tab_precip:
    st.markdown("### Precipitation Anomaly & Drought Risk")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        pz = ctx.precip_z_score
        pcls = ctx.precip_anomaly_classification
        abs_pz = abs(pz)
        p_direction = "wetter" if pz > 0 else "drier"

        st.markdown('<p class="section-header">Precipitation Signal</p>',
                    unsafe_allow_html=True)

        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            st.metric("Observed precip.", f"{ctx.precip_observed_mm:.1f} mm")
        with pm2:
            st.metric("Baseline mean", f"{ctx.precip_baseline_mm:.1f} mm")
        with pm3:
            delta_mm = ctx.precip_observed_mm - ctx.precip_baseline_mm
            st.metric("Delta", f"{delta_mm:+.1f} mm",
                      delta_color="inverse" if pz > 0 else "normal")

        st.metric("Precip Z-score", f"{pz:+.2f} σ")

        if pz == 0.0 and ctx.precip_baseline_mm == 0.0:
            st.warning("⚠️ Precipitation baseline data unavailable for this location/period.")
        elif abs_pz <= Z_NORMAL:
            st.info(f"🔵 **Within normal variability** — Z = {pz:+.2f}σ.")
        elif abs_pz > Z_EXCEPTIONAL:
            st.error(f"🔴 **Exceptional precipitation anomaly** — {abs_pz:.2f}σ {p_direction}.")
        else:
            st.warning(f"🟡 **Notable precipitation anomaly** — {abs_pz:.2f}σ {p_direction}.")

        # Flood risk signal
        st.markdown('<p class="section-header" style="margin-top:16px">Flood Risk Signal</p>',
                    unsafe_allow_html=True)
        if pz >= Z_EXCEPTIONAL:
            st.error(
                "🚨 **Elevated flood risk signal** — precipitation is more than 3σ above "
                "baseline. Assess drainage capacity and surface water management."
            )
        elif pz >= Z_NORMAL:
            st.warning(
                "🟡 **Moderate excess precipitation** — above-baseline rainfall may increase "
                "localised flood exposure. Monitor river levels and stormwater infrastructure."
            )
        else:
            st.info("🔵 No significant flood risk signal from precipitation anomaly.")

        # Drought indicator
        st.markdown('<p class="section-header" style="margin-top:16px">Drought Indicator</p>',
                    unsafe_allow_html=True)
        drought = ctx.drought_indicator
        if drought == "severe":
            st.error(
                "🔴 **Severe drought indicator** — precipitation Z-score ≤ −3σ. "
                "Significant moisture deficit; water resource stress is likely."
            )
        elif drought == "moderate":
            st.warning(
                "🟡 **Moderate drought indicator** — precipitation Z-score ≤ −1.5σ. "
                "Below-normal rainfall; monitor soil moisture and reservoir levels."
            )
        else:
            st.success("🟢 No drought signal — precipitation is at or above baseline levels.")

    with col_right:
        st.markdown('<p class="section-header">Precipitation vs. Baseline (per year)</p>',
                    unsafe_allow_html=True)

        years_p, totals_p = _year_annual_precip(baseline_series)
        fitted_p, slope_p = _linear_fit(years_p, totals_p)

        if years_p:
            fig_precip = go.Figure()
            fig_precip.add_trace(go.Bar(
                x=years_p, y=totals_p,
                name="Baseline annual precip",
                marker_color="rgba(31,119,180,0.55)",
            ))
            fig_precip.add_trace(go.Scatter(
                x=years_p, y=fitted_p,
                mode="lines",
                name=f"Trend ({slope_p * 10:+.1f} mm/decade)",
                line=dict(color="#d62728", width=2, dash="dot"),
            ))
            fig_precip.add_hline(
                y=ctx.precip_baseline_mm,
                line_dash="dash", line_color="grey", line_width=1,
                annotation_text=f"Monthly mean {ctx.precip_baseline_mm:.0f} mm",
            )
            # Observed value
            obs_year_p = datetime.fromisoformat(obs.date_range_end).year
            fig_precip.add_trace(go.Scatter(
                x=[obs_year_p], y=[ctx.precip_observed_mm],
                mode="markers",
                name=f"Observed ({ctx.precip_observed_mm:.0f} mm)",
                marker=dict(color="#ff7f0e", size=12, symbol="star"),
            ))
            fig_precip.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="Year", yaxis_title="Precip (mm)",
            )
            st.plotly_chart(fig_precip, use_container_width=True)

        # Observed vs. baseline bar
        st.markdown('<p class="section-header">Observed vs. Baseline</p>',
                    unsafe_allow_html=True)
        fig_pb = go.Figure(data=[
            go.Bar(
                x=["Observed", "Baseline mean"],
                y=[ctx.precip_observed_mm, ctx.precip_baseline_mm],
                marker_color=["#ff7f0e", "rgba(31,119,180,0.55)"],
                text=[f"{ctx.precip_observed_mm:.1f} mm", f"{ctx.precip_baseline_mm:.1f} mm"],
                textposition="outside",
            )
        ])
        fig_pb.update_layout(
            height=240, margin=dict(l=0, r=0, t=10, b=20),
            yaxis_title="Precipitation (mm)",
        )
        st.plotly_chart(fig_pb, use_container_width=True)


# ── Tab: Wind ────────────────────────────────────────────────────────────────

with tab_wind:
    st.markdown("### Wind Speed Anomaly")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        wz = ctx.wind_z_score
        wcls = ctx.wind_anomaly_classification
        wobs = ctx.wind_speed_max_ms
        wbase = ctx.wind_baseline_ms

        st.markdown('<p class="section-header">Wind Signal</p>', unsafe_allow_html=True)

        if wobs == 0.0:
            st.warning(
                "⚠️ Wind speed data was not returned by the archive API for this location "
                "and period. Wind anomaly analysis is unavailable."
            )
        else:
            wm1, wm2, wm3 = st.columns(3)
            with wm1:
                st.metric("Observed (mean daily max)", f"{wobs:.1f} m/s")
            with wm2:
                st.metric("Baseline mean", f"{wbase:.1f} m/s")
            with wm3:
                st.metric("Wind Z-score", f"{wz:+.2f} σ")

            abs_wz = abs(wz)
            w_direction = "stronger" if wz > 0 else "weaker"
            if abs_wz <= Z_NORMAL:
                st.info(f"🔵 **Within normal variability** — wind Z = {wz:+.2f}σ.")
            elif abs_wz > Z_EXCEPTIONAL:
                st.error(f"🔴 **Exceptional wind anomaly** — {abs_wz:.2f}σ {w_direction} than baseline.")
            else:
                st.warning(f"🟡 **Notable wind anomaly** — {abs_wz:.2f}σ {w_direction} than baseline.")

            # Beaufort scale reference
            st.markdown('<p class="section-header" style="margin-top:16px">Reference Scale</p>',
                        unsafe_allow_html=True)
            beaufort = [
                (0, 0.5, "Calm"), (0.5, 1.5, "Light air"), (1.5, 3.3, "Light breeze"),
                (3.3, 5.5, "Gentle breeze"), (5.5, 7.9, "Moderate breeze"),
                (7.9, 10.7, "Fresh breeze"), (10.7, 13.8, "Strong breeze"),
                (13.8, 17.1, "Near gale"), (17.1, 20.7, "Gale"),
                (20.7, 24.4, "Strong gale"), (24.4, 28.4, "Storm"),
                (28.4, 32.6, "Violent storm"), (32.6, 999, "Hurricane"),
            ]
            for low, high, name in beaufort:
                if low <= wobs < high:
                    st.info(f"🌬️ **{wobs:.1f} m/s** falls in the Beaufort **{name}** category.")
                    break

    with col_right:
        if wobs > 0:
            st.markdown('<p class="section-header">Wind: Observed vs. Baseline</p>',
                        unsafe_allow_html=True)
            fig_wind = go.Figure(data=[
                go.Bar(
                    x=["Observed", "Baseline mean"],
                    y=[wobs, wbase],
                    marker_color=["#9467bd", "rgba(148,103,189,0.4)"],
                    text=[f"{wobs:.1f} m/s", f"{wbase:.1f} m/s"],
                    textposition="outside",
                )
            ])
            fig_wind.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=20),
                yaxis_title="Wind speed (m/s)",
            )
            st.plotly_chart(fig_wind, use_container_width=True)

            # Z-score waterfall style indicator
            st.markdown('<p class="section-header">Z-score Indicator</p>',
                        unsafe_allow_html=True)
            fig_wz = go.Figure(go.Indicator(
                mode="number+gauge+delta",
                value=wz,
                delta={"reference": 0, "valueformat": "+.2f"},
                title={"text": "Wind Z-score (σ)"},
                gauge={
                    "axis": {"range": [-4, 4], "tickwidth": 1},
                    "bar": {"color": "#9467bd"},
                    "steps": [
                        {"range": [-4, -Z_EXCEPTIONAL], "color": "#d62728"},
                        {"range": [-Z_EXCEPTIONAL, -Z_NORMAL], "color": "#ff7f0e"},
                        {"range": [-Z_NORMAL, Z_NORMAL], "color": "#2ca02c"},
                        {"range": [Z_NORMAL, Z_EXCEPTIONAL], "color": "#ff7f0e"},
                        {"range": [Z_EXCEPTIONAL, 4], "color": "#d62728"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": wz,
                    },
                },
            ))
            fig_wz.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=0))
            st.plotly_chart(fig_wz, use_container_width=True)


# ── Tab: Economic Impact ──────────────────────────────────────────────────────

with tab_econ:
    st.markdown("### Economic Impact Estimate")
    st.caption(
        "Additional cooling energy expenditure attributable to the CDD anomaly — "
        "per 100 m² residential unit for the observation period."
    )

    tier = eco.electricity_price_tier
    cost = eco.delta_energy_cost_usd
    uncertainty = eco.uncertainty_band_pct
    price = eco.electricity_price_per_kwh_usd
    source = eco.electricity_price_source

    tier_icons = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"}
    tier_icon = tier_icons.get(tier, "⚪")

    if tier >= 4:
        st.warning(
            f"⚠️ **Estimated — Regional Proxy** — {loc.city_name} ({loc.country}) "
            f"falls into Pricing **Tier {tier}** ({source}). No direct electricity price data "
            f"is available. Uncertainty band: **±{uncertainty:.0f}%**."
        )

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        st.metric(f"Electricity price  {tier_icon} Tier {tier}", f"${price:.3f}/kWh",
                  help=f"Source: {source}")
    with ec2:
        st.metric("Uncertainty band", f"±{uncertainty:.0f}%")
    with ec3:
        st.metric("Economic confidence", eco.confidence.capitalize())
    with ec4:
        if cost > 0:
            st.metric("Additional cooling cost", f"${cost:.2f} USD",
                      delta_color="inverse")
        else:
            st.metric("Additional cooling cost", "None")

    if cost > 0:
        margin = cost * uncertainty / 100.0
        low_c = max(0.0, cost - margin)
        high_c = cost + margin

        st.caption(f"_{eco.per_unit_description}_  ·  Tier {tier} · {source}")

        # Range visualisation
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(
            x=["Estimated additional cost"],
            y=[cost],
            error_y=dict(type="data", symmetric=False,
                         array=[high_c - cost], arrayminus=[cost - low_c]),
            marker_color="#ff7f0e",
            text=[f"${cost:.2f}"],
            textposition="outside",
        ))
        fig_cost.add_hline(y=0, line_color="grey", line_width=1)
        fig_cost.update_layout(
            height=280, margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="USD",
            title=f"Cost range: ${low_c:.2f} – ${high_c:.2f} USD",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    else:
        st.info(
            "💡 **No additional cooling cost estimated.** "
            "Observed CDD is at or below the 1991–2020 baseline."
        )

    if ctx.confidence_note:
        with st.expander("Data quality notes", expanded=False):
            st.caption(ctx.confidence_note)


# ── Tab: Methodology ─────────────────────────────────────────────────────────

with tab_method:
    st.markdown("## Methodology & Data Sources")

    st.markdown("""
### Baseline Period
All statistics use the **1991–2020 WMO 30-year climatological normal** — the current
international standard reference period (World Meteorological Organisation, 2020).

---

### Z-score Anomaly Detection
```
Z = (observed period mean − 1991–2020 baseline mean) / baseline standard deviation
```

| Z-score | Classification | Action |
|---------|---------------|--------|
| abs(Z) ≤ 1.5σ | Normal | Within seasonal variability — no anomaly narrative |
| 1.5σ < abs(Z) ≤ 3.0σ | Notable | Climate-context weather anomaly |
| abs(Z) > 3.0σ | Exceptional | Rare statistical departure |

Applied to: **temperature**, **precipitation**, and **wind speed** independently.

---

### Stull (2011) Wet-bulb Temperature
Computed from dry-bulb temperature and relative humidity (derived from dewpoint via
August-Roche-Magnus formula). Valid 5–40 °C, 5–99% RH; accuracy ±1 °C.

> Stull, R. (2011). *Journal of Applied Meteorology and Climatology*, 50(11), 2267–2269.

---

### Cooling/Heating Degree Days (CDD/HDD)
Base temperature: **18 °C** (WMO standard).
- CDD = max(0, T_mean − 18 °C) per day — proxy for cooling energy demand
- HDD = max(0, 18 °C − T_mean) per day — proxy for heating energy demand

Energy cost:
```
ΔCost = CDD_delta × 100 m² × 0.06 kWh/m²/°-day × price_per_kWh
```

---

### Precipitation Anomaly & Drought
Monthly precipitation totals are compared against the same calendar month across 1991–2020.
A Z-score ≤ −1.5σ triggers a **moderate drought indicator**; ≤ −3σ triggers **severe**.

---

### Wind Anomaly
Mean daily maximum wind speeds over the observation period are compared against the
1991–2020 baseline distribution for the same calendar month.

---

### 5-Tier Electricity Pricing System

| Tier | Coverage | Source | Uncertainty |
|------|----------|--------|-------------|
| 🟢 1 | USA, Canada | EIA, NRCan | ±15% |
| 🟢 2 | EU27, UK, Norway | EUROSTAT | ±15% |
| 🟡 3 | Major OECD (Japan, Australia, Korea…) | IEA World Energy Prices | ±25% |
| 🟠 4 | Emerging markets (India, Brazil, South Africa…) | World Bank Energy Data | ±40% |
| 🔴 5 | All others | Regional median proxy | ±60% |

---

### Trend Signal
Linear regression on annual mean temperatures for 2011–2020 within the 1991–2020
baseline. Expressed in °C/decade. Distinct from the Z-score short-term anomaly.

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
