# Climate Impact App — Update Feedback Notes

Source: initial user feedback round, collected [date]. Organized by branch/track and rough priority for planning purposes. This is raw material for a `/grill-me` planning session, not a finalized spec — several items are marked "needs definition" and should be interviewed before any code is written.

---

## Track 1 — Quick fix (low risk, do first)

**Branch:** `fix/lpca-output-formatting`

- Narrative output has inconsistent formatting: mixed fonts, stray brackets, and raw math/markdown notation leaking into rendered text (e.g. `$45.58USD(range:27.35-63.81)per100m^2residentialunit...` instead of plain readable text). Likely a formatting bug in how cost figures get inserted into the LPCA narrative string, not a redesign.
- Also fix while touching this area: `.env.example` is missing from the repo despite CLAUDE.md requiring it be committed.

---

## Track 2 — Content/logic changes (self-contained, needs light definition)

**Branch:** `feat/economic-scope-transparency`

- Economic analysis section only covers heat-stress/cooling-demand cost. Feedback: the app should be explicit about whether it has data on the economic impact of other hazards (precipitation/drought, wind) or not — currently silent on this, which reads as an omission.
- **Open question for grill-me:** does the app already have underlying data for non-heat hazard economic impact that just isn't surfaced, or does it need graceful "no data available" messaging? Needs codebase check before spec'ing the fix.

---

## Track 3 — New feature: city event timeline

**Branch:** `feat/city-event-timeline`

- Add a per-city climate events timeline: headline + link only, no summarization (avoids copyright issues, cheaper, simpler).
- Seeded with a small curated city list, then grows automatically via cache-on-miss (first search for a new city triggers enrichment and caches it).
- Plus a periodic refresh job to keep entries fresh.
- Requires a small persistent key-value/lightweight DB store (not static repo files).
- Needs a geocoding-match check as a cost/abuse safeguard.
- **Deferred to v2:** expanding event-timeline city coverage beyond the initial seed list; deeper disaster-database integration (EM-DAT/GDELT).

---

## Track 4 — New feature: user-supplied data override (needs full grill-me pass — least defined)

**Branch:** `feat/user-data-override`

- Allow users to supply their own city-specific data to substitute for the app's general/baseline data.
- **Open questions — none of these are answered yet, all need interview before spec'ing:**
  - What format does a user submit data in (upload file, form fields, API)?
  - Does supplied data fully replace the baseline dataset for that city, or blend/annotate alongside it?
  - Is this persisted (per-user account, needs auth) or session-only (re-entered each visit)?
  - How is arbitrary user-submitted data validated/sanitized before it feeds into cost calculations and public-facing narrative text?
- Treat as slow-track: do not build until grill-me has produced a real spec for this one specifically.

---

## Track 5 — Design system (direction decision, not purely technical)

**Branch:** depends on Track 6 outcome — see below.

- Feedback: current visuals rely on default chart-library styling; wants a deliberate visual design system (palette, typography, iconography).
- This item turned out to be entangled with a bigger decision — see Track 6.

---

## Track 6 — Frontend/stack decision: Next.js + Vercel rebuild

**Branch:** `rebuild/nextjs-frontend` (separate long-lived track, sequenced LAST)

- Underlying driver for Track 5: current Streamlit UI looks dated; this app is going into a job-application portfolio, so frontend polish matters for hiring-manager impressions.
- Decision: **do not do a full logic rewrite.** Keep the existing Python analytics pipeline (`geocoding.py`, `weather.py`, `climate_stats.py`, `economic_impact.py`) exactly as-is — it has 155 passing tests and validated domain logic (Stull wet-bulb calc, Z-scores, trend regression, etc.) that shouldn't be re-implemented from scratch.
- Approach: wrap the existing Python pipeline in a lightweight FastAPI service, deployed separately (Render/Railway free tier). Build a new Next.js + Tailwind frontend on Vercel (Hobby/free tier — fine for non-commercial portfolio use) that calls the Python service via API.
- Sequencing rationale: land Tracks 1–2 (cheap, low-risk fixes) and let Tracks 3/6-7 (below) mature on the current stack first, since that's backend/data logic the new frontend will need to consume regardless of what UI sits on top. Do the frontend rebuild once, after the underlying data model has stabilized — not before.
- This is where Track 5's actual design system (palette, typography, iconography) gets implemented, since it's a real frontend build rather than a CSS patch on Streamlit.

---

## Track 7 — Automation: research agent + site agent

**Branch:** `infra/research-and-site-agents` (two related PRs: agent build, then agent-triggered site-change PRs)

- **Research agent:** scheduled GitHub Action (quarterly). Uses Claude + web search restricted to a curated source allowlist (Yale/GMU climate comms programs, IPCC, peer-reviewed journals). Writes dated reports to `/research/`.
- **Site agent:** triggered by new research reports. Proposes site changes as a PR with rationale. Never auto-merges — human approval required for every change.
- Both agents must be testable via manual workflow trigger at any time, without waiting for the scheduled cadence.
- **Infra note:** GitHub Actions has no browser for OAuth login, so this agent must authenticate via an Anthropic API key (the separately purchased/metered one, not the Claude Pro subscription) stored as a GitHub secret. Set a rough budget/frequency ceiling before building, since every scheduled + manual-trigger run incurs real API cost.
- **Deferred to v2:** a second research agent for climate science findings — likely folded into the existing research agent as a second report section instead of built as a separate agent.

---

## Track 8 — Fallback narrative hazard-classification bug (correctness fix, higher priority than numbering suggests)

**Branch:** `fix/fallback-narrative-hazard-classification`

**Origin:** discovered during Track 2's codebase check, deliberately deferred out of that branch to keep it scoped to disclosure only.

**The bug:** `_fallback_narrative()` in `llm_synthesis.py` (the deterministic template used whenever the live LLM/API path is unavailable — which is the current default behavior in this environment, since no API key is set) decides whether to use "normal" or "anomaly" language based on the temperature classification only. It never reads `precip_z_score`, `wind_z_score`, or `drought_indicator`. Practical effect: if temperature is normal but precipitation or wind is independently flagged as notable/exceptional (e.g. a real flood or windstorm signal), the fallback narrative can say "conditions fall within normal operational parameters" — actively burying a real, already-detected hazard. This is a correctness issue, not a coverage gap: the live LLM path is instructed to mention flood/drought/wind qualitatively in Actionable Framing; the fallback path currently never does, in any of the four LPCA beats.

**Scope of the real fix (per Claude Code's assessment during Track 2):**
- Fix branch-selection logic to check all three hazard classifications (temp, precip/drought, wind), not temperature alone.
- Bring precip/wind/drought mentions into all four LPCA beats in the fallback template, at parity with the live LLM path.
- Add new test coverage — zero of the existing 38 `test_llm_synthesis.py` tests currently touch precip/wind/drought, so this isn't just editing existing tests.

**Why it's flagged as higher priority than its position in this list:** unlike the other deferred/v2 items, this is a silent-failure bug on the app's default runtime path (fallback template), not a missing feature. Worth considering for the next branch after Track 2 merges, ahead of Track 3, rather than waiting for its numeric turn.

---

## Suggested build order

1. ~~Track 1 (quick fix)~~ — **done, merged.**
2. ~~Track 2 (economic scope transparency)~~ — **done, merged.**
3. **Track 8 (fallback narrative hazard-classification bug)** — discovered during Track 2, correctness issue on the app's default runtime path. Recommend doing this next, ahead of Track 3, despite its numbering.
4. Track 3 — event timeline. Can run in parallel with Track 7 if desired.
5. Track 7 (research + site agents) — separate CI/automation concern.
6. Track 4 (user data override) — slow track, full grill-me spec pass required before any code.
7. Track 6 (Next.js rebuild, incl. Track 5's design system) — last, once backend/data model is stable.
