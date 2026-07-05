# ssvi — IV skew / SSVI scanner for NASDAQ-100 options

Daily scanner that pulls option chains from Polygon.io (massive.com),
fits an SSVI implied-volatility surface per underlying, and ranks:

- **Wheel entries** — short-dated (<45 DTE) cash-secured put sales,
  ranked by variance risk premium (30d ATM IV − 20d realized vol),
  25Δ put-skew richness, and IV rank.
- **LEAPS buys** — long-dated (>1y) call purchases, ranked by how cheap
  long-dated IV is vs. its own history and the term-structure slope.

## Setup

    python -m venv .venv && .venv/bin/pip install -e '.[dev]'
    export POLYGON_API_KEY=...   # Options Starter plan or better

Verify entitlements once: `python scripts/probe_api.py`

## Daily use (after market close; data is 15-min delayed)

    python -m ssvi scan --plots      # full universe, report + PNGs in plots/
    python -m ssvi scan --tickers NVDA,AAPL
    python -m ssvi plot NVDA         # re-plot from today's cache

IV rank / IV percentile columns are NaN until ~20 daily scans have
accumulated in `data/history/` — run the scan daily (cron it) to build
history.

## Data model (Options Starter has no quotes or IV)

A live probe against Options Starter showed the snapshot endpoint has
**no bid/ask, no greeks, and no `implied_volatility`** — only delayed
last-trade aggregates (`day.close`) and open interest. So this tool
computes its own numbers:

- **Forward per expiry** comes from put-call parity on trade-close
  prices (`bs.py` / `clean.py`), which bakes in dividends and borrow
  cost automatically — no separate dividend-yield estimate needed.
- **IV** is inverted from `day.close` via Black-76 (`ssvi/bs.py`,
  Brent's method), using a small hand-maintained risk-free rate curve
  (`config.RATE_CURVE`) instead of a rates feed Polygon doesn't provide
  at this tier.
- Contracts are filtered by `open_interest`, staleness of the last
  trade (`config.MAX_STALE_DAYS`), and minimum days-to-expiry
  (`config.MIN_DTE`, excludes near-0DTE noise irrelevant to both
  strategies).
- `arb_flags` (butterfly/calendar static-arbitrage checks on the fitted
  surface) is a **diagnostic column**, not an automatic veto: a single
  global 3-parameter SSVI fit routinely flags *some* tenor on
  richly-skewed single-stock smiles (SSVI was designed for smoother
  index surfaces like SPX). Check it before trusting a specific tenor,
  but it won't silently empty out every signal.

## Caveats

- Signals are relative-value screens, not trade instructions. Check
  earnings dates before selling short-dated puts.
- No backtest: history accumulates forward from your own scans.
- Realized vol is close-to-close over 20 days.
- The stock-aggregates endpoint may be rate-limited on lower Polygon
  tiers (~5 calls/min); if a ticker is skipped with a 429 after
  retries, just rerun `scan` — the per-day disk cache means
  already-successful tickers won't be re-fetched.
