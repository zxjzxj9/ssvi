# IV Skew / SSVI Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A daily CLI scanner that pulls option chains for the top-20 NASDAQ-100 stocks from Polygon.io, fits per-expiry SVI smiles and a global SSVI surface, computes skew/term-structure/variance-risk-premium metrics, stores daily history, and ranks candidates for two strategies: short-dated (<45 DTE) option selling (wheel entries) and long-dated (>1 year) option buying (LEAPS).

**Architecture:** A Python package `ssvi` with a layered pipeline: rate-limited Polygon client → chain snapshot fetcher → cleaning/forward extraction → per-slice SVI fit → global SSVI calibration (Gatheral–Jacquier power-law φ) → metrics → parquet history store → signal scoring → CLI report + PNG plots. Everything runs once daily on 15-min-delayed data (Options Starter plan), which is fine because trade frequency is ≥ 1 day.

**Tech Stack:** Python ≥3.11, httpx, pandas, numpy, scipy, pyarrow, matplotlib, pytest. No web framework, no database server — parquet files on disk.

## Global Constraints

- Polygon plan is **Options Starter**: 15-min delayed data, unlimited calls but be polite — client must retry on HTTP 429 with backoff.
- Stock daily bars (for realized vol) may be on a free/limited stocks entitlement: the client must survive 5-calls/min throttling via the same 429 backoff.
- API key comes from env var `POLYGON_API_KEY` only — never hardcoded, never committed.
- Universe: top-20 NASDAQ-100 by weight, editable static list in `config.py`.
- **Revised after live probe (Task 3):** Options Starter has no bid/ask quotes, no greeks, and no `implied_volatility` on the snapshot endpoint (confirmed empty/absent on both bulk and single-contract snapshot; `/v3/quotes/...` is 403). Only `day` trade aggregates (delayed close/high/low/volume/vwap) and `open_interest` are available. IV is therefore computed in-house: `bs.py` (Black-76, forward-based) inverts IV from `day.close`. Risk-free rate is `config.risk_free_rate(T)`, a small hand-maintained term-structure curve (`config.RATE_CURVE`), not a flat constant — used for parity-forward discounting and Black-76 pricing. Forwards are still extracted via put-call parity (Task 6/7), which bakes in dividends/borrow cost without a separate dividend-yield estimate. Contracts are filtered by `open_interest` and by staleness of `day.last_updated` (`config.MAX_STALE_DAYS`) instead of a bid/ask spread filter.
- All dates are timezone-naive dates in US-Eastern trading terms; year fraction is ACT/365.
- Data directories: `data/cache/` (raw API JSON, keyed by date+ticker), `data/history/` (parquet). Both gitignored.
- All money-losing-relevant math (parity forward, SVI, SSVI, deltas) must have tests that recover known synthetic inputs.
- Python package layout: `src/ssvi/`, tests in `tests/`. Run tests with `python -m pytest`.

## Strategy Logic (context for the implementer — read once)

The edge hypotheses this tool screens for:

1. **Short-dated selling (wheel entry via cash-secured puts, DTE ≤ 45):** equity index options persistently price implied vol above subsequently-realized vol (variance risk premium). We rank tickers by `VRP = 30d ATM IV − 20d realized vol`, by IV rank vs. our own stored history, and by put-skew richness (25Δ risk reversal). High values → selling ~25Δ puts is better paid.
2. **Long-dated buying (LEAPS calls, T ≥ 1y):** long-dated IV is sticky and mean-reverting; when the SSVI long end is cheap relative to its own history and to the short end, buying 0.70–0.80Δ LEAPS calls buys vega+delta cheaply. We rank by long-dated (1y interpolated) ATM IV percentile (low = good) and term-structure steepness.
3. The SSVI fit is what makes cross-expiry comparison honest: raw chain IVs at different strikes/expiries aren't comparable; the fitted surface gives consistent ATM IV, skew, and interpolated tenors, plus static-arbitrage sanity checks so we don't trade off a broken fit.

**Not in scope (deliberately):** backtesting (needs historical options data beyond Starter's practical reach — history accumulates forward from our daily snapshots), earnings calendars (manually check earnings dates before selling 45-DTE puts — the report prints a reminder), execution/order routing.

## File Structure

```
ssvi/
├── pyproject.toml
├── .gitignore
├── src/ssvi/
│   ├── __init__.py
│   ├── config.py          # API key, universe, constants, paths
│   ├── polygon_client.py  # rate-limited, paginating HTTP client + JSON disk cache
│   ├── chain.py           # option-chain snapshot → tidy DataFrame (trade prices, no quotes)
│   ├── stocks.py          # daily bars → realized vol
│   ├── bs.py              # Black-76 forward pricer + IV solver
│   ├── clean.py           # filters, parity forward, log-moneyness, total variance, IV inversion
│   ├── svi.py             # raw SVI slice fit
│   ├── ssvi.py            # SSVI surface fit + no-arbitrage checks
│   ├── metrics.py         # ATM term structure, 25Δ RR, VRP, interpolated tenors
│   ├── store.py           # parquet daily history + IV rank
│   ├── signals.py         # wheel & LEAPS candidate scoring
│   ├── plots.py           # smile / term-structure / surface PNGs
│   └── cli.py             # `python -m ssvi scan|plot`
├── scripts/
│   └── probe_api.py       # one-off: verify what the Starter key can access
└── tests/
    ├── fixtures/
    │   └── snapshot_page.json
    ├── test_config.py
    ├── test_polygon_client.py
    ├── test_chain.py
    ├── test_stocks.py
    ├── test_clean.py
    ├── test_svi.py
    ├── test_ssvi.py
    ├── test_metrics.py
    ├── test_store.py
    ├── test_signals.py
    ├── test_plots.py
    └── test_cli.py
```

---

### Task 1: Project scaffold and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/ssvi/__init__.py`, `src/ssvi/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.get_api_key() -> str` (raises `RuntimeError` if `POLYGON_API_KEY` unset), `config.UNIVERSE: list[str]`, `config.RISK_FREE_RATE: float`, `config.CACHE_DIR: Path`, `config.HISTORY_DIR: Path`, `config.PLOTS_DIR: Path`.

- [ ] **Step 1: Initialize repo and scaffold**

```bash
cd /Users/victor/Programs/ssvi
git init
mkdir -p src/ssvi tests scripts
```

`pyproject.toml`:

```toml
[project]
name = "ssvi"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pandas>=2.2",
    "numpy>=1.26",
    "scipy>=1.13",
    "pyarrow>=16",
    "matplotlib>=3.9",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:

```
__pycache__/
*.egg-info/
.venv/
data/
plots/
```

`src/ssvi/__init__.py`: empty file.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

(All subsequent `python`/`pytest` commands in this plan mean `.venv/bin/python` / `.venv/bin/python -m pytest`.)

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:

```python
import pytest

from ssvi import config


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-123")
    assert config.get_api_key() == "test-key-123"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="POLYGON_API_KEY"):
        config.get_api_key()


def test_universe_is_reasonable():
    assert 15 <= len(config.UNIVERSE) <= 25
    assert "NVDA" in config.UNIVERSE
    assert len(set(config.UNIVERSE)) == len(config.UNIVERSE)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ImportError` / `AttributeError` (config module missing).

- [ ] **Step 4: Write minimal implementation**

`src/ssvi/config.py`:

```python
import os
from pathlib import Path

# Top NASDAQ-100 constituents by weight. Edit freely; the pipeline
# treats this as the scan universe.
UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "AMZN", "AVGO", "META", "GOOGL", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "LIN", "CSCO", "TMUS", "INTU",
    "QCOM", "TXN", "AMGN", "ADBE",
]

RISK_FREE_RATE = 0.045  # flat rate used for parity forwards and BS deltas

BASE_URL = "https://api.polygon.io"

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "cache"
HISTORY_DIR = _ROOT / "data" / "history"
PLOTS_DIR = _ROOT / "plots"

# Liquidity filters applied to raw chains
MIN_OPEN_INTEREST = 10
MAX_REL_SPREAD = 0.25       # (ask-bid)/mid must be below this
SHORT_DTE_MAX = 45          # short-vol strategy horizon
LEAPS_T_MIN = 1.0           # years; long-vol strategy horizon


def get_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError(
            "POLYGON_API_KEY environment variable is not set. "
            "Export your massive.com/polygon.io key first."
        )
    return key
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/ssvi/__init__.py src/ssvi/config.py tests/test_config.py
git commit -m "feat: project scaffold and config"
```

---

### Task 2: Polygon HTTP client with backoff, pagination, and disk cache

**Files:**
- Create: `src/ssvi/polygon_client.py`
- Test: `tests/test_polygon_client.py`

**Interfaces:**
- Consumes: `config.get_api_key()`, `config.BASE_URL`, `config.CACHE_DIR`.
- Produces: `class PolygonClient` with:
  - `__init__(self, api_key: str | None = None, cache_date: str | None = None)` — `cache_date` like `"2026-07-05"`; when set, responses are cached to/served from `CACHE_DIR/<cache_date>/<sha1(url+params)>.json`.
  - `get_json(self, path: str, params: dict | None = None) -> dict` — single GET, retries HTTP 429/5xx up to 5 times with exponential backoff (1, 2, 4, 8, 16 s).
  - `get_paginated(self, path: str, params: dict | None = None) -> list[dict]` — follows Polygon `next_url` links, concatenating each page's `results` list.

- [ ] **Step 1: Write the failing test**

`tests/test_polygon_client.py`:

```python
import json

import httpx
import pytest

from ssvi.polygon_client import PolygonClient


def make_client(handler, **kwargs):
    client = PolygonClient(api_key="k", **kwargs)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_get_json_adds_api_key():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"status": "OK", "results": []})

    client = make_client(handler)
    out = client.get_json("/v3/things", {"limit": 5})
    assert out["status"] == "OK"
    assert "apiKey=k" in seen["url"]
    assert "limit=5" in seen["url"]


def test_retries_on_429(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"status": "OK"})

    monkeypatch.setattr("time.sleep", lambda s: None)
    client = make_client(handler)
    assert client.get_json("/v3/things")["status"] == "OK"
    assert calls["n"] == 3


def test_paginated_follows_next_url():
    def handler(request):
        if "cursor" in str(request.url):
            return httpx.Response(200, json={"results": [{"i": 2}]})
        return httpx.Response(
            200,
            json={
                "results": [{"i": 1}],
                "next_url": "https://api.polygon.io/v3/things?cursor=abc",
            },
        )

    client = make_client(handler)
    rows = client.get_paginated("/v3/things")
    assert [r["i"] for r in rows] == [1, 2]


def test_disk_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("ssvi.config.CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"status": "OK", "results": [1]})

    client = make_client(handler, cache_date="2026-07-05")
    first = client.get_json("/v3/things")
    second = client.get_json("/v3/things")  # must come from disk
    assert first == second
    assert calls["n"] == 1
    cached = list((tmp_path / "2026-07-05").glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text())["status"] == "OK"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_polygon_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.polygon_client'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/polygon_client.py`:

```python
import hashlib
import json
import time

import httpx

from ssvi import config


class PolygonClient:
    def __init__(self, api_key: str | None = None, cache_date: str | None = None):
        self.api_key = api_key or config.get_api_key()
        self.cache_date = cache_date
        self._http = httpx.Client(timeout=30.0)

    def _cache_path(self, url: str, params: dict):
        if self.cache_date is None:
            return None
        key = hashlib.sha1(
            (url + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        d = config.CACHE_DIR / self.cache_date
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _get(self, url: str, params: dict) -> dict:
        cache = self._cache_path(url, params)
        if cache is not None and cache.exists():
            return json.loads(cache.read_text())
        delay = 1.0
        for attempt in range(5):
            resp = self._http.get(url, params={**params, "apiKey": self.api_key})
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            if cache is not None:
                cache.write_text(json.dumps(data))
            return data
        resp.raise_for_status()
        raise RuntimeError(f"gave up after retries: {url}")

    def get_json(self, path: str, params: dict | None = None) -> dict:
        url = path if path.startswith("http") else config.BASE_URL + path
        return self._get(url, dict(params or {}))

    def get_paginated(self, path: str, params: dict | None = None) -> list[dict]:
        out: list[dict] = []
        data = self.get_json(path, params)
        out.extend(data.get("results", []))
        while data.get("next_url"):
            data = self.get_json(data["next_url"])
            out.extend(data.get("results", []))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_polygon_client.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/polygon_client.py tests/test_polygon_client.py
git commit -m "feat: polygon client with backoff, pagination, disk cache"
```

---

### Task 3: API entitlement probe script (manual, no test)

**Files:**
- Create: `scripts/probe_api.py`

**Interfaces:**
- Consumes: `PolygonClient.get_json`.
- Produces: nothing importable — a manual diagnostic. Run once with a real key before trusting the pipeline.

- [ ] **Step 1: Write the script**

`scripts/probe_api.py`:

```python
"""Verify what the Options Starter key can access. Run manually:

    POLYGON_API_KEY=... python scripts/probe_api.py
"""
from ssvi.polygon_client import PolygonClient

client = PolygonClient()

checks = [
    ("option chain snapshot (needs IV+greeks)",
     "/v3/snapshot/options/AAPL", {"limit": 10}),
    ("stock daily bars (for realized vol)",
     "/v2/aggs/ticker/AAPL/range/1/day/2026-05-01/2026-07-01", {"limit": 5}),
]

for label, path, params in checks:
    try:
        data = client.get_json(path, params)
        results = data.get("results", [])
        print(f"OK   {label}: {len(results)} results")
        if "snapshot" in path and results:
            r = results[0]
            has_iv = r.get("implied_volatility") is not None
            has_greeks = bool(r.get("greeks"))
            print(f"     implied_volatility present: {has_iv}, greeks present: {has_greeks}")
    except Exception as e:  # noqa: BLE001 - diagnostic script
        print(f"FAIL {label}: {e}")
```

- [ ] **Step 2: Run it with the real key and record the outcome**

Run: `POLYGON_API_KEY=<real key> python scripts/probe_api.py`
Expected: both lines print `OK`, and `implied_volatility present: True`.
**If IV/greeks are missing** on Starter: stop and flag to the user — Task 5 would then need a Black-Scholes IV inversion from mid prices (add `implied_vol_from_price` to `clean.py`); do not silently continue.
**If stock bars fail**: realized vol (Task 4) falls back to computing RV from the `underlying_asset.price` field captured daily in our own history — flag to the user either way.

- [ ] **Step 3: Commit**

```bash
git add scripts/probe_api.py
git commit -m "feat: API entitlement probe script"
```

---

### Task 4: Chain snapshot fetcher (REVISED after Task 3 probe)

**Files:**
- Create: `src/ssvi/chain.py`, `tests/fixtures/snapshot_page.json`
- Test: `tests/test_chain.py`

**Interfaces:**
- Consumes: `PolygonClient.get_paginated`.
- Produces: `chain.fetch_chain(client: PolygonClient, underlying: str) -> pd.DataFrame` with columns:
  `underlying (str), contract (str), type ('call'|'put'), strike (float), expiry (datetime64[ns]), price (float), last_updated (datetime64[ns]), open_interest (float), volume (float)`.
  `price` is `day.close` (last-trade close, 15-min delayed on Starter). `last_updated` is `day.last_updated` (Polygon reports this in **nanoseconds** since epoch) converted to a timestamp, used later for staleness filtering. Rows with missing/zero `price` or missing `expiration_date` are dropped here (structural completeness only; liquidity/staleness filtering happens in Task 6). There is no bid/ask, no greeks, no IV, and no spot price on this endpoint — spot comes from `stocks.py` (Task 5) instead.

- [ ] **Step 1: Create the fixture**

`tests/fixtures/snapshot_page.json` (real-shape Options Starter snapshot page, verified against a live probe):

```json
{
  "results": [
    {
      "details": {"ticker": "O:AAPL260618C00200000", "contract_type": "call",
                  "strike_price": 200.0, "expiration_date": "2026-06-18"},
      "day": {"close": 12.8, "volume": 320, "last_updated": 1751500000000000000},
      "greeks": {},
      "open_interest": 1500
    },
    {
      "details": {"ticker": "O:AAPL260618P00200000", "contract_type": "put",
                  "strike_price": 200.0, "expiration_date": "2026-06-18"},
      "day": {"close": 7.2, "volume": 150, "last_updated": 1751500000000000000},
      "greeks": {},
      "open_interest": 900
    },
    {
      "details": {"ticker": "O:AAPL260618C00300000", "contract_type": "call",
                  "strike_price": 300.0, "expiration_date": "2026-06-18"},
      "day": {},
      "greeks": {},
      "open_interest": 10
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_chain.py`:

```python
import json
from pathlib import Path

import pandas as pd

from ssvi.chain import fetch_chain

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "snapshot_page.json").read_text()
)


class FakeClient:
    def get_paginated(self, path, params=None):
        assert path == "/v3/snapshot/options/AAPL"
        assert params["limit"] == 250
        return FIXTURE["results"]


def test_fetch_chain_parses_and_drops_incomplete():
    df = fetch_chain(FakeClient(), "AAPL")
    # third row has no day.close -> dropped
    assert len(df) == 2
    row = df[df["type"] == "call"].iloc[0]
    assert row["strike"] == 200.0
    assert row["price"] == 12.8
    assert row["expiry"] == pd.Timestamp("2026-06-18")
    assert row["underlying"] == "AAPL"
    assert row["contract"] == "O:AAPL260618C00200000"
    assert row["open_interest"] == 1500
    assert row["last_updated"] == pd.Timestamp(1751500000000000000, unit="ns")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_chain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.chain'`.

- [ ] **Step 4: Write minimal implementation**

`src/ssvi/chain.py`:

```python
import pandas as pd


def fetch_chain(client, underlying: str) -> pd.DataFrame:
    """Full option-chain snapshot for one underlying, as a tidy DataFrame.

    Options Starter has no bid/ask, greeks, or IV on this endpoint -- only
    delayed trade aggregates. `price` is the last trade close; IV is
    computed downstream in clean.py via Black-76 inversion.
    """
    results = client.get_paginated(
        f"/v3/snapshot/options/{underlying}", {"limit": 250}
    )
    rows = []
    for r in results:
        det = r.get("details", {})
        day = r.get("day") or {}
        price = day.get("close")
        expiration = det.get("expiration_date")
        if not price or not expiration:
            continue
        rows.append({
            "underlying": underlying,
            "contract": det.get("ticker"),
            "type": det.get("contract_type"),
            "strike": float(det.get("strike_price")),
            "expiry": pd.Timestamp(expiration),
            "price": float(price),
            "last_updated": pd.Timestamp(day.get("last_updated"), unit="ns"),
            "open_interest": float(r.get("open_interest") or 0),
            "volume": float(day.get("volume") or 0),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_chain.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ssvi/chain.py tests/fixtures/snapshot_page.json tests/test_chain.py
git commit -m "feat: option chain snapshot fetcher (trade prices, no quotes)"
```

---

### Task 5: Stock bars and realized volatility

**Files:**
- Create: `src/ssvi/stocks.py`
- Test: `tests/test_stocks.py`

**Interfaces:**
- Consumes: `PolygonClient.get_paginated`.
- Produces:
  - `stocks.fetch_daily_closes(client, ticker: str, days: int = 90) -> pd.Series` — close prices indexed by date, ascending.
  - `stocks.realized_vol(closes: pd.Series, window: int = 20) -> float` — annualized close-to-close vol of the last `window` log returns: `std(log returns, ddof=1) * sqrt(252)`. Returns `float('nan')` if fewer than `window+1` closes.

- [ ] **Step 1: Write the failing test**

`tests/test_stocks.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ssvi.stocks import fetch_daily_closes, realized_vol


class FakeClient:
    def get_paginated(self, path, params=None):
        assert path.startswith("/v2/aggs/ticker/AAPL/range/1/day/")
        return [
            {"t": 1750000000000, "c": 100.0},
            {"t": 1750086400000, "c": 101.0},
            {"t": 1750172800000, "c": 102.0},
        ]


def test_fetch_daily_closes():
    s = fetch_daily_closes(FakeClient(), "AAPL", days=10)
    assert list(s.values) == [100.0, 101.0, 102.0]
    assert s.index.is_monotonic_increasing


def test_realized_vol_of_known_series():
    # alternating +1%/-1% log returns -> std is exactly 0.01 * correction
    rets = np.array([0.01, -0.01] * 10)
    closes = pd.Series(100 * np.exp(np.cumsum(np.insert(rets, 0, 0.0))))
    rv = realized_vol(closes, window=20)
    expected = np.std(rets[-20:], ddof=1) * np.sqrt(252)
    assert rv == pytest.approx(expected, rel=1e-9)


def test_realized_vol_insufficient_data():
    assert np.isnan(realized_vol(pd.Series([1.0, 2.0]), window=20))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stocks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.stocks'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/stocks.py`:

```python
from datetime import date, timedelta

import numpy as np
import pandas as pd


def fetch_daily_closes(client, ticker: str, days: int = 90) -> pd.Series:
    end = date.today()
    start = end - timedelta(days=days)
    results = client.get_paginated(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        {"adjusted": "true", "sort": "asc", "limit": 5000},
    )
    idx = pd.to_datetime([r["t"] for r in results], unit="ms")
    return pd.Series([r["c"] for r in results], index=idx).sort_index()


def realized_vol(closes: pd.Series, window: int = 20) -> float:
    if len(closes) < window + 1:
        return float("nan")
    rets = np.diff(np.log(closes.values))[-window:]
    return float(np.std(rets, ddof=1) * np.sqrt(252))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stocks.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/stocks.py tests/test_stocks.py
git commit -m "feat: stock daily closes and realized vol"
```

---

### Task 6: Cleaning, parity forwards, log-moneyness, total variance (REVISED after Task 3 probe)

**Files:**
- Create: `src/ssvi/clean.py`
- Test: `tests/test_clean.py`

**Interfaces:**
- Consumes: chain DataFrame from `chain.fetch_chain` (Task 4 revised column schema: `price`, `last_updated`, no bid/ask/iv), `config.risk_free_rate(T)`, `config.MIN_OPEN_INTEREST`, `config.MAX_STALE_DAYS`, `bs.implied_vol` (Task 3b).
- Produces: `clean.prepare(chain_df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame` — filtered to liquid OTM options, with added columns:
  - `T (float)` — ACT/365 year fraction from `asof` to expiry; rows with `T <= 1/365` dropped.
  - `forward (float)` — per-expiry forward from put–call parity on trade-close prices, at the strike where `|call_price − put_price|` is smallest: `F = K + exp(r(T)·T)·(C_price − P_price)`. Expiries lacking a call/put pair are dropped. This forward already reflects the market's dividend/borrow cost — no separate dividend-yield estimate is needed.
  - `k (float)` — log-moneyness `ln(strike / forward)`.
  - `iv (float)` — Black-76 implied vol from `bs.implied_vol(price, forward, strike, T, r(T), type)`. Rows where the trade price violates the no-arbitrage intrinsic-value bound (raises `ValueError`) are dropped.
  - `w (float)` — total implied variance `iv² · T`.
  - Keeps only OTM rows: puts with `strike <= forward`, calls with `strike >= forward`.
  Filters: `price > 0`, `open_interest >= MIN_OPEN_INTEREST`, `last_updated >= asof - MAX_STALE_DAYS days` (drops contracts that haven't traded recently — no bid/ask spread to filter on anymore).

- [ ] **Step 1: Write the failing test**

`tests/test_clean.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ssvi.bs import black76_price
from ssvi.clean import prepare
from ssvi import config

ASOF = pd.Timestamp("2026-07-05")
EXPIRY = pd.Timestamp("2027-07-05")  # T ~ 1.0
FRESH = ASOF - pd.Timedelta(hours=1)


def make_row(**kw):
    base = dict(
        underlying="TEST", contract="X", type="call", strike=100.0,
        expiry=EXPIRY, price=10.0, last_updated=FRESH,
        open_interest=100, volume=10,
    )
    base.update(kw)
    return base


def parity_pair(strike, forward, sigma=0.30, T=1.0):
    """Call/put trade prices consistent with a given forward via Black-76."""
    r = config.risk_free_rate(T)
    call_price = black76_price(forward, strike, T, r, sigma, "call")
    put_price = black76_price(forward, strike, T, r, sigma, "put")
    return (
        make_row(type="call", strike=strike, price=call_price),
        make_row(type="put", strike=strike, price=put_price),
    )


def test_forward_from_parity_and_derived_columns():
    c, p = parity_pair(strike=100.0, forward=103.0, sigma=0.30)
    df = prepare(pd.DataFrame([c, p]), asof=ASOF)
    assert df["forward"].iloc[0] == pytest.approx(103.0, rel=1e-4)
    put = df[df["type"] == "put"].iloc[0]      # put at K=100 < F -> OTM, kept
    assert put["k"] == pytest.approx(np.log(100.0 / 103.0), rel=1e-6)
    assert put["iv"] == pytest.approx(0.30, rel=1e-3)
    assert put["w"] == pytest.approx(0.30**2 * 1.0, rel=1e-2)
    assert put["T"] == pytest.approx(1.0, abs=0.01)
    assert (df[df["type"] == "call"]["strike"] >= 103.0).all() or \
           df[df["type"] == "call"].empty  # ITM call at 100 dropped


def test_liquidity_and_staleness_filters():
    c, p = parity_pair(strike=100.0, forward=100.0)
    bad_oi = make_row(strike=110.0, price=5.0, open_interest=1)
    stale = make_row(strike=115.0, price=5.0,
                     last_updated=ASOF - pd.Timedelta(days=30))
    no_price = make_row(strike=120.0, price=0.0)
    df = prepare(pd.DataFrame([c, p, bad_oi, stale, no_price]), asof=ASOF)
    assert not (df["strike"] >= 110.0).any()


def test_expired_and_unpaired_expiries_dropped():
    lonely_call = make_row(expiry=pd.Timestamp("2027-08-20"))
    expired = make_row(expiry=ASOF)
    df = prepare(pd.DataFrame([lonely_call, expired]), asof=ASOF)
    assert df.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.clean'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/clean.py`:

```python
import numpy as np
import pandas as pd

from ssvi import bs, config


def _forward_for_expiry(g: pd.DataFrame, T: float, r: float) -> float | None:
    calls = g[g["type"] == "call"].set_index("strike")["price"]
    puts = g[g["type"] == "put"].set_index("strike")["price"]
    common = calls.index.intersection(puts.index)
    if common.empty:
        return None
    diff = (calls[common] - puts[common]).abs()
    k0 = diff.idxmin()
    return float(k0 + np.exp(r * T) * (calls[k0] - puts[k0]))


def prepare(chain_df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    df = chain_df.copy()
    stale_cutoff = asof - pd.Timedelta(days=config.MAX_STALE_DAYS)
    df = df[
        (df["price"] > 0)
        & (df["open_interest"] >= config.MIN_OPEN_INTEREST)
        & (df["last_updated"] >= stale_cutoff)
    ]
    df["T"] = (df["expiry"] - asof).dt.days / 365.0
    df = df[df["T"] > 1 / 365]

    out = []
    for expiry, g in df.groupby("expiry"):
        T = float(g["T"].iloc[0])
        r = config.risk_free_rate(T)
        fwd = _forward_for_expiry(g, T, r)
        if fwd is None or fwd <= 0:
            continue
        g = g.copy()
        g["forward"] = fwd
        g["k"] = np.log(g["strike"] / fwd)
        otm = ((g["type"] == "put") & (g["strike"] <= fwd)) | (
            (g["type"] == "call") & (g["strike"] >= fwd)
        )
        g = g[otm]

        ivs, keep_idx = [], []
        for idx, row in g.iterrows():
            try:
                iv = bs.implied_vol(row["price"], fwd, row["strike"], T, r,
                                    row["type"])
            except ValueError:
                continue
            ivs.append(iv)
            keep_idx.append(idx)
        g = g.loc[keep_idx].copy()
        g["iv"] = ivs
        g["w"] = g["iv"] ** 2 * g["T"]
        out.append(g)
    if not out:
        return df.iloc[0:0].assign(forward=np.nan, k=np.nan, iv=np.nan, w=np.nan)
    return pd.concat(out, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clean.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/clean.py tests/test_clean.py
git commit -m "feat: chain cleaning, parity forwards from trade prices, Black-76 IV inversion"
```

---

### Task 7: Raw SVI slice fit

**Files:**
- Create: `src/ssvi/svi.py`
- Test: `tests/test_svi.py`

**Interfaces:**
- Consumes: prepared per-expiry data (`k`, `w` arrays).
- Produces:
  - `svi.svi_total_variance(k: np.ndarray, a, b, rho, m, sigma) -> np.ndarray` — raw SVI: `w(k) = a + b*(rho*(k-m) + sqrt((k-m)**2 + sigma**2))`.
  - `svi.fit_slice(k: np.ndarray, w: np.ndarray) -> dict` — keys `a, b, rho, m, sigma, rmse` (rmse in total-variance units). Uses `scipy.optimize.least_squares` with bounds `b >= 0`, `-0.999 <= rho <= 0.999`, `sigma >= 1e-4`, `a >= -max(w)` (loose), multi-start from 3 initial guesses, best RMSE wins. Raises `ValueError` if fewer than 5 points.
  - `svi.atm_total_variance(params: dict) -> float` — `svi_total_variance(0.0, ...)`, the slice θ used by SSVI.

- [ ] **Step 1: Write the failing test**

`tests/test_svi.py`:

```python
import numpy as np
import pytest

from ssvi.svi import atm_total_variance, fit_slice, svi_total_variance

TRUE = dict(a=0.02, b=0.4, rho=-0.6, m=0.05, sigma=0.2)


def test_svi_function_shape():
    w = svi_total_variance(np.array([0.0]), **TRUE)
    expected = 0.02 + 0.4 * (-0.6 * -0.05 + np.sqrt(0.05**2 + 0.2**2))
    assert w[0] == pytest.approx(expected, rel=1e-12)


def test_fit_recovers_known_params():
    k = np.linspace(-0.5, 0.5, 25)
    w = svi_total_variance(k, **TRUE)
    fit = fit_slice(k, w)
    refit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"],
                               fit["m"], fit["sigma"])
    assert fit["rmse"] < 1e-6
    np.testing.assert_allclose(refit, w, atol=1e-5)
    assert atm_total_variance(fit) == pytest.approx(
        svi_total_variance(np.array([0.0]), **TRUE)[0], rel=1e-4
    )


def test_fit_noisy_data_reasonable():
    rng = np.random.default_rng(42)
    k = np.linspace(-0.4, 0.4, 30)
    w = svi_total_variance(k, **TRUE) + rng.normal(0, 0.0005, 30)
    fit = fit_slice(k, w)
    assert fit["rmse"] < 0.002
    assert -1 < fit["rho"] < 1


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_slice(np.array([0.0, 0.1]), np.array([0.04, 0.05]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_svi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.svi'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/svi.py`:

```python
import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def fit_slice(k: np.ndarray, w: np.ndarray) -> dict:
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(k) < 5:
        raise ValueError(f"need >=5 points to fit SVI slice, got {len(k)}")

    w_max = float(np.max(w))

    def residuals(p):
        a, b, rho, m, sigma = p
        return svi_total_variance(k, a, b, rho, m, sigma) - w

    lb = [-w_max, 0.0, -0.999, -1.0, 1e-4]
    ub = [w_max * 2, 5.0, 0.999, 1.0, 2.0]
    starts = [
        [np.min(w) * 0.5, 0.1, -0.5, 0.0, 0.1],
        [np.min(w) * 0.9, 0.5, 0.0, 0.1, 0.3],
        [0.0, 0.3, -0.8, -0.1, 0.2],
    ]
    best = None
    for x0 in starts:
        try:
            res = least_squares(residuals, x0, bounds=(lb, ub))
        except ValueError:
            continue
        rmse = float(np.sqrt(np.mean(res.fun**2)))
        if best is None or rmse < best[1]:
            best = (res.x, rmse)
    a, b, rho, m, sigma = best[0]
    return {"a": float(a), "b": float(b), "rho": float(rho),
            "m": float(m), "sigma": float(sigma), "rmse": best[1]}


def atm_total_variance(params: dict) -> float:
    return float(svi_total_variance(
        np.array([0.0]), params["a"], params["b"], params["rho"],
        params["m"], params["sigma"],
    )[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_svi.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/svi.py tests/test_svi.py
git commit -m "feat: raw SVI slice calibration"
```

---

### Task 8: SSVI global surface fit with no-arbitrage checks

**Files:**
- Create: `src/ssvi/ssvi.py`
- Test: `tests/test_ssvi.py`

**Interfaces:**
- Consumes: prepared DataFrame (columns `T`, `k`, `w`) and per-slice θ from `svi.fit_slice`/`svi.atm_total_variance`.
- Produces:
  - `ssvi.ssvi_total_variance(k, theta, rho, eta, gamma) -> np.ndarray` — Gatheral–Jacquier SSVI with power-law φ:
    `phi = eta / (theta**gamma * (1 + theta)**(1 - gamma))`;
    `w = 0.5 * theta * (1 + rho*phi*k + sqrt((phi*k + rho)**2 + 1 - rho**2))`.
  - `class SSVISurface` with fields `rho, eta, gamma: float`, `thetas: dict[float, float]` (T → θ), `rmse: float`, and methods:
    - `theta_at(self, T: float) -> float` — monotone linear interpolation of θ in T (via `np.interp` on sorted knots; θ is clipped to be nondecreasing before interpolation; extrapolation is flat-slope linear using the last two knots, floored at the last θ).
    - `w(self, k, T) -> np.ndarray` — surface total variance.
    - `iv(self, k, T) -> np.ndarray` — `sqrt(w / T)`.
    - `check_arbitrage(self) -> list[str]` — empty list if clean; else messages. Checks per θ knot: butterfly conditions `theta*phi*(1+|rho|) <= 4` and `theta*phi**2*(1+|rho|) <= 4`; calendar condition: raw slice θs nondecreasing in T (before clipping).
  - `ssvi.fit_surface(prepared: pd.DataFrame) -> SSVISurface` — computes θ per expiry via `svi.fit_slice` on each slice with ≥5 points (slices with fewer are skipped), then fits `(rho, eta, gamma)` by `least_squares` over all points, bounds `-0.999<=rho<=0.999`, `0.01<=eta<=10`, `0.01<=gamma<=0.99`. Raises `ValueError` if fewer than 2 usable slices.

- [ ] **Step 1: Write the failing test**

`tests/test_ssvi.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ssvi.ssvi import SSVISurface, fit_surface, ssvi_total_variance

RHO, ETA, GAMMA = -0.55, 1.2, 0.45
THETAS = {0.1: 0.009, 0.25: 0.02, 0.5: 0.038, 1.0: 0.07, 2.0: 0.13}


def synthetic_surface() -> pd.DataFrame:
    rows = []
    for T, theta in THETAS.items():
        k = np.linspace(-0.4, 0.4, 21)
        w = ssvi_total_variance(k, theta, RHO, ETA, GAMMA)
        for ki, wi in zip(k, w):
            rows.append({"T": T, "k": ki, "w": wi})
    return pd.DataFrame(rows)


def test_ssvi_atm_equals_theta():
    w0 = ssvi_total_variance(np.array([0.0]), 0.04, RHO, ETA, GAMMA)
    assert w0[0] == pytest.approx(0.04, rel=1e-12)


def test_fit_recovers_global_params():
    surf = fit_surface(synthetic_surface())
    assert surf.rho == pytest.approx(RHO, abs=0.05)
    assert surf.rmse < 1e-4
    # surface reproduces inputs at an interior tenor
    k = np.array([-0.2, 0.0, 0.2])
    np.testing.assert_allclose(
        surf.w(k, 0.5),
        ssvi_total_variance(k, THETAS[0.5], RHO, ETA, GAMMA),
        atol=5e-4,
    )


def test_theta_interpolation_and_iv():
    surf = fit_surface(synthetic_surface())
    t_mid = surf.theta_at(0.75)
    assert THETAS[0.5] < t_mid < THETAS[1.0]
    iv_atm = float(surf.iv(np.array([0.0]), 1.0)[0])
    assert iv_atm == pytest.approx(np.sqrt(surf.theta_at(1.0) / 1.0), rel=1e-6)


def test_arbitrage_check_flags_calendar_violation():
    surf = SSVISurface(rho=-0.5, eta=1.0, gamma=0.5,
                       thetas={0.5: 0.05, 1.0: 0.03}, rmse=0.0)
    msgs = surf.check_arbitrage()
    assert any("calendar" in m.lower() for m in msgs)


def test_arbitrage_check_clean_surface():
    surf = fit_surface(synthetic_surface())
    assert surf.check_arbitrage() == []


def test_too_few_slices_raises():
    df = synthetic_surface()
    with pytest.raises(ValueError):
        fit_surface(df[df["T"] == 0.5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ssvi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.ssvi'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/ssvi.py`:

```python
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ssvi import svi


def _phi(theta, eta, gamma):
    return eta / (theta**gamma * (1 + theta) ** (1 - gamma))


def ssvi_total_variance(k, theta, rho, eta, gamma):
    k = np.asarray(k, dtype=float)
    p = _phi(theta, eta, gamma)
    return 0.5 * theta * (
        1 + rho * p * k + np.sqrt((p * k + rho) ** 2 + 1 - rho**2)
    )


@dataclass
class SSVISurface:
    rho: float
    eta: float
    gamma: float
    thetas: dict[float, float]   # T -> theta (raw slice ATM total variance)
    rmse: float
    _ts: np.ndarray = field(init=False, repr=False)
    _th: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        ts = np.array(sorted(self.thetas))
        th = np.array([self.thetas[t] for t in ts])
        self._ts = ts
        self._th = np.maximum.accumulate(th)  # enforce calendar monotonicity

    def theta_at(self, T: float) -> float:
        ts, th = self._ts, self._th
        if T <= ts[-1]:
            return float(np.interp(T, ts, th))
        if len(ts) >= 2:  # linear extrapolation, floored at last knot
            slope = (th[-1] - th[-2]) / (ts[-1] - ts[-2])
            return float(max(th[-1], th[-1] + slope * (T - ts[-1])))
        return float(th[-1])

    def w(self, k, T: float) -> np.ndarray:
        return ssvi_total_variance(k, self.theta_at(T), self.rho,
                                   self.eta, self.gamma)

    def iv(self, k, T: float) -> np.ndarray:
        return np.sqrt(self.w(k, T) / T)

    def check_arbitrage(self) -> list[str]:
        msgs = []
        raw = np.array([self.thetas[t] for t in sorted(self.thetas)])
        if np.any(np.diff(raw) < -1e-8):
            msgs.append("calendar arbitrage: slice ATM total variance "
                        "decreases with maturity")
        cap = 4.0 * (1 + 1e-9)
        for t in sorted(self.thetas):
            theta = self.thetas[t]
            p = _phi(theta, self.eta, self.gamma)
            if theta * p * (1 + abs(self.rho)) > cap:
                msgs.append(f"butterfly condition 1 violated at T={t:.3f}")
            if theta * p**2 * (1 + abs(self.rho)) > cap:
                msgs.append(f"butterfly condition 2 violated at T={t:.3f}")
        return msgs


def fit_surface(prepared: pd.DataFrame) -> SSVISurface:
    thetas: dict[float, float] = {}
    for T, g in prepared.groupby("T"):
        if len(g) < 5:
            continue
        params = svi.fit_slice(g["k"].values, g["w"].values)
        thetas[float(T)] = svi.atm_total_variance(params)
    if len(thetas) < 2:
        raise ValueError(f"need >=2 usable expiries, got {len(thetas)}")

    pts = prepared[prepared["T"].isin(thetas)]
    k = pts["k"].values
    w = pts["w"].values
    theta_row = pts["T"].map(thetas).values

    def residuals(p):
        rho, eta, gamma = p
        return ssvi_total_variance(k, theta_row, rho, eta, gamma) - w

    res = least_squares(
        residuals, x0=[-0.5, 1.0, 0.5],
        bounds=([-0.999, 0.01, 0.01], [0.999, 10.0, 0.99]),
    )
    rho, eta, gamma = res.x
    rmse = float(np.sqrt(np.mean(res.fun**2)))
    return SSVISurface(rho=float(rho), eta=float(eta), gamma=float(gamma),
                       thetas=thetas, rmse=rmse)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ssvi.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/ssvi.py tests/test_ssvi.py
git commit -m "feat: SSVI surface calibration with arbitrage checks"
```

---

### Task 9: Metrics — ATM term structure, 25Δ risk reversal, VRP

**Files:**
- Create: `src/ssvi/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `SSVISurface` (methods `iv`, `w`, `theta_at`), `stocks.realized_vol` output.
- Produces:
  - `metrics.atm_iv(surface: SSVISurface, T: float) -> float` — `sqrt(theta_at(T)/T)`.
  - `metrics.delta_k(surface, T, target_delta: float) -> float` — log-moneyness where forward Black-Scholes delta hits `target_delta`. Call delta `N(d1)` with `d1 = -k/sqrt(w(k,T)) + sqrt(w(k,T))/2`; for puts, `target_delta` is negative (put delta = `N(d1) - 1`). Solved with `scipy.optimize.brentq` on `k ∈ [-2, 2]`.
  - `metrics.rr25(surface, T) -> float` — 25Δ risk reversal `iv(k_25dput) − iv(k_25dcall)` (positive = puts rich, normal equity skew).
  - `metrics.atm_skew(surface, T) -> float` — numerical `d(iv)/dk` at `k=0` (central difference, `h=0.01`).
  - `metrics.ticker_metrics(surface, rv20: float, spot: float) -> dict` — keys:
    `iv30, iv1y, rr25_30d, rr25_1y, atm_skew_30d, term_slope (iv1y − iv30), vrp (iv30 − rv20), rv20, spot, ssvi_rho, ssvi_rmse, arb_flags (';'.join of check_arbitrage())`. `iv30` uses `T=30/365`, `iv1y` uses `T=1.0`.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:

```python
import numpy as np
import pytest

from ssvi.metrics import atm_iv, atm_skew, delta_k, rr25, ticker_metrics
from ssvi.ssvi import SSVISurface


def flat_surface(vol=0.30):
    # rho=0 and tiny eta -> essentially flat smile at each tenor
    thetas = {T: vol**2 * T for T in (0.1, 0.5, 1.0, 2.0)}
    return SSVISurface(rho=0.0, eta=0.01, gamma=0.5, thetas=thetas, rmse=0.0)


def skewed_surface():
    thetas = {T: 0.09 * T for T in (0.1, 0.5, 1.0, 2.0)}
    return SSVISurface(rho=-0.6, eta=1.5, gamma=0.45, thetas=thetas, rmse=0.0)


def test_atm_iv_flat():
    assert atm_iv(flat_surface(0.30), 0.5) == pytest.approx(0.30, rel=1e-6)


def test_delta_k_flat_surface_symmetry():
    surf = flat_surface(0.30)
    k_call = delta_k(surf, T=0.25, target_delta=0.25)
    k_put = delta_k(surf, T=0.25, target_delta=-0.25)
    assert k_call > 0 > k_put
    # flat vol: 25d call and put are symmetric around w/2 shift
    w = 0.30**2 * 0.25
    from scipy.stats import norm
    d1 = -k_call / np.sqrt(w) + np.sqrt(w) / 2
    assert norm.cdf(d1) == pytest.approx(0.25, abs=1e-6)


def test_rr25_zero_on_flat_positive_on_skewed():
    assert rr25(flat_surface(), 0.25) == pytest.approx(0.0, abs=1e-3)
    assert rr25(skewed_surface(), 0.25) > 0.01  # rho<0 -> puts rich


def test_atm_skew_sign():
    assert abs(atm_skew(flat_surface(), 0.25)) < 1e-3
    assert atm_skew(skewed_surface(), 0.25) < 0


def test_ticker_metrics_keys_and_vrp():
    m = ticker_metrics(flat_surface(0.30), rv20=0.22, spot=100.0)
    assert m["vrp"] == pytest.approx(0.30 - 0.22, abs=1e-3)
    assert m["iv30"] == pytest.approx(0.30, rel=1e-3)
    assert m["iv1y"] == pytest.approx(0.30, rel=1e-3)
    assert m["term_slope"] == pytest.approx(0.0, abs=1e-3)
    assert m["arb_flags"] == ""
    for key in ("rr25_30d", "rr25_1y", "atm_skew_30d", "ssvi_rho",
                "ssvi_rmse", "rv20", "spot"):
        assert key in m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.metrics'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/metrics.py`:

```python
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ssvi.ssvi import SSVISurface


def atm_iv(surface: SSVISurface, T: float) -> float:
    return float(np.sqrt(surface.theta_at(T) / T))


def _call_delta(surface: SSVISurface, k: float, T: float) -> float:
    w = float(surface.w(np.array([k]), T)[0])
    d1 = -k / np.sqrt(w) + np.sqrt(w) / 2
    return float(norm.cdf(d1))


def delta_k(surface: SSVISurface, T: float, target_delta: float) -> float:
    if target_delta > 0:  # call
        f = lambda k: _call_delta(surface, k, T) - target_delta
    else:  # put: delta = N(d1) - 1
        f = lambda k: (_call_delta(surface, k, T) - 1) - target_delta
    return float(brentq(f, -2.0, 2.0, xtol=1e-8))


def rr25(surface: SSVISurface, T: float) -> float:
    k_put = delta_k(surface, T, -0.25)
    k_call = delta_k(surface, T, 0.25)
    iv_put = float(surface.iv(np.array([k_put]), T)[0])
    iv_call = float(surface.iv(np.array([k_call]), T)[0])
    return iv_put - iv_call


def atm_skew(surface: SSVISurface, T: float, h: float = 0.01) -> float:
    iv = surface.iv(np.array([-h, h]), T)
    return float((iv[1] - iv[0]) / (2 * h))


def ticker_metrics(surface: SSVISurface, rv20: float, spot: float) -> dict:
    T30 = 30 / 365
    iv30 = atm_iv(surface, T30)
    iv1y = atm_iv(surface, 1.0)
    return {
        "iv30": iv30,
        "iv1y": iv1y,
        "rr25_30d": rr25(surface, T30),
        "rr25_1y": rr25(surface, 1.0),
        "atm_skew_30d": atm_skew(surface, T30),
        "term_slope": iv1y - iv30,
        "vrp": iv30 - rv20,
        "rv20": rv20,
        "spot": spot,
        "ssvi_rho": surface.rho,
        "ssvi_rmse": surface.rmse,
        "arb_flags": ";".join(surface.check_arbitrage()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/metrics.py tests/test_metrics.py
git commit -m "feat: surface metrics (ATM term structure, 25d RR, VRP)"
```

---

### Task 10: History store and IV rank

**Files:**
- Create: `src/ssvi/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: metrics dicts from `metrics.ticker_metrics`, `config.HISTORY_DIR`.
- Produces:
  - `store.save_metrics(rows: list[dict], asof: str) -> None` — each row must include `underlying`; writes `HISTORY_DIR/metrics/<asof>.parquet` (overwrites same-day reruns) with an added `date` column.
  - `store.load_history() -> pd.DataFrame` — concatenation of all daily parquet files, empty DataFrame (with no rows) if none.
  - `store.iv_rank(history: pd.DataFrame, underlying: str, current_iv30: float, min_obs: int = 20) -> float` — percentile (0–100) of `current_iv30` within that underlying's historical `iv30` values; `float('nan')` if fewer than `min_obs` observations. Percentile = `100 * mean(historical <= current)`.

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ssvi import store


@pytest.fixture(autouse=True)
def tmp_history(tmp_path, monkeypatch):
    monkeypatch.setattr("ssvi.config.HISTORY_DIR", tmp_path)
    return tmp_path


def test_save_and_load_roundtrip():
    rows = [{"underlying": "NVDA", "iv30": 0.45, "vrp": 0.08}]
    store.save_metrics(rows, asof="2026-07-05")
    hist = store.load_history()
    assert len(hist) == 1
    assert hist.iloc[0]["underlying"] == "NVDA"
    assert str(hist.iloc[0]["date"]) == "2026-07-05"


def test_same_day_rerun_overwrites():
    store.save_metrics([{"underlying": "NVDA", "iv30": 0.45}], "2026-07-05")
    store.save_metrics([{"underlying": "NVDA", "iv30": 0.50}], "2026-07-05")
    hist = store.load_history()
    assert len(hist) == 1
    assert hist.iloc[0]["iv30"] == 0.50


def test_load_history_empty():
    assert store.load_history().empty


def test_iv_rank():
    dates = [f"2026-06-{d:02d}" for d in range(1, 26)]
    hist = pd.DataFrame({
        "underlying": ["NVDA"] * 25,
        "date": dates,
        "iv30": np.linspace(0.30, 0.54, 25),  # 25 obs
    })
    rank = store.iv_rank(hist, "NVDA", current_iv30=0.54)
    assert rank == pytest.approx(100.0)
    assert store.iv_rank(hist, "NVDA", 0.29) == pytest.approx(0.0)
    mid = store.iv_rank(hist, "NVDA", 0.42)
    assert 40 <= mid <= 60


def test_iv_rank_insufficient_history():
    hist = pd.DataFrame({"underlying": ["NVDA"] * 5,
                         "date": ["2026-07-01"] * 5,
                         "iv30": [0.4] * 5})
    assert np.isnan(store.iv_rank(hist, "NVDA", 0.45, min_obs=20))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL with `AttributeError` / `ModuleNotFoundError` for `ssvi.store`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/store.py`:

```python
import numpy as np
import pandas as pd

from ssvi import config


def _metrics_dir():
    d = config.HISTORY_DIR / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_metrics(rows: list[dict], asof: str) -> None:
    df = pd.DataFrame(rows)
    df["date"] = asof
    df.to_parquet(_metrics_dir() / f"{asof}.parquet", index=False)


def load_history() -> pd.DataFrame:
    files = sorted(_metrics_dir().glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def iv_rank(history: pd.DataFrame, underlying: str, current_iv30: float,
            min_obs: int = 20) -> float:
    if history.empty or "iv30" not in history.columns:
        return float("nan")
    vals = history.loc[history["underlying"] == underlying, "iv30"].dropna()
    if len(vals) < min_obs:
        return float("nan")
    return float(100.0 * np.mean(vals <= current_iv30))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/store.py tests/test_store.py
git commit -m "feat: parquet history store and IV rank"
```

---

### Task 11: Signal scoring — wheel entries and LEAPS buys

**Files:**
- Create: `src/ssvi/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: per-ticker metrics dict (Task 9 keys) augmented with `underlying` and `iv_rank` (Task 10); `SSVISurface` + `metrics.delta_k` for strike suggestions; `config.SHORT_DTE_MAX`, `config.LEAPS_T_MIN`.
- Produces:
  - `signals.score_wheel(m: dict) -> float` — short-vol attractiveness:
    `score = 100 * m["vrp"] + 50 * m["rr25_30d"] + 0.2 * (m["iv_rank"] if not NaN else 50.0)`; returns `float('-inf')` if `m["arb_flags"]` is non-empty or `m["vrp"] <= 0` (no premium → no trade).
  - `signals.score_leaps(m: dict) -> float` — long-vol attractiveness:
    `score = 0.5 * (100 − iv1y_percentile) + 100 * max(m["term_slope"], 0)` where `iv1y_percentile = m["iv1y_rank"]` (same rank mechanics as iv_rank but on `iv1y`; NaN → 50.0); returns `float('-inf')` if `m["arb_flags"]` non-empty.
  - `signals.suggest_strikes(surface: SSVISurface, spot: float) -> dict` — keys:
    `wheel_put_strike` — spot × exp(k) at `delta_k(surface, T=30/365, target_delta=-0.25)`, rounded to nearest 2.5;
    `leaps_call_strike` — spot × exp(k) at `delta_k(surface, T=1.25, target_delta=0.75)`, rounded to nearest 2.5.
  - `signals.build_report(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]` — `(wheel_df, leaps_df)`, each sorted by score descending, `-inf` rows dropped, columns: `underlying, score, iv30, iv1y, vrp, rr25_30d, iv_rank, term_slope, spot, wheel_put_strike` (wheel) / `leaps_call_strike` (leaps).

- [ ] **Step 1: Write the failing test**

`tests/test_signals.py`:

```python
import math

import pytest

from ssvi.signals import build_report, score_leaps, score_wheel, suggest_strikes
from ssvi.ssvi import SSVISurface


def base_metrics(**kw):
    m = dict(
        underlying="NVDA", iv30=0.45, iv1y=0.40, vrp=0.08, rr25_30d=0.03,
        rr25_1y=0.02, atm_skew_30d=-0.1, term_slope=-0.05, rv20=0.37,
        spot=1000.0, iv_rank=70.0, iv1y_rank=20.0, ssvi_rho=-0.6,
        ssvi_rmse=0.001, arb_flags="",
    )
    m.update(kw)
    return m


def test_wheel_score_ordering():
    rich = base_metrics(vrp=0.10, rr25_30d=0.05, iv_rank=90.0)
    poor = base_metrics(vrp=0.01, rr25_30d=0.00, iv_rank=10.0)
    assert score_wheel(rich) > score_wheel(poor)


def test_wheel_rejects_negative_vrp_and_arb():
    assert score_wheel(base_metrics(vrp=-0.02)) == float("-inf")
    assert score_wheel(base_metrics(arb_flags="butterfly")) == float("-inf")


def test_wheel_handles_nan_iv_rank():
    s = score_wheel(base_metrics(iv_rank=float("nan")))
    assert math.isfinite(s)


def test_leaps_score_prefers_cheap_long_vol():
    cheap = base_metrics(iv1y_rank=5.0, term_slope=-0.08)
    dear = base_metrics(iv1y_rank=95.0, term_slope=-0.08)
    assert score_leaps(cheap) > score_leaps(dear)


def test_suggest_strikes():
    thetas = {T: 0.09 * T for T in (0.1, 0.5, 1.0, 2.0)}
    surf = SSVISurface(rho=-0.6, eta=1.5, gamma=0.45, thetas=thetas, rmse=0.0)
    s = suggest_strikes(surf, spot=100.0)
    assert s["wheel_put_strike"] < 100.0          # OTM put below spot
    assert s["leaps_call_strike"] < 100.0          # 75-delta call is ITM
    assert s["wheel_put_strike"] % 2.5 == pytest.approx(0.0, abs=1e-9)


def test_build_report_sorts_and_drops_rejected():
    rows = [
        {**base_metrics(underlying="A", vrp=0.02), "wheel_put_strike": 95.0,
         "leaps_call_strike": 90.0},
        {**base_metrics(underlying="B", vrp=0.10), "wheel_put_strike": 95.0,
         "leaps_call_strike": 90.0},
        {**base_metrics(underlying="C", vrp=-0.01), "wheel_put_strike": 95.0,
         "leaps_call_strike": 90.0},
    ]
    wheel, leaps = build_report(rows)
    assert list(wheel["underlying"]) == ["B", "A"]  # C rejected (vrp<0)
    assert len(leaps) == 3
    assert "wheel_put_strike" in wheel.columns
    assert "leaps_call_strike" in leaps.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.signals'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/signals.py`:

```python
import math

import numpy as np
import pandas as pd

from ssvi.metrics import delta_k
from ssvi.ssvi import SSVISurface


def _or50(x: float) -> float:
    return 50.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else x


def score_wheel(m: dict) -> float:
    if m["arb_flags"] or m["vrp"] <= 0:
        return float("-inf")
    return (100 * m["vrp"] + 50 * m["rr25_30d"]
            + 0.2 * _or50(m["iv_rank"]))


def score_leaps(m: dict) -> float:
    if m["arb_flags"]:
        return float("-inf")
    return (0.5 * (100 - _or50(m.get("iv1y_rank")))
            + 100 * max(m["term_slope"], 0.0))


def _round_strike(x: float) -> float:
    return round(x / 2.5) * 2.5


def suggest_strikes(surface: SSVISurface, spot: float) -> dict:
    k_put = delta_k(surface, T=30 / 365, target_delta=-0.25)
    k_call = delta_k(surface, T=1.25, target_delta=0.75)
    return {
        "wheel_put_strike": _round_strike(spot * np.exp(k_put)),
        "leaps_call_strike": _round_strike(spot * np.exp(k_call)),
    }


_WHEEL_COLS = ["underlying", "score", "iv30", "iv1y", "vrp", "rr25_30d",
               "iv_rank", "term_slope", "spot", "wheel_put_strike"]
_LEAPS_COLS = ["underlying", "score", "iv30", "iv1y", "vrp", "rr25_30d",
               "iv_rank", "term_slope", "spot", "leaps_call_strike"]


def build_report(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    wheel_rows, leaps_rows = [], []
    for m in rows:
        ws, ls = score_wheel(m), score_leaps(m)
        if ws != float("-inf"):
            wheel_rows.append({**m, "score": ws})
        if ls != float("-inf"):
            leaps_rows.append({**m, "score": ls})
    wheel = (pd.DataFrame(wheel_rows, columns=None)
             .reindex(columns=_WHEEL_COLS)
             .sort_values("score", ascending=False, ignore_index=True)
             if wheel_rows else pd.DataFrame(columns=_WHEEL_COLS))
    leaps = (pd.DataFrame(leaps_rows)
             .reindex(columns=_LEAPS_COLS)
             .sort_values("score", ascending=False, ignore_index=True)
             if leaps_rows else pd.DataFrame(columns=_LEAPS_COLS))
    return wheel, leaps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signals.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/signals.py tests/test_signals.py
git commit -m "feat: wheel and LEAPS signal scoring"
```

---

### Task 12: Plots

**Files:**
- Create: `src/ssvi/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: prepared DataFrame (columns `T`, `k`, `w`, `iv`), `SSVISurface`.
- Produces: `plots.plot_ticker(prepared: pd.DataFrame, surface: SSVISurface, underlying: str, out_dir: Path) -> list[Path]` — writes and returns three PNGs:
  - `<underlying>_smiles.png` — market IV points vs SSVI fitted line, one subplot per expiry (up to 8 nearest expiries).
  - `<underlying>_term.png` — ATM IV vs T (fitted curve + slice knots).
  - `<underlying>_surface.png` — heatmap of `surface.iv` over `k ∈ [-0.5, 0.5]`, `T ∈ [0.05, 2.0]`.
  Uses the `Agg` matplotlib backend (no display needed).

- [ ] **Step 1: Write the failing test**

`tests/test_plots.py`:

```python
import numpy as np
import pandas as pd

from ssvi.plots import plot_ticker
from ssvi.ssvi import SSVISurface, ssvi_total_variance


def test_plot_ticker_writes_three_pngs(tmp_path):
    thetas = {0.25: 0.02, 1.0: 0.07}
    surf = SSVISurface(rho=-0.5, eta=1.2, gamma=0.45, thetas=thetas, rmse=0.0)
    rows = []
    for T, theta in thetas.items():
        k = np.linspace(-0.3, 0.3, 11)
        w = ssvi_total_variance(k, theta, -0.5, 1.2, 0.45)
        for ki, wi in zip(k, w):
            rows.append({"T": T, "k": ki, "w": wi, "iv": np.sqrt(wi / T)})
    prepared = pd.DataFrame(rows)

    paths = plot_ticker(prepared, surf, "TEST", tmp_path)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
    names = {p.name for p in paths}
    assert names == {"TEST_smiles.png", "TEST_term.png", "TEST_surface.png"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plots.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssvi.plots'`.

- [ ] **Step 3: Write minimal implementation**

`src/ssvi/plots.py`:

```python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssvi.ssvi import SSVISurface


def plot_ticker(prepared: pd.DataFrame, surface: SSVISurface,
                underlying: str, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Smiles per expiry
    tenors = sorted(prepared["T"].unique())[:8]
    n = len(tenors)
    fig, axes = plt.subplots(
        (n + 3) // 4, min(n, 4), figsize=(4 * min(n, 4), 3 * ((n + 3) // 4)),
        squeeze=False,
    )
    for ax, T in zip(axes.flat, tenors):
        g = prepared[prepared["T"] == T]
        kk = np.linspace(g["k"].min(), g["k"].max(), 100)
        ax.plot(g["k"], np.sqrt(g["w"] / T), "o", ms=3, label="market")
        ax.plot(kk, surface.iv(kk, T), "-", label="SSVI")
        ax.set_title(f"T={T:.2f}y")
        ax.set_xlabel("log-moneyness k")
    axes.flat[0].legend()
    fig.suptitle(f"{underlying} smiles")
    fig.tight_layout()
    p = out_dir / f"{underlying}_smiles.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 2. ATM term structure
    fig, ax = plt.subplots(figsize=(6, 4))
    ts = np.linspace(0.05, max(2.0, max(surface.thetas)), 100)
    ax.plot(ts, [np.sqrt(surface.theta_at(t) / t) for t in ts], "-",
            label="SSVI ATM IV")
    knots = sorted(surface.thetas)
    ax.plot(knots, [np.sqrt(surface.thetas[t] / t) for t in knots], "o",
            label="slice fits")
    ax.set_xlabel("T (years)")
    ax.set_ylabel("ATM IV")
    ax.set_title(f"{underlying} ATM term structure")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"{underlying}_term.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 3. Surface heatmap
    kk = np.linspace(-0.5, 0.5, 60)
    tt = np.linspace(0.05, 2.0, 60)
    grid = np.array([surface.iv(kk, t) for t in tt])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(kk, tt, grid, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="IV")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("T (years)")
    ax.set_title(f"{underlying} SSVI surface")
    fig.tight_layout()
    p = out_dir / f"{underlying}_surface.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_plots.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssvi/plots.py tests/test_plots.py
git commit -m "feat: smile, term structure, and surface plots"
```

---

### Task 13: CLI orchestration

**Files:**
- Create: `src/ssvi/cli.py`, `src/ssvi/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above. Exact call chain per ticker in `scan`:
  1. `chain.fetch_chain(client, ticker)`
  2. `clean.prepare(chain_df, asof)` — also add `iv` column pass-through for plots (`prepare` output already has `iv`).
  3. `ssvi.fit_surface(prepared)`
  4. `stocks.fetch_daily_closes(client, ticker)` → `stocks.realized_vol(closes)`
  5. `metrics.ticker_metrics(surface, rv20, spot)` where `spot = chain_df["spot"].iloc[0]`
  6. augment with `underlying`, `iv_rank = store.iv_rank(history, ticker, m["iv30"])`, `iv1y_rank` (same call pattern on a history filtered column — implemented as `store.iv_rank` with the history's `iv30` column swapped: add helper `store.metric_rank(history, underlying, column, current, min_obs=20)` and make `iv_rank` delegate to it)
  7. `signals.suggest_strikes(surface, spot)` merged into the row
  8. after all tickers: `store.save_metrics(rows, asof)`, `signals.build_report(rows)`, print tables, optionally `plots.plot_ticker`.
- Produces:
  - `cli.scan(tickers: list[str] | None = None, asof: str | None = None, make_plots: bool = False, client=None) -> tuple[pd.DataFrame, pd.DataFrame]` — programmatic entry; `client=None` builds a real `PolygonClient(cache_date=asof)`. Per-ticker failures are caught, printed as warnings, and skipped.
  - `python -m ssvi scan [--plots] [--tickers NVDA,AAPL]` and `python -m ssvi plot NVDA` (re-plots from today's cache).
  - The printed report ends with the line: `Reminder: check earnings dates before selling short-dated puts.`

**Note:** this task also adds `store.metric_rank` (generalization of `iv_rank`); update `store.iv_rank` to delegate: `iv_rank(...) = metric_rank(history, underlying, "iv30", current, min_obs)`. Add `metric_rank` tests to `tests/test_store.py` as shown below.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`:

```python
def test_metric_rank_generalizes():
    hist = pd.DataFrame({
        "underlying": ["NVDA"] * 25,
        "date": [f"2026-06-{d:02d}" for d in range(1, 26)],
        "iv30": np.linspace(0.30, 0.54, 25),
        "iv1y": np.linspace(0.25, 0.49, 25),
    })
    assert store.metric_rank(hist, "NVDA", "iv1y", 0.49) == pytest.approx(100.0)
    assert store.metric_rank(hist, "NVDA", "iv1y", 0.10) == pytest.approx(0.0)
    assert np.isnan(store.metric_rank(hist, "NVDA", "missing_col", 0.4))
```

`tests/test_cli.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssvi import cli

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "snapshot_page.json").read_text()
)


class FakeClient:
    """Serves a synthetic but realistic chain + stock bars for any ticker."""

    def get_paginated(self, path, params=None):
        if path.startswith("/v2/aggs/"):
            rng = np.random.default_rng(0)
            closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
            return [{"t": 1750000000000 + i * 86400000, "c": float(c)}
                    for i, c in enumerate(closes)]
        # option snapshot: build a dense synthetic chain around spot=100
        results = []
        for expiry, T_days in [("2026-08-21", 47), ("2026-10-16", 103),
                               ("2027-01-15", 194), ("2027-09-17", 439)]:
            for strike in np.arange(70, 131, 5.0):
                for typ in ("call", "put"):
                    iv = 0.30 + 0.002 * (100 - strike) / 5  # mild put skew
                    intrinsic = max(100 - strike, 0) if typ == "put" \
                        else max(strike - 100, 0)
                    mid = intrinsic + 8.0 * np.sqrt(T_days / 365) * iv / 0.3
                    results.append({
                        "details": {
                            "ticker": f"O:X{expiry}{typ[0].upper()}{int(strike)}",
                            "contract_type": typ, "strike_price": float(strike),
                            "expiration_date": expiry,
                        },
                        "last_quote": {"bid": round(mid - 0.1, 2),
                                       "ask": round(mid + 0.1, 2)},
                        "implied_volatility": float(iv),
                        "greeks": {"delta": 0.5 if typ == "call" else -0.5},
                        "open_interest": 500,
                        "day": {"volume": 100},
                        "underlying_asset": {"price": 100.0},
                    })
        return results


def test_scan_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("ssvi.config.HISTORY_DIR", tmp_path / "hist")
    monkeypatch.setattr("ssvi.config.PLOTS_DIR", tmp_path / "plots")
    wheel, leaps = cli.scan(tickers=["FAKE"], asof="2026-07-05",
                            client=FakeClient())
    out = capsys.readouterr().out
    assert "FAKE" in out
    assert "earnings" in out.lower()
    # metrics were persisted
    files = list((tmp_path / "hist" / "metrics").glob("*.parquet"))
    assert len(files) == 1
    saved = pd.read_parquet(files[0])
    assert saved.iloc[0]["underlying"] == "FAKE"
    assert {"iv30", "iv1y", "vrp", "wheel_put_strike"} <= set(saved.columns)


def test_scan_survives_ticker_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("ssvi.config.HISTORY_DIR", tmp_path / "hist")

    class BrokenClient(FakeClient):
        def get_paginated(self, path, params=None):
            if "BROKEN" in path:
                raise RuntimeError("boom")
            return super().get_paginated(path, params)

    wheel, leaps = cli.scan(tickers=["BROKEN", "FAKE"], asof="2026-07-05",
                            client=BrokenClient())
    out = capsys.readouterr().out
    assert "BROKEN" in out and "skip" in out.lower()
    assert (wheel["underlying"] == "FAKE").any() or wheel.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py tests/test_store.py -v`
Expected: `test_cli.py` FAILs with `ModuleNotFoundError`; new store test FAILs with `AttributeError: metric_rank`.

- [ ] **Step 3: Implement `store.metric_rank` and refactor `iv_rank`**

In `src/ssvi/store.py`, replace the `iv_rank` function with:

```python
def metric_rank(history: pd.DataFrame, underlying: str, column: str,
                current: float, min_obs: int = 20) -> float:
    if history.empty or column not in history.columns:
        return float("nan")
    vals = history.loc[history["underlying"] == underlying, column].dropna()
    if len(vals) < min_obs:
        return float("nan")
    return float(100.0 * np.mean(vals <= current))


def iv_rank(history: pd.DataFrame, underlying: str, current_iv30: float,
            min_obs: int = 20) -> float:
    return metric_rank(history, underlying, "iv30", current_iv30, min_obs)
```

- [ ] **Step 4: Write the CLI**

`src/ssvi/cli.py`:

```python
import argparse
from datetime import date

import pandas as pd

from ssvi import chain, clean, config, metrics, plots, signals, ssvi, stocks, store
from ssvi.polygon_client import PolygonClient


def _process_ticker(client, ticker: str, asof_ts: pd.Timestamp,
                    history: pd.DataFrame) -> dict:
    chain_df = chain.fetch_chain(client, ticker)
    if chain_df.empty:
        raise RuntimeError("empty chain")
    prepared = clean.prepare(chain_df, asof=asof_ts)
    surface = ssvi.fit_surface(prepared)
    closes = stocks.fetch_daily_closes(client, ticker)
    rv20 = stocks.realized_vol(closes)
    spot = float(chain_df["spot"].iloc[0])
    m = metrics.ticker_metrics(surface, rv20=rv20, spot=spot)
    m["underlying"] = ticker
    m["iv_rank"] = store.metric_rank(history, ticker, "iv30", m["iv30"])
    m["iv1y_rank"] = store.metric_rank(history, ticker, "iv1y", m["iv1y"])
    m.update(signals.suggest_strikes(surface, spot))
    m["_prepared"] = prepared
    m["_surface"] = surface
    return m


def scan(tickers=None, asof=None, make_plots=False, client=None):
    tickers = tickers or config.UNIVERSE
    asof = asof or str(date.today())
    asof_ts = pd.Timestamp(asof)
    if client is None:
        client = PolygonClient(cache_date=asof)
    history = store.load_history()

    rows = []
    for ticker in tickers:
        try:
            rows.append(_process_ticker(client, ticker, asof_ts, history))
            print(f"ok   {ticker}")
        except Exception as e:  # noqa: BLE001 - one bad ticker must not kill scan
            print(f"skip {ticker}: {e}")

    plain = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in rows]
    if plain:
        store.save_metrics(plain, asof)
    wheel, leaps = signals.build_report(plain)

    pd.set_option("display.width", 200)
    print(f"\n=== Wheel / short-put candidates (<={config.SHORT_DTE_MAX} DTE, "
          f"sell ~25d puts) — {asof} ===")
    print(wheel.round(3).to_string(index=False) if not wheel.empty
          else "(none)")
    print(f"\n=== LEAPS candidates (>= {config.LEAPS_T_MIN}y, "
          f"buy ~75d calls) — {asof} ===")
    print(leaps.round(3).to_string(index=False) if not leaps.empty
          else "(none)")
    print("\nReminder: check earnings dates before selling short-dated puts.")

    if make_plots:
        for r in rows:
            paths = plots.plot_ticker(r["_prepared"], r["_surface"],
                                      r["underlying"], config.PLOTS_DIR)
            print(f"plots: {', '.join(str(p) for p in paths)}")
    return wheel, leaps


def plot_one(ticker: str, asof=None):
    asof = asof or str(date.today())
    client = PolygonClient(cache_date=asof)
    asof_ts = pd.Timestamp(asof)
    chain_df = chain.fetch_chain(client, ticker)
    prepared = clean.prepare(chain_df, asof=asof_ts)
    surface = ssvi.fit_surface(prepared)
    for p in plots.plot_ticker(prepared, surface, ticker, config.PLOTS_DIR):
        print(p)


def main():
    parser = argparse.ArgumentParser(prog="ssvi")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_scan = sub.add_parser("scan", help="daily scan of the universe")
    p_scan.add_argument("--plots", action="store_true")
    p_scan.add_argument("--tickers", type=str, default=None,
                        help="comma-separated override, e.g. NVDA,AAPL")
    p_plot = sub.add_parser("plot", help="plot one ticker from today's cache")
    p_plot.add_argument("ticker")
    args = parser.parse_args()
    if args.cmd == "scan":
        tickers = args.tickers.split(",") if args.tickers else None
        scan(tickers=tickers, make_plots=args.plots)
    elif args.cmd == "plot":
        plot_one(args.ticker)


if __name__ == "__main__":
    main()
```

`src/ssvi/__main__.py`:

```python
from ssvi.cli import main

main()
```

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests pass (config 3, client 4, chain 1, stocks 3, clean 3, svi 4, ssvi 6, metrics 5, store 6, signals 6, plots 1, cli 2).

- [ ] **Step 6: Commit**

```bash
git add src/ssvi/cli.py src/ssvi/__main__.py src/ssvi/store.py tests/test_cli.py tests/test_store.py
git commit -m "feat: CLI scan/plot orchestration"
```

---

### Task 14: First live run and README

**Files:**
- Create: `README.md`
- Modify: nothing (live validation)

**Interfaces:**
- Consumes: the full CLI.
- Produces: verified live behavior + usage docs.

- [ ] **Step 1: Live smoke test on two tickers**

Run: `POLYGON_API_KEY=<real key> python -m ssvi scan --tickers NVDA,AAPL --plots`
Expected: `ok NVDA`, `ok AAPL`, two report tables with plausible numbers (iv30 roughly 0.2–0.6, |ssvi_rho| < 1), PNGs in `plots/`, one parquet in `data/history/metrics/`.
If a ticker is skipped, read the error — most likely causes: entitlement gaps found in Task 3, or filters too strict (tune `MIN_OPEN_INTEREST` / `MAX_STALE_DAYS` in `config.py`).

**Findings from the actual live run (recorded here, already fixed in the code):**
1. The initial run returned zero signals for both tickers because near-0DTE (2-3 day) expiries were included in the chain, adding sparse/noisy slices that distorted the global SSVI fit and triggered butterfly-arbitrage flags almost everywhere. Fixed by adding `config.MIN_DTE = 7` and filtering on it in `clean.prepare`.
2. Even after that fix, real single-stock smiles still flagged butterfly violations at *some* tenor — a single global 3-parameter SSVI power-law can't satisfy strict static-arbitrage bounds everywhere on richly-skewed single-name surfaces (SSVI was designed for smoother index surfaces like SPX). The original design vetoed the whole ticker if `arb_flags` was non-empty anywhere, which emptied out every signal for every real name. Fixed by removing the veto in `signals.score_wheel`/`score_leaps` — `arb_flags` is now a visible diagnostic column in the report instead of an automatic reject. Only `vrp <= 0` still vetoes wheel candidates (no premium, no trade).

- [ ] **Step 2: Full universe run**

Run: `POLYGON_API_KEY=<real key> python -m ssvi scan --plots`
Expected: most of the 20 tickers `ok`; scan completes in minutes (Starter has unlimited calls; pagination is the bulk of the time). Rerun is near-instant thanks to the daily cache.

- [ ] **Step 3: Write README**

`README.md`:

```markdown
# ssvi — IV skew / SSVI scanner for NASDAQ-100 options

Daily scanner that pulls option chains from Polygon.io (massive.com),
fits an SSVI implied-volatility surface per underlying, and ranks:

- **Wheel entries** — short-dated (<45 DTE) cash-secured put sales,
  ranked by variance risk premium (30d ATM IV − 20d realized vol),
  25Δ put-skew richness, and IV rank.
- **LEAPS buys** — long-dated (>1y) call purchases, ranked by how cheap
  long-dated IV is vs. its own history.

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

## Caveats

- Signals are relative-value screens, not trade instructions. Check
  earnings dates before selling short-dated puts.
- No backtest: history accumulates forward from your own scans.
- Realized vol is close-to-close over 20 days; risk-free rate is a
  constant in `config.py`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and daily workflow"
```

---

## Self-Review (completed)

- **Spec coverage:** Polygon API ✓ (Tasks 2–5), IV skew ✓ (Tasks 7–9: SVI slices, 25Δ RR, ATM skew), SSVI global surface ✓ (Task 8), money-making exploration for <45 DTE selling and >1y buying with wheel/LEAPS framing ✓ (Tasks 9–11, 13), top-20 NASDAQ-100 universe ✓ (Task 1), Options Starter constraints (delay, backoff, daily cadence) ✓ (Global Constraints, Tasks 2–3).
- **Placeholder scan:** every code step contains complete code; the only "manual" steps are the two live runs (Tasks 3 and 14), which are inherently manual.
- **Type consistency:** `fetch_chain` column schema (Task 4) matches `prepare` consumption (Task 6); `prepare` output columns (`T,k,w,iv,forward,spot`) match `fit_surface` (Task 8) and `plot_ticker` (Task 12); `SSVISurface` method names (`theta_at`, `w`, `iv`, `check_arbitrage`) are used identically in Tasks 9, 11, 12; `ticker_metrics` keys (Task 9) match `score_wheel`/`score_leaps` consumption (Task 11) with `iv_rank`/`iv1y_rank` added in Task 13 via `store.metric_rank`; `suggest_strikes` keys match `_WHEEL_COLS`/`_LEAPS_COLS` and the CLI test's saved-column assertions.
