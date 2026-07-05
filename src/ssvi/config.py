import os
from pathlib import Path

# Top NASDAQ-100 constituents by weight. Edit freely; the pipeline
# treats this as the scan universe.
UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "AMZN", "AVGO", "META", "GOOGL", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "LIN", "CSCO", "TMUS", "INTU",
    "QCOM", "TXN", "AMGN", "ADBE",
]

# Approximate USD risk-free (SOFR/UST) term structure, tenor in years -> rate.
# Polygon's plan tiers don't include a rates feed, so this is a hand-maintained
# curve used to discount parity forwards and Black-76 prices. Update
# periodically from UST par yields; a flat 4-4.5% error here has negligible
# effect on IV inversion relative to staleness in the underlying trade prints.
RATE_CURVE = {
    0.083: 0.052,   # 1mo
    0.25: 0.051,    # 3mo
    0.5: 0.048,     # 6mo
    1.0: 0.044,     # 1y
    2.0: 0.042,     # 2y
    5.0: 0.043,     # 5y
}

BASE_URL = "https://api.polygon.io"

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "cache"
HISTORY_DIR = _ROOT / "data" / "history"
PLOTS_DIR = _ROOT / "plots"

# Liquidity filters applied to raw chains
MIN_OPEN_INTEREST = 10
MAX_STALE_DAYS = 5          # drop contracts whose last trade is older than this
SHORT_DTE_MAX = 45          # short-vol strategy horizon
LEAPS_T_MIN = 1.0           # years; long-vol strategy horizon


def risk_free_rate(T: float) -> float:
    """Interpolate RATE_CURVE at tenor T (years); flat extrapolation at ends."""
    tenors = sorted(RATE_CURVE)
    rates = [RATE_CURVE[t] for t in tenors]
    if T <= tenors[0]:
        return rates[0]
    if T >= tenors[-1]:
        return rates[-1]
    for i in range(len(tenors) - 1):
        lo, hi = tenors[i], tenors[i + 1]
        if lo <= T <= hi:
            frac = (T - lo) / (hi - lo)
            return rates[i] + frac * (rates[i + 1] - rates[i])
    return rates[-1]


def get_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError(
            "POLYGON_API_KEY environment variable is not set. "
            "Export your massive.com/polygon.io key first."
        )
    return key
