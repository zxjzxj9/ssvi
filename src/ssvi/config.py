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
