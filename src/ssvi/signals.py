import math

import numpy as np
import pandas as pd

from ssvi.metrics import delta_k
from ssvi.ssvi import SSVISurface


def _or50(x: float) -> float:
    return 50.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else x


def score_wheel(m: dict) -> float:
    # Note: arb_flags is surfaced as a diagnostic column, not vetoed here.
    # A single global 3-parameter SSVI power-law routinely fails strict
    # butterfly bounds *somewhere* on richly-skewed single-stock smiles
    # (it was designed for smoother index surfaces like SPX) -- vetoing
    # the whole ticker over a flag at some unrelated distant tenor would
    # empty out every real name. Only VRP <= 0 (no premium) blocks a trade.
    if m["vrp"] <= 0:
        return float("-inf")
    return (100 * m["vrp"] + 50 * m["rr25_30d"]
            + 0.2 * _or50(m["iv_rank"]))


def score_leaps(m: dict) -> float:
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
               "iv_rank", "term_slope", "spot", "wheel_put_strike",
               "arb_flags"]
_LEAPS_COLS = ["underlying", "score", "iv30", "iv1y", "vrp", "rr25_30d",
               "iv_rank", "term_slope", "spot", "leaps_call_strike",
               "arb_flags"]


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
