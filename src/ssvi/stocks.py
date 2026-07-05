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
