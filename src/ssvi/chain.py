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
