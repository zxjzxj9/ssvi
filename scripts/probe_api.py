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
