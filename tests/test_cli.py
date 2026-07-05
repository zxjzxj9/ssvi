import numpy as np
import pandas as pd

from ssvi import cli, config
from ssvi.bs import black76_price

ASOF = "2026-07-05"
ASOF_TS = pd.Timestamp(ASOF)
FRESH_NS = int((ASOF_TS - pd.Timedelta(hours=1)).value)

EXPIRIES = [
    ("2026-08-21", 47),
    ("2026-10-16", 103),
    ("2027-01-15", 194),
    ("2027-09-17", 439),
]


class FakeClient:
    """Serves a synthetic but realistic chain + stock bars for any ticker."""

    def get_paginated(self, path, params=None):
        if path.startswith("/v2/aggs/"):
            rng = np.random.default_rng(0)
            closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 60)))
            return [{"t": 1750000000000 + i * 86400000, "c": float(c)}
                    for i, c in enumerate(closes)]
        results = []
        forward = 100.0
        for expiry, days in EXPIRIES:
            T = days / 365.0
            r = config.risk_free_rate(T)
            for strike in np.arange(70.0, 131.0, 5.0):
                sigma = 0.30 + 0.05 * (1 - T) + 0.002 * (forward - strike) / 5.0
                sigma = max(sigma, 0.05)
                for typ in ("call", "put"):
                    price = black76_price(forward, strike, T, r, sigma, typ)
                    results.append({
                        "details": {
                            "ticker": f"O:X{expiry}{typ[0].upper()}{int(strike)}",
                            "contract_type": typ, "strike_price": float(strike),
                            "expiration_date": expiry,
                        },
                        "day": {"close": round(float(price), 4), "volume": 100,
                               "last_updated": FRESH_NS},
                        "greeks": {},
                        "open_interest": 500,
                    })
        return results


def test_scan_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("ssvi.config.HISTORY_DIR", tmp_path / "hist")
    monkeypatch.setattr("ssvi.config.PLOTS_DIR", tmp_path / "plots")
    wheel, leaps = cli.scan(tickers=["FAKE"], asof=ASOF, client=FakeClient())
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
            if path == "/v3/snapshot/options/BROKEN":
                raise RuntimeError("boom")
            return super().get_paginated(path, params)

    wheel, leaps = cli.scan(tickers=["BROKEN", "FAKE"], asof=ASOF,
                            client=BrokenClient())
    out = capsys.readouterr().out
    assert "BROKEN" in out and "skip" in out.lower()
    assert (wheel["underlying"] == "FAKE").any() or wheel.empty
