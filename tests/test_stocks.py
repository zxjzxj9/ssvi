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
