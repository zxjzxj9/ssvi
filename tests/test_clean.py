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
