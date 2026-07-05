import numpy as np
import pytest

from ssvi.bs import black76_price, implied_vol


def test_call_put_parity_black76():
    F, K, T, r, sigma = 100.0, 95.0, 0.5, 0.04, 0.25
    call = black76_price(F, K, T, r, sigma, "call")
    put = black76_price(F, K, T, r, sigma, "put")
    assert (call - put) == pytest.approx(np.exp(-r * T) * (F - K), rel=1e-8)


def test_atm_call_price_reasonable():
    F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.0, 0.20
    price = black76_price(F, K, T, r, sigma, "call")
    assert 7.0 < price < 9.0  # ATM straddle-ish approx: 0.4*sigma*sqrt(T)*F


def test_price_increases_with_vol():
    F, K, T, r = 100.0, 100.0, 1.0, 0.04
    p_low = black76_price(F, K, T, r, 0.1, "call")
    p_high = black76_price(F, K, T, r, 0.5, "call")
    assert p_high > p_low


def test_implied_vol_recovers_known_sigma():
    F, K, T, r, sigma_true = 100.0, 105.0, 0.75, 0.03, 0.35
    price = black76_price(F, K, T, r, sigma_true, "put")
    iv = implied_vol(price, F, K, T, r, "put")
    assert iv == pytest.approx(sigma_true, rel=1e-4)


def test_implied_vol_recovers_across_moneyness():
    F, T, r, sigma_true = 100.0, 0.3, 0.045, 0.28
    for K, typ in [(80.0, "put"), (100.0, "call"), (120.0, "call")]:
        price = black76_price(F, K, T, r, sigma_true, typ)
        iv = implied_vol(price, F, K, T, r, typ)
        assert iv == pytest.approx(sigma_true, rel=1e-4)


def test_implied_vol_raises_on_arbitrage_violating_price():
    F, K, T, r = 100.0, 100.0, 1.0, 0.0
    with pytest.raises(ValueError):
        implied_vol(price=-5.0, F=F, K=K, T=T, r=r, option_type="call")
