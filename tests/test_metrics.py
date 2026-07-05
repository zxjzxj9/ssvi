import numpy as np
import pytest

from ssvi.metrics import atm_iv, atm_skew, delta_k, rr25, ticker_metrics
from ssvi.ssvi import SSVISurface


def flat_surface(vol=0.30):
    # rho=0 and tiny eta -> essentially flat smile at each tenor.
    # Include a knot below 30/365y so ticker_metrics' iv30 interpolates
    # between real knots instead of flat-extrapolating outside the range.
    thetas = {T: vol**2 * T for T in (0.05, 0.1, 0.5, 1.0, 2.0)}
    return SSVISurface(rho=0.0, eta=0.01, gamma=0.5, thetas=thetas, rmse=0.0)


def skewed_surface():
    thetas = {T: 0.09 * T for T in (0.05, 0.1, 0.5, 1.0, 2.0)}
    return SSVISurface(rho=-0.6, eta=1.5, gamma=0.45, thetas=thetas, rmse=0.0)


def test_atm_iv_flat():
    assert atm_iv(flat_surface(0.30), 0.5) == pytest.approx(0.30, rel=1e-6)


def test_delta_k_flat_surface_symmetry():
    surf = flat_surface(0.30)
    k_call = delta_k(surf, T=0.25, target_delta=0.25)
    k_put = delta_k(surf, T=0.25, target_delta=-0.25)
    assert k_call > 0 > k_put
    # flat vol: 25d call and put are symmetric around w/2 shift
    w = 0.30**2 * 0.25
    from scipy.stats import norm
    d1 = -k_call / np.sqrt(w) + np.sqrt(w) / 2
    # eta=0.01 gives a near-flat (not exactly flat) SSVI smile, so this
    # hand-rolled flat-vol d1 check has a small residual curvature error.
    assert norm.cdf(d1) == pytest.approx(0.25, abs=2e-4)


def test_rr25_zero_on_flat_positive_on_skewed():
    assert rr25(flat_surface(), 0.25) == pytest.approx(0.0, abs=1e-3)
    assert rr25(skewed_surface(), 0.25) > 0.01  # rho<0 -> puts rich


def test_atm_skew_sign():
    assert abs(atm_skew(flat_surface(), 0.25)) < 1e-3
    assert atm_skew(skewed_surface(), 0.25) < 0


def test_ticker_metrics_keys_and_vrp():
    m = ticker_metrics(flat_surface(0.30), rv20=0.22, spot=100.0)
    assert m["vrp"] == pytest.approx(0.30 - 0.22, abs=1e-3)
    assert m["iv30"] == pytest.approx(0.30, rel=1e-3)
    assert m["iv1y"] == pytest.approx(0.30, rel=1e-3)
    assert m["term_slope"] == pytest.approx(0.0, abs=1e-3)
    assert m["arb_flags"] == ""
    for key in ("rr25_30d", "rr25_1y", "atm_skew_30d", "ssvi_rho",
                "ssvi_rmse", "rv20", "spot"):
        assert key in m
