import math

import pytest

from ssvi.signals import build_report, score_leaps, score_wheel, suggest_strikes
from ssvi.ssvi import SSVISurface


def base_metrics(**kw):
    m = dict(
        underlying="NVDA", iv30=0.45, iv1y=0.40, vrp=0.08, rr25_30d=0.03,
        rr25_1y=0.02, atm_skew_30d=-0.1, term_slope=-0.05, rv20=0.37,
        spot=1000.0, iv_rank=70.0, iv1y_rank=20.0, ssvi_rho=-0.6,
        ssvi_rmse=0.001, arb_flags="",
    )
    m.update(kw)
    return m


def test_wheel_score_ordering():
    rich = base_metrics(vrp=0.10, rr25_30d=0.05, iv_rank=90.0)
    poor = base_metrics(vrp=0.01, rr25_30d=0.00, iv_rank=10.0)
    assert score_wheel(rich) > score_wheel(poor)


def test_wheel_rejects_negative_vrp_and_arb():
    assert score_wheel(base_metrics(vrp=-0.02)) == float("-inf")
    assert score_wheel(base_metrics(arb_flags="butterfly")) == float("-inf")


def test_wheel_handles_nan_iv_rank():
    s = score_wheel(base_metrics(iv_rank=float("nan")))
    assert math.isfinite(s)


def test_leaps_score_prefers_cheap_long_vol():
    cheap = base_metrics(iv1y_rank=5.0, term_slope=-0.08)
    dear = base_metrics(iv1y_rank=95.0, term_slope=-0.08)
    assert score_leaps(cheap) > score_leaps(dear)


def test_suggest_strikes():
    thetas = {T: 0.09 * T for T in (0.1, 0.5, 1.0, 2.0)}
    surf = SSVISurface(rho=-0.6, eta=1.5, gamma=0.45, thetas=thetas, rmse=0.0)
    s = suggest_strikes(surf, spot=100.0)
    assert s["wheel_put_strike"] < 100.0          # OTM put below spot
    assert s["leaps_call_strike"] < 100.0          # 75-delta call is ITM
    assert s["wheel_put_strike"] % 2.5 == pytest.approx(0.0, abs=1e-9)


def test_build_report_sorts_and_drops_rejected():
    rows = [
        {**base_metrics(underlying="A", vrp=0.02), "wheel_put_strike": 95.0,
         "leaps_call_strike": 90.0},
        {**base_metrics(underlying="B", vrp=0.10), "wheel_put_strike": 95.0,
         "leaps_call_strike": 90.0},
        {**base_metrics(underlying="C", vrp=-0.01), "wheel_put_strike": 95.0,
         "leaps_call_strike": 90.0},
    ]
    wheel, leaps = build_report(rows)
    assert list(wheel["underlying"]) == ["B", "A"]  # C rejected (vrp<0)
    assert len(leaps) == 3
    assert "wheel_put_strike" in wheel.columns
    assert "leaps_call_strike" in leaps.columns
