import numpy as np
import pandas as pd
import pytest

from ssvi.ssvi import SSVISurface, fit_surface, ssvi_total_variance

RHO, ETA, GAMMA = -0.55, 1.2, 0.45
THETAS = {0.1: 0.009, 0.25: 0.02, 0.5: 0.038, 1.0: 0.07, 2.0: 0.13}


def synthetic_surface() -> pd.DataFrame:
    rows = []
    for T, theta in THETAS.items():
        k = np.linspace(-0.4, 0.4, 21)
        w = ssvi_total_variance(k, theta, RHO, ETA, GAMMA)
        for ki, wi in zip(k, w):
            rows.append({"T": T, "k": ki, "w": wi})
    return pd.DataFrame(rows)


def test_ssvi_atm_equals_theta():
    w0 = ssvi_total_variance(np.array([0.0]), 0.04, RHO, ETA, GAMMA)
    assert w0[0] == pytest.approx(0.04, rel=1e-12)


def test_fit_recovers_global_params():
    surf = fit_surface(synthetic_surface())
    assert surf.rho == pytest.approx(RHO, abs=0.05)
    assert surf.rmse < 1e-4
    # surface reproduces inputs at an interior tenor
    k = np.array([-0.2, 0.0, 0.2])
    np.testing.assert_allclose(
        surf.w(k, 0.5),
        ssvi_total_variance(k, THETAS[0.5], RHO, ETA, GAMMA),
        atol=5e-4,
    )


def test_theta_interpolation_and_iv():
    surf = fit_surface(synthetic_surface())
    t_mid = surf.theta_at(0.75)
    assert THETAS[0.5] < t_mid < THETAS[1.0]
    iv_atm = float(surf.iv(np.array([0.0]), 1.0)[0])
    assert iv_atm == pytest.approx(np.sqrt(surf.theta_at(1.0) / 1.0), rel=1e-6)


def test_arbitrage_check_flags_calendar_violation():
    surf = SSVISurface(rho=-0.5, eta=1.0, gamma=0.5,
                       thetas={0.5: 0.05, 1.0: 0.03}, rmse=0.0)
    msgs = surf.check_arbitrage()
    assert any("calendar" in m.lower() for m in msgs)


def test_arbitrage_check_clean_surface():
    surf = fit_surface(synthetic_surface())
    assert surf.check_arbitrage() == []


def test_too_few_slices_raises():
    df = synthetic_surface()
    with pytest.raises(ValueError):
        fit_surface(df[df["T"] == 0.5])
