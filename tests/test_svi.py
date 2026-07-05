import numpy as np
import pytest

from ssvi.svi import atm_total_variance, fit_slice, svi_total_variance

TRUE = dict(a=0.02, b=0.4, rho=-0.6, m=0.05, sigma=0.2)


def test_svi_function_shape():
    w = svi_total_variance(np.array([0.0]), **TRUE)
    expected = 0.02 + 0.4 * (-0.6 * -0.05 + np.sqrt(0.05**2 + 0.2**2))
    assert w[0] == pytest.approx(expected, rel=1e-12)


def test_fit_recovers_known_params():
    k = np.linspace(-0.5, 0.5, 25)
    w = svi_total_variance(k, **TRUE)
    fit = fit_slice(k, w)
    refit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"],
                               fit["m"], fit["sigma"])
    assert fit["rmse"] < 1e-6
    np.testing.assert_allclose(refit, w, atol=1e-5)
    assert atm_total_variance(fit) == pytest.approx(
        svi_total_variance(np.array([0.0]), **TRUE)[0], rel=1e-4
    )


def test_fit_noisy_data_reasonable():
    rng = np.random.default_rng(42)
    k = np.linspace(-0.4, 0.4, 30)
    w = svi_total_variance(k, **TRUE) + rng.normal(0, 0.0005, 30)
    fit = fit_slice(k, w)
    assert fit["rmse"] < 0.002
    assert -1 < fit["rho"] < 1


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_slice(np.array([0.0, 0.1]), np.array([0.04, 0.05]))
