import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ssvi.ssvi import SSVISurface


def atm_iv(surface: SSVISurface, T: float) -> float:
    return float(np.sqrt(surface.theta_at(T) / T))


def _call_delta(surface: SSVISurface, k: float, T: float) -> float:
    w = float(surface.w(np.array([k]), T)[0])
    d1 = -k / np.sqrt(w) + np.sqrt(w) / 2
    return float(norm.cdf(d1))


def delta_k(surface: SSVISurface, T: float, target_delta: float) -> float:
    if target_delta > 0:  # call
        f = lambda k: _call_delta(surface, k, T) - target_delta
    else:  # put: delta = N(d1) - 1
        f = lambda k: (_call_delta(surface, k, T) - 1) - target_delta
    return float(brentq(f, -2.0, 2.0, xtol=1e-12, rtol=1e-12))


def rr25(surface: SSVISurface, T: float) -> float:
    k_put = delta_k(surface, T, -0.25)
    k_call = delta_k(surface, T, 0.25)
    iv_put = float(surface.iv(np.array([k_put]), T)[0])
    iv_call = float(surface.iv(np.array([k_call]), T)[0])
    return iv_put - iv_call


def atm_skew(surface: SSVISurface, T: float, h: float = 0.01) -> float:
    iv = surface.iv(np.array([-h, h]), T)
    return float((iv[1] - iv[0]) / (2 * h))


def ticker_metrics(surface: SSVISurface, rv20: float, spot: float) -> dict:
    T30 = 30 / 365
    iv30 = atm_iv(surface, T30)
    iv1y = atm_iv(surface, 1.0)
    return {
        "iv30": iv30,
        "iv1y": iv1y,
        "rr25_30d": rr25(surface, T30),
        "rr25_1y": rr25(surface, 1.0),
        "atm_skew_30d": atm_skew(surface, T30),
        "term_slope": iv1y - iv30,
        "vrp": iv30 - rv20,
        "rv20": rv20,
        "spot": spot,
        "ssvi_rho": surface.rho,
        "ssvi_rmse": surface.rmse,
        "arb_flags": ";".join(surface.check_arbitrage()),
    }
