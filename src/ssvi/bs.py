import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76_price(F: float, K: float, T: float, r: float, sigma: float,
                  option_type: str) -> float:
    """Black-76 price of a European option on forward F."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option_type == "call":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price: float, F: float, K: float, T: float, r: float,
                option_type: str) -> float:
    """Invert black76_price for sigma via Brent's method."""
    disc = np.exp(-r * T)
    intrinsic = disc * max(F - K, 0.0) if option_type == "call" else \
        disc * max(K - F, 0.0)
    if price < intrinsic - 1e-8:
        raise ValueError(
            f"price {price} below intrinsic value {intrinsic}: "
            "violates no-arbitrage bound"
        )

    def f(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - price

    lo, hi = 1e-4, 5.0
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-8))
