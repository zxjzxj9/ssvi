import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def fit_slice(k: np.ndarray, w: np.ndarray) -> dict:
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(k) < 5:
        raise ValueError(f"need >=5 points to fit SVI slice, got {len(k)}")

    w_max = float(np.max(w))

    def residuals(p):
        a, b, rho, m, sigma = p
        return svi_total_variance(k, a, b, rho, m, sigma) - w

    lb = [-w_max, 0.0, -0.999, -1.0, 1e-4]
    ub = [w_max * 2, 5.0, 0.999, 1.0, 2.0]
    starts = [
        [np.min(w) * 0.5, 0.1, -0.5, 0.0, 0.1],
        [np.min(w) * 0.9, 0.5, 0.0, 0.1, 0.3],
        [0.0, 0.3, -0.8, -0.1, 0.2],
    ]
    best = None
    for x0 in starts:
        try:
            res = least_squares(residuals, x0, bounds=(lb, ub))
        except ValueError:
            continue
        rmse = float(np.sqrt(np.mean(res.fun**2)))
        if best is None or rmse < best[1]:
            best = (res.x, rmse)
    a, b, rho, m, sigma = best[0]
    return {"a": float(a), "b": float(b), "rho": float(rho),
            "m": float(m), "sigma": float(sigma), "rmse": best[1]}


def atm_total_variance(params: dict) -> float:
    return float(svi_total_variance(
        np.array([0.0]), params["a"], params["b"], params["rho"],
        params["m"], params["sigma"],
    )[0])
