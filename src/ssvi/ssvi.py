from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ssvi import svi


def _phi(theta, eta, gamma):
    return eta / (theta**gamma * (1 + theta) ** (1 - gamma))


def ssvi_total_variance(k, theta, rho, eta, gamma):
    k = np.asarray(k, dtype=float)
    p = _phi(theta, eta, gamma)
    return 0.5 * theta * (
        1 + rho * p * k + np.sqrt((p * k + rho) ** 2 + 1 - rho**2)
    )


@dataclass
class SSVISurface:
    rho: float
    eta: float
    gamma: float
    thetas: dict[float, float]   # T -> theta (raw slice ATM total variance)
    rmse: float
    _ts: np.ndarray = field(init=False, repr=False)
    _th: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        ts = np.array(sorted(self.thetas))
        th = np.array([self.thetas[t] for t in ts])
        self._ts = ts
        self._th = np.maximum.accumulate(th)  # enforce calendar monotonicity

    def theta_at(self, T: float) -> float:
        ts, th = self._ts, self._th
        if T <= ts[-1]:
            return float(np.interp(T, ts, th))
        if len(ts) >= 2:  # linear extrapolation, floored at last knot
            slope = (th[-1] - th[-2]) / (ts[-1] - ts[-2])
            return float(max(th[-1], th[-1] + slope * (T - ts[-1])))
        return float(th[-1])

    def w(self, k, T: float) -> np.ndarray:
        return ssvi_total_variance(k, self.theta_at(T), self.rho,
                                   self.eta, self.gamma)

    def iv(self, k, T: float) -> np.ndarray:
        return np.sqrt(self.w(k, T) / T)

    def check_arbitrage(self) -> list[str]:
        msgs = []
        raw = np.array([self.thetas[t] for t in sorted(self.thetas)])
        if np.any(np.diff(raw) < -1e-8):
            msgs.append("calendar arbitrage: slice ATM total variance "
                        "decreases with maturity")
        cap = 4.0 * (1 + 1e-9)
        for t in sorted(self.thetas):
            theta = self.thetas[t]
            p = _phi(theta, self.eta, self.gamma)
            if theta * p * (1 + abs(self.rho)) > cap:
                msgs.append(f"butterfly condition 1 violated at T={t:.3f}")
            if theta * p**2 * (1 + abs(self.rho)) > cap:
                msgs.append(f"butterfly condition 2 violated at T={t:.3f}")
        return msgs


def fit_surface(prepared: pd.DataFrame) -> SSVISurface:
    thetas: dict[float, float] = {}
    for T, g in prepared.groupby("T"):
        if len(g) < 5:
            continue
        params = svi.fit_slice(g["k"].values, g["w"].values)
        thetas[float(T)] = svi.atm_total_variance(params)
    if len(thetas) < 2:
        raise ValueError(f"need >=2 usable expiries, got {len(thetas)}")

    pts = prepared[prepared["T"].isin(thetas)]
    k = pts["k"].values
    w = pts["w"].values
    theta_row = pts["T"].map(thetas).values

    def residuals(p):
        rho, eta, gamma = p
        return ssvi_total_variance(k, theta_row, rho, eta, gamma) - w

    res = least_squares(
        residuals, x0=[-0.5, 1.0, 0.5],
        bounds=([-0.999, 0.01, 0.01], [0.999, 10.0, 0.99]),
    )
    rho, eta, gamma = res.x
    rmse = float(np.sqrt(np.mean(res.fun**2)))
    return SSVISurface(rho=float(rho), eta=float(eta), gamma=float(gamma),
                       thetas=thetas, rmse=rmse)
