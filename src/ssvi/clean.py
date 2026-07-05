import numpy as np
import pandas as pd

from ssvi import bs, config


def _forward_for_expiry(g: pd.DataFrame, T: float, r: float) -> float | None:
    calls = g[g["type"] == "call"].set_index("strike")["price"]
    puts = g[g["type"] == "put"].set_index("strike")["price"]
    common = calls.index.intersection(puts.index)
    if common.empty:
        return None
    diff = (calls[common] - puts[common]).abs()
    k0 = diff.idxmin()
    return float(k0 + np.exp(r * T) * (calls[k0] - puts[k0]))


def prepare(chain_df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    df = chain_df.copy()
    stale_cutoff = asof - pd.Timedelta(days=config.MAX_STALE_DAYS)
    df = df[
        (df["price"] > 0)
        & (df["open_interest"] >= config.MIN_OPEN_INTEREST)
        & (df["last_updated"] >= stale_cutoff)
    ]
    df["T"] = (df["expiry"] - asof).dt.days / 365.0
    df = df[df["T"] > 1 / 365]

    out = []
    for expiry, g in df.groupby("expiry"):
        T = float(g["T"].iloc[0])
        r = config.risk_free_rate(T)
        fwd = _forward_for_expiry(g, T, r)
        if fwd is None or fwd <= 0:
            continue
        g = g.copy()
        g["forward"] = fwd
        g["k"] = np.log(g["strike"] / fwd)
        otm = ((g["type"] == "put") & (g["strike"] <= fwd)) | (
            (g["type"] == "call") & (g["strike"] >= fwd)
        )
        g = g[otm]

        ivs, keep_idx = [], []
        for idx, row in g.iterrows():
            try:
                iv = bs.implied_vol(row["price"], fwd, row["strike"], T, r,
                                    row["type"])
            except ValueError:
                continue
            ivs.append(iv)
            keep_idx.append(idx)
        g = g.loc[keep_idx].copy()
        g["iv"] = ivs
        g["w"] = g["iv"] ** 2 * g["T"]
        out.append(g)
    if not out:
        return df.iloc[0:0].assign(forward=np.nan, k=np.nan, iv=np.nan, w=np.nan)
    return pd.concat(out, ignore_index=True)
