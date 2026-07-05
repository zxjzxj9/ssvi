from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssvi.ssvi import SSVISurface


def plot_ticker(prepared: pd.DataFrame, surface: SSVISurface,
                underlying: str, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Smiles per expiry
    tenors = sorted(prepared["T"].unique())[:8]
    n = len(tenors)
    fig, axes = plt.subplots(
        (n + 3) // 4, min(n, 4), figsize=(4 * min(n, 4), 3 * ((n + 3) // 4)),
        squeeze=False,
    )
    for ax, T in zip(axes.flat, tenors):
        g = prepared[prepared["T"] == T]
        kk = np.linspace(g["k"].min(), g["k"].max(), 100)
        ax.plot(g["k"], np.sqrt(g["w"] / T), "o", ms=3, label="market")
        ax.plot(kk, surface.iv(kk, T), "-", label="SSVI")
        ax.set_title(f"T={T:.2f}y")
        ax.set_xlabel("log-moneyness k")
    axes.flat[0].legend()
    fig.suptitle(f"{underlying} smiles")
    fig.tight_layout()
    p = out_dir / f"{underlying}_smiles.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 2. ATM term structure
    fig, ax = plt.subplots(figsize=(6, 4))
    ts = np.linspace(0.05, max(2.0, max(surface.thetas)), 100)
    ax.plot(ts, [np.sqrt(surface.theta_at(t) / t) for t in ts], "-",
            label="SSVI ATM IV")
    knots = sorted(surface.thetas)
    ax.plot(knots, [np.sqrt(surface.thetas[t] / t) for t in knots], "o",
            label="slice fits")
    ax.set_xlabel("T (years)")
    ax.set_ylabel("ATM IV")
    ax.set_title(f"{underlying} ATM term structure")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"{underlying}_term.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 3. Surface heatmap
    kk = np.linspace(-0.5, 0.5, 60)
    tt = np.linspace(0.05, 2.0, 60)
    grid = np.array([surface.iv(kk, t) for t in tt])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(kk, tt, grid, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="IV")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("T (years)")
    ax.set_title(f"{underlying} SSVI surface")
    fig.tight_layout()
    p = out_dir / f"{underlying}_surface.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    return paths
