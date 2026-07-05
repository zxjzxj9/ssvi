import numpy as np
import pandas as pd

from ssvi.plots import plot_ticker
from ssvi.ssvi import SSVISurface, ssvi_total_variance


def test_plot_ticker_writes_three_pngs(tmp_path):
    thetas = {0.25: 0.02, 1.0: 0.07}
    surf = SSVISurface(rho=-0.5, eta=1.2, gamma=0.45, thetas=thetas, rmse=0.0)
    rows = []
    for T, theta in thetas.items():
        k = np.linspace(-0.3, 0.3, 11)
        w = ssvi_total_variance(k, theta, -0.5, 1.2, 0.45)
        for ki, wi in zip(k, w):
            rows.append({"T": T, "k": ki, "w": wi, "iv": np.sqrt(wi / T)})
    prepared = pd.DataFrame(rows)

    paths = plot_ticker(prepared, surf, "TEST", tmp_path)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
    names = {p.name for p in paths}
    assert names == {"TEST_smiles.png", "TEST_term.png", "TEST_surface.png"}
