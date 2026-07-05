import numpy as np
import pandas as pd
import pytest

from ssvi import store


@pytest.fixture(autouse=True)
def tmp_history(tmp_path, monkeypatch):
    monkeypatch.setattr("ssvi.config.HISTORY_DIR", tmp_path)
    return tmp_path


def test_save_and_load_roundtrip():
    rows = [{"underlying": "NVDA", "iv30": 0.45, "vrp": 0.08}]
    store.save_metrics(rows, asof="2026-07-05")
    hist = store.load_history()
    assert len(hist) == 1
    assert hist.iloc[0]["underlying"] == "NVDA"
    assert str(hist.iloc[0]["date"]) == "2026-07-05"


def test_same_day_rerun_overwrites():
    store.save_metrics([{"underlying": "NVDA", "iv30": 0.45}], "2026-07-05")
    store.save_metrics([{"underlying": "NVDA", "iv30": 0.50}], "2026-07-05")
    hist = store.load_history()
    assert len(hist) == 1
    assert hist.iloc[0]["iv30"] == 0.50


def test_load_history_empty():
    assert store.load_history().empty


def test_iv_rank():
    dates = [f"2026-06-{d:02d}" for d in range(1, 26)]
    hist = pd.DataFrame({
        "underlying": ["NVDA"] * 25,
        "date": dates,
        "iv30": np.linspace(0.30, 0.54, 25),  # 25 obs
    })
    rank = store.iv_rank(hist, "NVDA", current_iv30=0.54)
    assert rank == pytest.approx(100.0)
    assert store.iv_rank(hist, "NVDA", 0.29) == pytest.approx(0.0)
    mid = store.iv_rank(hist, "NVDA", 0.42)
    assert 40 <= mid <= 60


def test_iv_rank_insufficient_history():
    hist = pd.DataFrame({"underlying": ["NVDA"] * 5,
                         "date": ["2026-07-01"] * 5,
                         "iv30": [0.4] * 5})
    assert np.isnan(store.iv_rank(hist, "NVDA", 0.45, min_obs=20))


def test_metric_rank_generalizes():
    hist = pd.DataFrame({
        "underlying": ["NVDA"] * 25,
        "date": [f"2026-06-{d:02d}" for d in range(1, 26)],
        "iv30": np.linspace(0.30, 0.54, 25),
        "iv1y": np.linspace(0.25, 0.49, 25),
    })
    assert store.metric_rank(hist, "NVDA", "iv1y", 0.49) == pytest.approx(100.0)
    assert store.metric_rank(hist, "NVDA", "iv1y", 0.10) == pytest.approx(0.0)
    assert np.isnan(store.metric_rank(hist, "NVDA", "missing_col", 0.4))
