import numpy as np
import pandas as pd

from ssvi import config


def _metrics_dir():
    d = config.HISTORY_DIR / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_metrics(rows: list[dict], asof: str) -> None:
    df = pd.DataFrame(rows)
    df["date"] = asof
    df.to_parquet(_metrics_dir() / f"{asof}.parquet", index=False)


def load_history() -> pd.DataFrame:
    files = sorted(_metrics_dir().glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def iv_rank(history: pd.DataFrame, underlying: str, current_iv30: float,
            min_obs: int = 20) -> float:
    if history.empty or "iv30" not in history.columns:
        return float("nan")
    vals = history.loc[history["underlying"] == underlying, "iv30"].dropna()
    if len(vals) < min_obs:
        return float("nan")
    return float(100.0 * np.mean(vals <= current_iv30))
