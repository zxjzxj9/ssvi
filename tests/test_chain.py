import json
from pathlib import Path

import pandas as pd

from ssvi.chain import fetch_chain

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "snapshot_page.json").read_text()
)


class FakeClient:
    def get_paginated(self, path, params=None):
        assert path == "/v3/snapshot/options/AAPL"
        assert params["limit"] == 250
        return FIXTURE["results"]


def test_fetch_chain_parses_and_drops_incomplete():
    df = fetch_chain(FakeClient(), "AAPL")
    # third row has no day.close -> dropped
    assert len(df) == 2
    row = df[df["type"] == "call"].iloc[0]
    assert row["strike"] == 200.0
    assert row["price"] == 12.8
    assert row["expiry"] == pd.Timestamp("2026-06-18")
    assert row["underlying"] == "AAPL"
    assert row["contract"] == "O:AAPL260618C00200000"
    assert row["open_interest"] == 1500
    assert row["last_updated"] == pd.Timestamp(1751500000000000000, unit="ns")
