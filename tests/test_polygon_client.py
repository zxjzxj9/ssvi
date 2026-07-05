import json

import httpx
import pytest

from ssvi.polygon_client import PolygonClient


def make_client(handler, **kwargs):
    client = PolygonClient(api_key="k", **kwargs)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_get_json_adds_api_key():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"status": "OK", "results": []})

    client = make_client(handler)
    out = client.get_json("/v3/things", {"limit": 5})
    assert out["status"] == "OK"
    assert "apiKey=k" in seen["url"]
    assert "limit=5" in seen["url"]


def test_retries_on_429(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"status": "OK"})

    monkeypatch.setattr("time.sleep", lambda s: None)
    client = make_client(handler)
    assert client.get_json("/v3/things")["status"] == "OK"
    assert calls["n"] == 3


def test_paginated_follows_next_url():
    def handler(request):
        if "cursor" in str(request.url):
            return httpx.Response(200, json={"results": [{"i": 2}]})
        return httpx.Response(
            200,
            json={
                "results": [{"i": 1}],
                "next_url": "https://api.polygon.io/v3/things?cursor=abc",
            },
        )

    client = make_client(handler)
    rows = client.get_paginated("/v3/things")
    assert [r["i"] for r in rows] == [1, 2]


def test_disk_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("ssvi.config.CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"status": "OK", "results": [1]})

    client = make_client(handler, cache_date="2026-07-05")
    first = client.get_json("/v3/things")
    second = client.get_json("/v3/things")  # must come from disk
    assert first == second
    assert calls["n"] == 1
    cached = list((tmp_path / "2026-07-05").glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text())["status"] == "OK"
