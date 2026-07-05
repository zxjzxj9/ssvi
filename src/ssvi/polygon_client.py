import hashlib
import json
import time

import httpx

from ssvi import config


class PolygonClient:
    def __init__(self, api_key: str | None = None, cache_date: str | None = None):
        self.api_key = api_key or config.get_api_key()
        self.cache_date = cache_date
        self._http = httpx.Client(timeout=30.0)

    def _cache_path(self, url: str, params: dict):
        if self.cache_date is None:
            return None
        key = hashlib.sha1(
            (url + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        d = config.CACHE_DIR / self.cache_date
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _get(self, url: str, params: dict) -> dict:
        cache = self._cache_path(url, params)
        if cache is not None and cache.exists():
            return json.loads(cache.read_text())
        delay = 1.0
        merged_url = httpx.URL(url).copy_merge_params(
            {**params, "apiKey": self.api_key}
        )
        for attempt in range(5):
            resp = self._http.get(merged_url)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            if cache is not None:
                cache.write_text(json.dumps(data))
            return data
        resp.raise_for_status()
        raise RuntimeError(f"gave up after retries: {url}")

    def get_json(self, path: str, params: dict | None = None) -> dict:
        url = path if path.startswith("http") else config.BASE_URL + path
        return self._get(url, dict(params or {}))

    def get_paginated(self, path: str, params: dict | None = None) -> list[dict]:
        out: list[dict] = []
        data = self.get_json(path, params)
        out.extend(data.get("results", []))
        while data.get("next_url"):
            data = self.get_json(data["next_url"])
            out.extend(data.get("results", []))
        return out
