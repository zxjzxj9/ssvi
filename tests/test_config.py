import pytest

from ssvi import config


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-123")
    assert config.get_api_key() == "test-key-123"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="POLYGON_API_KEY"):
        config.get_api_key()


def test_universe_is_reasonable():
    assert 15 <= len(config.UNIVERSE) <= 25
    assert "NVDA" in config.UNIVERSE
    assert len(set(config.UNIVERSE)) == len(config.UNIVERSE)
