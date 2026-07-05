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


def test_risk_free_rate_at_knots():
    for T, r in config.RATE_CURVE.items():
        assert config.risk_free_rate(T) == pytest.approx(r)


def test_risk_free_rate_interpolates_between_knots():
    r = config.risk_free_rate(0.75)  # between 0.5y (0.048) and 1y (0.044)
    assert 0.044 < r < 0.048


def test_risk_free_rate_flat_extrapolation():
    assert config.risk_free_rate(0.01) == pytest.approx(config.RATE_CURVE[0.083])
    assert config.risk_free_rate(30.0) == pytest.approx(config.RATE_CURVE[5.0])
