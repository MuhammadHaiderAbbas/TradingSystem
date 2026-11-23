import pytest
import yaml

from sentinel.config import Settings, apply_env_overrides, dump_default_config, load_config
from sentinel.core.errors import ConfigError
from sentinel.core.types import Mode


def write(tmp_path, data) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


def test_defaults_valid_and_paper():
    s = Settings()
    assert s.app.mode is Mode.PAPER
    assert s.risk.max_leverage == 1.0
    assert s.privacy.telemetry is False


def test_example_config_is_valid():
    s = load_config("config/sentinel.example.yaml", environ={})
    assert s.market.symbols == ["BTC/USDT", "ETH/USDT"]


def test_default_dump_roundtrip(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(dump_default_config())
    assert load_config(p, environ={}).risk.risk_per_trade_pct == 0.01


@pytest.mark.parametrize("bad", [
    {"risk": {"risk_per_trade_pct": 0.5}},                      # above 5% cap
    {"risk": {"max_leverage": 10}},                              # leverage cap
    {"risk": {"max_daily_loss_pct": 0.1, "max_weekly_loss_pct": 0.05}},  # daily > weekly
    {"market": {"timeframe": "7m"}},
    {"market": {"symbols": []}},
    {"market": {"symbols": ["BTC/USDT", "BTC/USDT"]}},
    {"unknown_section": {"x": 1}},                               # unknown keys rejected
    {"risk": {"not_a_field": 1}},
    {"costs": {"taker_fee": 0, "slippage_bps": 0, "spread_bps": 0}},  # zero-cost fantasy
    {"privacy": {"telemetry": True}},
    {"strategies": [{"name": "ema_adx", "enabled": False}]},     # nothing enabled
    {"strategies": [{"name": "ema_adx"}, {"name": "rsi_bb"}]},   # several but no combiner
    {"sentiment": {"endpoint": "https://api.cloud-llm.example.com"}},  # cloud LLM refused
])
def test_invalid_config_refused(tmp_path, bad):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), environ={})


def test_secret_in_config_refused(tmp_path):
    with pytest.raises(ConfigError, match="vault"):
        load_config(write(tmp_path, {"market": {"api_key": "abcd1234"}}), environ={})


def test_missing_explicit_file_refused(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml", environ={})


def test_env_overrides(tmp_path):
    env = {"SENTINEL__RISK__MAX_DAILY_LOSS_PCT": "0.02", "SENTINEL__MARKET__SYMBOLS": "[BTC/USDT]",
           "SENTINEL__LOGGING__LEVEL": "DEBUG", "SENTINEL_HOME": str(tmp_path / "h")}
    s = load_config(write(tmp_path, {}), environ=env)
    assert s.risk.max_daily_loss_pct == 0.02
    assert s.market.symbols == ["BTC/USDT"]
    assert s.logging.level == "DEBUG"
    assert s.app.home == str(tmp_path / "h")


def test_env_override_invalid_refused(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, {}), environ={"SENTINEL__RISK__MAX_LEVERAGE": "50"})


def test_apply_env_nested_creates_sections():
    d = apply_env_overrides({}, {"SENTINEL__RISK__CIRCUIT_BREAKERS__MAX_API_ERRORS": "3"})
    assert d == {"risk": {"circuit_breakers": {"max_api_errors": 3}}}
