"""Configuration: YAML file + environment overrides, validated by pydantic.

Sentinel refuses to start on invalid config: unknown keys, out-of-range limits or
inconsistent settings all raise :class:`ConfigError`.

Env overrides use ``SENTINEL__<SECTION>__<KEY>=value`` (double underscores separate
nesting levels; values are parsed as YAML scalars, so ``0.02``, ``true`` and
``[a, b]`` work).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from sentinel.core.errors import ConfigError
from sentinel.core.types import Mode

ENV_PREFIX = "SENTINEL__"
TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AppConfig(_Model):
    name: str = "sentinel"
    home: str = "var"
    mode: Mode = Mode.PAPER


class LoggingConfig(_Model):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: str = "var/logs"
    max_bytes: int = Field(10_000_000, gt=10_000)
    backup_count: int = Field(10, ge=1, le=1000)
    console: bool = True


class MarketConfig(_Model):
    venue: str = "binance"  # ccxt exchange id, "alpaca" or "sim"
    testnet: bool = True
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "1h"
    quote_currency: str = "USDT"

    @field_validator("timeframe")
    @classmethod
    def _tf(cls, v: str) -> str:
        if v not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
        return v

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one symbol is required")
        if len(set(v)) != len(v):
            raise ValueError("duplicate symbols")
        return v


class DataConfig(_Model):
    db_path: str = "var/market.db"
    parquet_dir: str = "var/parquet"
    history_start: str = "2021-01-01"
    outlier_zscore: float = Field(8.0, gt=2)
    stale_feed_seconds: int = Field(180, gt=0)


class CostConfig(_Model):
    maker_fee: float = Field(0.001, ge=0, le=0.01)
    taker_fee: float = Field(0.001, ge=0, le=0.01)
    spread_bps: float = Field(2.0, ge=0, le=500)
    slippage_bps: float = Field(2.0, ge=0, le=500)
    slippage_vol_k: float = Field(0.02, ge=0, le=1.0)  # fraction of bar range
    funding_rate_8h: float = Field(0.0001, ge=-0.01, le=0.01)  # applied to shorts / leverage
    max_participation: float = Field(0.1, gt=0, le=1.0)  # max share of bar volume per fill

    @model_validator(mode="after")
    def _costs_nonzero(self) -> CostConfig:
        if self.taker_fee == 0 and self.slippage_bps == 0 and self.spread_bps == 0:
            raise ValueError("refusing zero-cost assumptions: set fees, spread or slippage > 0")
        return self


class RulesConfig(_Model):
    min_qty: float = Field(0.0, ge=0)
    step_size: float = Field(0.0, ge=0)
    tick_size: float = Field(0.0, ge=0)
    min_notional: float = Field(10.0, ge=0)


class BacktestConfig(_Model):
    initial_cash: float = Field(10_000.0, gt=0)
    allow_short: bool = False
    holdout_fraction: float = Field(0.2, ge=0.05, le=0.5)
    periods_per_year: float | None = None  # inferred from timeframe if None


class StrategyEntry(_Model):
    name: str
    enabled: bool = True
    weight: float = Field(1.0, ge=0)
    params: dict[str, Any] = Field(default_factory=dict)
    regimes: list[str] | None = None  # None = all regimes


class CombinerConfig(_Model):
    method: Literal["none", "vote", "weighted"] = "none"
    threshold: float = Field(0.5, gt=0, le=1)


class CircuitBreakerConfig(_Model):
    max_consecutive_losses: int = Field(5, ge=1)
    extreme_vol_zscore: float = Field(4.0, gt=1)
    stale_data_seconds: int = Field(180, gt=0)
    max_api_errors: int = Field(5, ge=1)


class RiskConfig(_Model):
    sizing: Literal["fixed_fractional", "atr", "kelly"] = "fixed_fractional"
    risk_per_trade_pct: float = Field(0.01, gt=0, le=0.05)
    kelly_fraction: float = Field(0.25, gt=0, le=1)
    kelly_cap_pct: float = Field(0.02, gt=0, le=0.05)
    atr_target_vol_pct: float = Field(0.01, gt=0, le=0.1)
    max_open_positions: int = Field(3, ge=1, le=100)
    max_exposure_per_asset_pct: float = Field(0.5, gt=0, le=1.0)
    max_total_exposure_pct: float = Field(1.0, gt=0, le=3.0)
    max_leverage: float = Field(1.0, ge=1.0, le=3.0)
    max_daily_loss_pct: float = Field(0.03, gt=0, le=0.5)
    max_weekly_loss_pct: float = Field(0.06, gt=0, le=0.5)
    max_drawdown_pct: float = Field(0.15, gt=0, le=0.9)
    max_correlation: float = Field(0.8, gt=0, le=1.0)
    correlation_lookback: int = Field(100, ge=20)
    stop_atr_mult: float = Field(2.0, gt=0)
    take_profit_atr_mult: float | None = Field(None, gt=0)
    trailing_stop_pct: float | None = Field(None, gt=0, lt=0.5)
    cooldown_minutes: int = Field(60, ge=0)
    circuit_breakers: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    @model_validator(mode="after")
    def _consistent(self) -> RiskConfig:
        if self.max_daily_loss_pct > self.max_weekly_loss_pct:
            raise ValueError("max_daily_loss_pct cannot exceed max_weekly_loss_pct")
        if self.max_exposure_per_asset_pct > self.max_total_exposure_pct:
            raise ValueError("max_exposure_per_asset_pct cannot exceed max_total_exposure_pct")
        if self.max_total_exposure_pct > self.max_leverage:
            raise ValueError("max_total_exposure_pct cannot exceed max_leverage")
        return self


class ExecutionConfig(_Model):
    order_retry_max: int = Field(5, ge=0, le=20)
    backoff_base_s: float = Field(0.5, ge=0)
    limit_first: bool = True
    limit_offset_bps: float = Field(1.0, ge=0)
    limit_timeout_s: int = Field(30, ge=1)
    twap_threshold_notional: float = Field(5_000.0, gt=0)
    twap_slices: int = Field(5, ge=2, le=100)
    twap_interval_s: int = Field(60, ge=1)
    min_bar_volume_quote: float = Field(0.0, ge=0)  # skip trading if liquidity below this
    reconcile_interval_min: int = Field(5, ge=1)
    reconcile_tolerance: float = Field(1e-6, ge=0)


class MLConfig(_Model):
    enabled: bool = False
    model: Literal["lightgbm", "random_forest", "logistic"] = "lightgbm"
    threshold: float = Field(0.55, ge=0.5, lt=1)
    tp_atr: float = Field(2.0, gt=0)
    sl_atr: float = Field(2.0, gt=0)
    max_holding_bars: int = Field(48, ge=1)
    n_splits: int = Field(5, ge=2)
    embargo_pct: float = Field(0.01, ge=0, le=0.2)
    registry_dir: str = "var/models"
    retrain_cron: str = "0 3 * * 0"  # Sundays 03:00 UTC
    drift_psi_threshold: float = Field(0.25, gt=0)


class SentimentConfig(_Model):
    enabled: bool = False
    endpoint: str = "http://localhost:11434"
    model: str = "llama3.1"

    @field_validator("endpoint")
    @classmethod
    def _local_only(cls, v: str) -> str:
        if not any(h in v for h in ("localhost", "127.0.0.1", "[::1]")):
            raise ValueError("sentiment endpoint must be local unless privacy.allow_cloud is set")
        return v


class DashboardConfig(_Model):
    host: str = "127.0.0.1"
    port: int = Field(8765, ge=1, le=65535)
    username: str = "admin"


class TelegramConfig(_Model):
    enabled: bool = False
    chat_id: str = ""


class EmailConfig(_Model):
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    sender: str = ""
    recipient: str = ""


class AlertsConfig(_Model):
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    daily_summary_cron: str = "5 0 * * *"


class GateConfig(_Model):
    min_paper_weeks: float = Field(8, ge=0)
    min_paper_trades: int = Field(50, ge=0)
    max_heartbeat_gap_hours: float = Field(6, gt=0)
    backtest_tolerance: float = Field(0.5, gt=0)  # relative tolerance on per-trade expectancy
    max_paper_drawdown_pct: float = Field(0.15, gt=0, le=1)
    live_capital_cap: float = Field(250.0, gt=0, le=1_000_000)
    cap_step_max_multiplier: float = Field(2.0, gt=1, le=10)
    min_weeks_between_cap_raises: float = Field(4, ge=0)


class PrivacyConfig(_Model):
    telemetry: Literal[False] = False
    allow_cloud: bool = False


class Settings(_Model):
    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategies: list[StrategyEntry] = Field(
        default_factory=lambda: [StrategyEntry(name="ema_adx", regimes=["trend"])]
    )
    combiner: CombinerConfig = Field(default_factory=CombinerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    @model_validator(mode="after")
    def _cross(self) -> Settings:
        if self.risk.max_drawdown_pct < self.risk.max_weekly_loss_pct:
            raise ValueError("risk.max_drawdown_pct must be >= risk.max_weekly_loss_pct")
        enabled = [s for s in self.strategies if s.enabled]
        if not enabled:
            raise ValueError("at least one strategy must be enabled")
        if self.combiner.method == "none" and len(enabled) > 1:
            raise ValueError("several strategies enabled: set combiner.method to 'vote' or 'weighted'")
        return self

    @property
    def home(self) -> Path:
        return Path(self.app.home)


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_env_overrides(data: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not path:
            continue
        node = data
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[path[-1]] = _parse_scalar(raw)
    return data


def load_config(path: str | Path | None = None, environ: dict[str, str] | None = None) -> Settings:
    """Load and validate config. Raises ConfigError on any problem."""
    env = os.environ if environ is None else environ
    cfg_path = Path(path or env.get("SENTINEL_CONFIG", "config/sentinel.yaml"))
    data: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            loaded = yaml.safe_load(cfg_path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"cannot parse {cfg_path}: {e}") from e
        if not isinstance(loaded, dict):
            raise ConfigError(f"{cfg_path} must contain a mapping at the top level")
        data = loaded
    elif path is not None:
        raise ConfigError(f"config file not found: {cfg_path}")
    data = apply_env_overrides(data, env)
    if "SENTINEL_HOME" in env:
        data.setdefault("app", {})["home"] = env["SENTINEL_HOME"]
    _reject_secrets(data)
    try:
        settings = Settings.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"invalid configuration:\n{e}") from e
    if settings.app.mode is Mode.LIVE:
        # Config alone can never turn on LIVE: the CLI must also pass the readiness gate.
        pass
    return settings


_SECRET_HINTS = ("api_key", "apikey", "secret", "password", "passphrase", "token", "private_key")


def _reject_secrets(data: Any, path: str = "") -> None:
    """Config files must never contain secrets; they belong in the vault."""
    if isinstance(data, dict):
        for k, v in data.items():
            kp = f"{path}.{k}" if path else str(k)
            if any(h in str(k).lower() for h in _SECRET_HINTS) and v not in (None, "", False):
                raise ConfigError(f"config key '{kp}' looks like a secret; store it in the vault instead")
            _reject_secrets(v, kp)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            _reject_secrets(v, f"{path}[{i}]")


def dump_default_config() -> str:
    return yaml.safe_dump(Settings().model_dump(mode="json"), sort_keys=False)
