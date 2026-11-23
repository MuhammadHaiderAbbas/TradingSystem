"""Event-driven (bar-by-bar) backtest engine.

Per timestamp:
  1. ``SimBroker.on_bar`` for each symbol: orders placed at the previous close fill at
     this bar's OPEN; stops/limits are checked against this bar's range.
  2. Mark to market at the close; the risk manager sees the new equity (daily/weekly
     loss and drawdown halts).
  3. ``Trader.on_bar_close`` decides using signals known at this close; any order
     goes through the risk manager and becomes active on the NEXT bar.

It runs the same Trader / RiskManager / OrderManager / SimBroker classes as paper and
live trading. Signals are precomputed for speed; every strategy is covered by an
automated test proving that its signal at t is identical when computed on data[:t+1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinel.backtest.metrics import buy_and_hold_equity, compute_metrics
from sentinel.config import RiskConfig, Settings
from sentinel.core.trader import Trader
from sentinel.core.types import Bar, MarketRules, Mode
from sentinel.data.timeframes import epoch_seconds, periods_per_year, tf_seconds
from sentinel.execution.order_manager import OrderManager
from sentinel.execution.sim import SimBroker
from sentinel.features import indicators as ind
from sentinel.risk.manager import RiskManager
from sentinel.strategies.combine import SignalPipeline


@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    signals: dict[str, pd.Series]
    benchmark: pd.Series
    metrics: dict[str, Any]
    vetoes: list[dict] = field(default_factory=list)
    halts: list[dict] = field(default_factory=list)
    open_positions: dict[str, float] = field(default_factory=dict)
    source: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        keys = ["total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "trades", "win_rate",
                "profit_factor", "expectancy", "exposure_time", "benchmark_total_return"]
        return {k: self.metrics.get(k) for k in keys}


def vol_zscore(close: pd.Series, n: int = 24, lookback: int = 500) -> pd.Series:
    rv = ind.realized_vol(close, n)
    return (rv - rv.rolling(lookback, min_periods=100).mean()) / rv.rolling(lookback, min_periods=100).std()


def _relaxed(cfg: RiskConfig) -> RiskConfig:
    """Risk config without account-level halts (for parameter sweeps / research only)."""
    return cfg.model_copy(update={"max_daily_loss_pct": 0.5, "max_weekly_loss_pct": 0.5, "max_drawdown_pct": 0.9,
                                  "circuit_breakers": cfg.circuit_breakers.model_copy(
                                      update={"max_consecutive_losses": 10_000})})


def run_backtest(
    data: dict[str, pd.DataFrame] | pd.DataFrame,
    settings: Settings,
    pipeline: SignalPipeline | None = None,
    signals: dict[str, pd.Series] | None = None,
    source: str | None = None,
    initial_cash: float | None = None,
    rules: MarketRules | dict[str, MarketRules] | None = None,
    risk_halts: bool = True,
    params: dict | None = None,
) -> BacktestResult:
    if isinstance(data, pd.DataFrame):
        data = {settings.market.symbols[0]: data}
    tf = settings.market.timeframe
    bar_s = tf_seconds(tf)
    cash = initial_cash if initial_cash is not None else settings.backtest.initial_cash
    if signals is None:
        pipeline = pipeline or SignalPipeline.from_settings(settings)
        results = {s: pipeline.run(df) for s, df in data.items()}
        signals = {s: r.target for s, r in results.items()}
        source = source or next(iter(results.values())).source
    source = source or "custom"
    if rules is None:
        r = settings.rules
        rules = MarketRules(r.min_qty, r.step_size, r.tick_size, r.min_notional)

    broker = SimBroker(cash, settings.costs, rules, bar_seconds=bar_s)
    clock = [0.0]
    risk_cfg = settings.risk if risk_halts else _relaxed(settings.risk)
    risk = RiskManager(risk_cfg, store=None, mode=Mode.BACKTEST, clock=lambda: clock[0], bar_seconds=bar_s)
    om = OrderManager(broker, risk, settings.execution, mode=Mode.BACKTEST, initial_cash=cash)
    trader = Trader(settings, broker, om, risk, Mode.BACKTEST)
    halts: list[dict] = []
    _orig_halt = risk.halt

    def _halt(kind, reason, ts=None):
        _orig_halt(kind, reason, ts)
        halts.append({"kind": kind, "reason": reason, "ts": ts if ts is not None else clock[0]})

    risk.halt = _halt  # type: ignore[method-assign]

    # ---- precompute arrays -----------------------------------------------------------
    symbols = list(data)
    arrs: dict[str, dict[str, np.ndarray]] = {}
    for s in symbols:
        df = data[s]
        sig = signals[s].reindex(df.index).fillna(0.0)
        arrs[s] = {
            "ts": epoch_seconds(df.index), "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
            "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float), "v": df["volume"].to_numpy(float),
            "sig": sig.to_numpy(float),
            "atr": ind.atr(df["high"], df["low"], df["close"], 14).to_numpy(float),
            "vz": vol_zscore(df["close"]).fillna(0.0).to_numpy(float),
            "lr": np.log(df["close"]).diff().fillna(0.0).to_numpy(float),
        }
    union = pd.DatetimeIndex(sorted(set().union(*[set(data[s].index) for s in symbols])))
    union_s = epoch_seconds(union)
    pos_idx = {s: np.searchsorted(arrs[s]["ts"], union_s) for s in symbols}
    lookback = settings.risk.correlation_lookback

    equity = np.empty(len(union))
    exposure = np.empty(len(union))
    for k, t in enumerate(union_s):
        live = []
        for s in symbols:
            i = pos_idx[s][k]
            a = arrs[s]
            if i < len(a["ts"]) and a["ts"][i] == t:
                clock[0] = t
                broker.on_bar(s, Bar(t, a["o"][i], a["h"][i], a["l"][i], a["c"][i], a["v"][i]))
                live.append((s, i))
        close_ts = t + bar_s
        clock[0] = close_ts
        eq = broker.portfolio.equity()
        risk.on_equity(close_ts, eq)
        equity[k] = eq
        exposure[k] = broker.portfolio.exposure_pct()
        i_map = dict(live)
        last_idx = {s2: i_map.get(s2, pos_idx[s2][k] - 1) for s2 in symbols}

        def returns(last_idx=last_idx):
            out = {}
            for s2, j in last_idx.items():
                if j >= 1:
                    out[s2] = arrs[s2]["lr"][max(0, j - lookback + 1): j + 1]
            return out

        for s, i in live:
            a = arrs[s]

            trader.on_bar_close(s, close_ts, a["c"][i], a["sig"][i], a["atr"][i], vol_z=a["vz"][i],
                                returns=returns, source=source)

    eq_s = pd.Series(equity, index=union, name="equity")
    exp_s = pd.Series(exposure, index=union, name="exposure")
    trades = pd.DataFrame(broker.portfolio.closed_trades)
    fills = pd.DataFrame([f.__dict__ | {"side": f.side.value, "purpose": f.purpose.value} for f in broker.fills])
    bench = buy_and_hold_equity(data, cash, settings.costs.taker_fee, union)
    turnover = float((fills["qty"] * fills["price"]).sum()) if len(fills) else 0.0
    ppy = settings.backtest.periods_per_year or periods_per_year(tf)
    metrics = compute_metrics(eq_s, trades, ppy, exp_s, bench, turnover)
    metrics["fees_paid"] = broker.portfolio.fees_paid
    metrics["funding_paid"] = broker.portfolio.funding_paid
    metrics["vetoes"] = len(risk.vetoes)
    metrics["halts"] = len(halts)
    return BacktestResult(eq_s, exp_s, trades, fills, signals, bench, metrics, risk.vetoes, halts,
                          broker.positions(), source, params or {})
