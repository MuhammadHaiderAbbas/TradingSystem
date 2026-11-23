"""Every limit breach -> veto or halt; approvals cannot be forged or reused."""

import dataclasses

import numpy as np
import pytest

from sentinel.config import CircuitBreakerConfig, RiskConfig
from sentinel.core.modes import HALT_KEY, KILL_KEY
from sentinel.core.types import MarketRules, Mode, Order, OrderPurpose, Side
from sentinel.risk import sizing
from sentinel.risk.manager import OrderIntent, RiskContext, RiskManager

T0 = 1_704_067_200.0  # 2024-01-01 00:00 UTC (Monday)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def rm(store=None, mode=Mode.PAPER, **cfg):
    clock = Clock()
    return RiskManager(RiskConfig(**cfg), store=store, mode=mode, clock=clock), clock


def ctx(**kw):
    base = dict(ts=T0, equity=10_000.0, cash=10_000.0, positions={}, prices={"BTC": 100.0, "ETH": 50.0})
    base.update(kw)
    return RiskContext(**base)


def entry(symbol="BTC", price=100.0, stop=98.0, side=Side.BUY, **kw):
    return OrderIntent(symbol=symbol, side=side, purpose=OrderPurpose.ENTRY, price=price, stop_price=stop, **kw)


# ------------------------------------------------------------------ sizing
def test_fixed_fractional_sizing_and_caps():
    r, _ = rm()
    d = r.evaluate(entry(), ctx())
    assert d.approved and d.qty == pytest.approx(10_000 * 0.01 / 2.0 if 50 * 100 <= 5000 else 50)
    # 1% risk over a $2 stop = 50 units = $5,000 = exactly the 50% per-asset cap
    assert d.qty == pytest.approx(50)
    d2 = r.evaluate(entry(stop=99.5), ctx())  # tighter stop -> 200 units, capped at 50% exposure (50 units)
    assert d2.qty == pytest.approx(50)
    d3 = r.evaluate(entry(strength=0.5), ctx())
    assert d3.qty == pytest.approx(25)


def test_cash_and_total_exposure_caps():
    r, _ = rm(max_exposure_per_asset_pct=0.5, max_total_exposure_pct=0.6)
    c = ctx(positions={"ETH": 100.0}, cash=5_000.0)  # $5,000 already in ETH
    d = r.evaluate(entry(), c)
    assert d.approved and d.qty * 100 <= 0.6 * 10_000 - 5_000 + 1e-6


def test_kelly_and_atr_sizing():
    assert sizing.kelly_fraction(0.5, 2.0) == pytest.approx(0.25)
    assert sizing.kelly_size(10_000, 100, 98, 0.4, 1.0) == 0.0  # negative edge -> no trade
    assert sizing.kelly_size(10_000, 100, 98, 0.6, 2.0, 0.25, 0.02) == pytest.approx(10_000 * 0.02 / 2)
    assert sizing.atr_target(10_000, 0.01, 2.0) == pytest.approx(50)
    r, _ = rm(sizing="kelly")
    d = r.evaluate(entry(), ctx())  # < 30 trades of history: half fixed-fractional
    assert d.qty == pytest.approx(25)


# ------------------------------------------------------------------ per-trade vetoes
@pytest.mark.parametrize("intent,reason", [
    (entry(stop=None), "stop-loss"),
    (entry(stop=101), "stop-loss"),                      # long stop above price
    (entry(side=Side.SELL, stop=99), "stop-loss"),       # short stop below price
])
def test_mandatory_stop(intent, reason):
    r, _ = rm()
    d = r.evaluate(intent, ctx())
    assert not d.approved and reason in d.reason


def test_max_open_positions():
    r, _ = rm(max_open_positions=2)
    d = r.evaluate(entry("SOL", 10, 9.5), ctx(positions={"BTC": 1, "ETH": 1}, prices={"BTC": 100, "ETH": 50, "SOL": 10}))
    assert not d.approved and "max open positions" in d.reason


def test_no_pyramiding():
    r, _ = rm()
    assert not r.evaluate(entry(), ctx(positions={"BTC": 1})).approved


def test_correlation_veto():
    r, _ = rm(max_correlation=0.8)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 200)
    rets = {"BTC": base + rng.normal(0, 0.001, 200), "ETH": base}
    d = r.evaluate(entry(), ctx(positions={"ETH": 10}, returns=rets))
    assert not d.approved and "correlated" in d.reason
    rets2 = {"BTC": rng.normal(0, 0.01, 200), "ETH": base}
    assert r.evaluate(entry(), ctx(positions={"ETH": 10}, returns=rets2)).approved


def test_exchange_minimum_veto():
    r, _ = rm()
    d = r.evaluate(entry(), ctx(equity=10, cash=10, rules=MarketRules(min_notional=50)))
    assert not d.approved and "minimum" in d.reason


def test_circuit_breakers_stale_and_extreme_vol():
    r, _ = rm(circuit_breakers=CircuitBreakerConfig(stale_data_seconds=60, extreme_vol_zscore=3))
    assert "stale" in r.evaluate(entry(), ctx(data_age_s=120)).reason
    assert "volatility" in r.evaluate(entry(), ctx(vol_z=5)).reason
    assert r.evaluate(entry(), ctx(vol_z=1)).approved


# ------------------------------------------------------------------ account halts
def test_daily_loss_halt_and_manual_reenable_with_cooldown(store):
    r, clock = rm(store, max_daily_loss_pct=0.03, cooldown_minutes=60)
    r.on_equity(T0, 10_000)
    r.on_equity(T0 + 3600, 9_650)
    assert store.get(HALT_KEY)["kind"] == "daily_loss"
    assert not r.evaluate(entry(), ctx()).approved
    # still halted the next day: halts in PAPER/LIVE need a manual re-enable
    clock.t = T0 + 86400 * 2
    r.on_equity(clock.t, 9_700)
    assert r.evaluate(entry(), ctx(ts=clock.t)).reason.startswith("halted")
    clock.t = T0 + 1800
    with pytest.raises(RuntimeError, match="cooldown"):
        r.reenable()
    clock.t = T0 + 86400 * 3
    r.reenable("checked")
    assert store.get(HALT_KEY) is None
    assert any(e["kind"] == "veto" for e in store.events())


def test_weekly_and_drawdown_halts(store):
    r, _ = rm(store, max_daily_loss_pct=0.05, max_weekly_loss_pct=0.06, max_drawdown_pct=0.10)
    r.on_equity(T0, 10_000)
    for d, eq in enumerate([9_800, 9_600, 9_390]):  # ~2% per day, never 5% in one day
        r.on_equity(T0 + 86400 * (d + 1), eq)
    assert store.get(HALT_KEY)["kind"] == "weekly_loss"
    r2, _ = rm(None, Mode.BACKTEST, max_drawdown_pct=0.10, max_daily_loss_pct=0.05, max_weekly_loss_pct=0.06)
    r2.on_equity(T0, 10_000)
    r2.on_equity(T0 + 86400 * 10, 12_000)
    r2.on_equity(T0 + 86400 * 20, 10_700)  # 10.8% below the peak
    assert r2.halt_state["kind"] == "drawdown"
    r2.on_equity(T0 + 86400 * 40, 10_800)  # backtest: drawdown halt does NOT auto-clear
    assert r2.halt_state["kind"] == "drawdown"


def test_consecutive_losses_halt(store):
    r, _ = rm(store, circuit_breakers=CircuitBreakerConfig(max_consecutive_losses=3))
    for _ in range(2):
        r.on_trade_closed({"pnl": -10, "exit_ts": T0})
    assert not store.get(HALT_KEY)
    r.on_trade_closed({"pnl": -10, "exit_ts": T0})
    assert store.get(HALT_KEY)["kind"] == "consecutive_losses"


def test_api_error_burst_halt(store):
    r, _ = rm(store, circuit_breakers=CircuitBreakerConfig(max_api_errors=3))
    for _ in range(3):
        r.on_api_error("503")
    assert store.get(HALT_KEY)["kind"] == "api_errors"


def test_backtest_daily_halt_auto_clears_next_day():
    r, clock = rm(None, Mode.BACKTEST, max_daily_loss_pct=0.03)
    r.on_equity(T0, 10_000)
    r.on_equity(T0 + 3600, 9_600)
    assert r.halt_state
    clock.t = T0 + 86400 + 3600
    assert r.evaluate(entry(), ctx(ts=clock.t)).approved


def test_exits_allowed_while_halted_and_killed(store):
    r, _ = rm(store)
    r.halt("drawdown", "test")
    store.set(KILL_KEY, {"ts": T0})
    ex = OrderIntent("BTC", Side.SELL, OrderPurpose.EXIT, 100.0, qty=5)
    d = r.evaluate(ex, ctx(positions={"BTC": 5}))
    assert d.approved and d.qty == 5
    assert "kill" in r.evaluate(entry(), ctx()).reason


def test_risk_state_persists_across_restart(store):
    r, _ = rm(store)
    r.on_equity(T0, 10_000)
    r.on_equity(T0 + 60, 11_000)
    r2, _ = rm(store)
    assert r2.peak_equity == 11_000 and r2.day_start_equity == 10_000


# ------------------------------------------------------------------ approvals
def _order(ap, **kw):
    base = dict(symbol=ap.symbol, side=Side(ap.side), qty=ap.max_qty)
    base.update(kw)
    return Order(**base)


def test_approval_single_use_and_budgeted():
    r, _ = rm()
    ap = r.evaluate(entry(), ctx()).approval
    assert r.verify_and_consume(ap, _order(ap, qty=ap.max_qty / 2))
    assert r.verify_and_consume(ap, _order(ap, qty=ap.max_qty / 2))
    assert not r.verify_and_consume(ap, _order(ap, qty=1e-6))  # budget exhausted


def test_approval_cannot_be_forged_or_altered():
    r, _ = rm()
    ap = r.evaluate(entry(), ctx()).approval
    assert not r.verify_and_consume(dataclasses.replace(ap, max_qty=ap.max_qty * 10), _order(ap, qty=ap.max_qty * 10))
    assert not r.verify_and_consume(dataclasses.replace(ap, sig="0" * 64), _order(ap))
    assert not r.verify_and_consume(ap, _order(ap, symbol="ETH"))
    assert not r.verify_and_consume(ap, _order(ap, side=Side.SELL))
    assert not r.verify_and_consume(ap, _order(ap, qty=ap.max_qty * 1.01))
    other, _ = rm()  # a different risk manager (different secret key)
    assert not other.verify_and_consume(ap, _order(ap))


def test_approval_expires():
    r, clock = rm()
    ap = r.evaluate(entry(), ctx()).approval
    clock.t += 301
    assert not r.verify_and_consume(ap, _order(ap))


def test_exit_approval_only_for_reduce_only():
    r, _ = rm()
    ap = r.approve_exit("BTC", Side.SELL, 5)
    assert not r.verify_and_consume(ap, Order("BTC", Side.SELL, 5, purpose=OrderPurpose.EXIT, reduce_only=False))
    assert r.verify_and_consume(ap, Order("BTC", Side.SELL, 5, purpose=OrderPurpose.EXIT, reduce_only=True))
    with pytest.raises(ValueError):
        r.approve_exit("BTC", Side.BUY, 1, OrderPurpose.ENTRY)
