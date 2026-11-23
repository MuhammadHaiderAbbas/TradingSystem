"""LIVE readiness gate and capital cap."""

import pytest

from sentinel.core.errors import KillSwitchEngaged, ModeLockedError, TradingHalted
from sentinel.core.modes import HALT_KEY, KILL_KEY, LIVE_CONFIRMATION, ModeManager
from sentinel.core.types import Mode
from sentinel.gate.readiness import WEEK, current_cap, evaluate_gate, raise_cap, record_gate

NOW = 1_800_000_000.0


def seed_good_paper(store, weeks=9, trades=60, exp=0.004):
    start = NOW - weeks * WEEK
    t = start
    while t <= NOW:
        store.heartbeat("runner", mode="PAPER", ts=t)
        t += 3600
    eq = 10_000.0
    for i in range(trades):
        pnl = 40.0 if i % 2 == 0 else -20.0
        ts = start + (i + 1) * (weeks * WEEK / (trades + 1))
        store.record_trade("PAPER", {"strategy": "ema_adx", "symbol": "BTC/USDT", "side": "long", "qty": 0.1,
                                     "entry_ts": ts - 3600, "exit_ts": ts, "entry_price": 50_000, "exit_price": 50_100,
                                     "pnl": pnl, "fees": 2, "return_pct": pnl / 5000})
        eq += pnl
        store.record_equity("PAPER", ts, eq, eq, 0.3)
    store.set("backtest.expectations", {"expectancy_pct": exp, "oos_expectancy_pct": 0.003})
    store.set("kill.tested", {"ts": NOW - 86400})
    store.set("keycheck.result", {"can_trade": True, "can_withdraw": False, "ip_restricted": True, "source": "test"})


def test_gate_fails_on_fresh_install(settings, store):
    rep = evaluate_gate(settings, store, now=NOW, run_test_suite=False)
    assert not rep.passed
    assert all(not c.passed for c in rep.checks if c.name != "reconciliation_clean")


def test_gate_passes_when_everything_checks_out(settings, store):
    seed_good_paper(store)
    rep = evaluate_gate(settings, store, now=NOW, test_runner=lambda: (True, "all passed"))
    assert rep.passed, [c for c in rep.checks if not c.passed]
    mm = ModeManager(settings, store)
    with pytest.raises(ModeLockedError):
        mm.require(Mode.LIVE, LIVE_CONFIRMATION)  # not recorded yet
    record_gate(settings, store, rep)
    with pytest.raises(ModeLockedError):
        mm.require(Mode.LIVE, "yes")  # wrong confirmation phrase
    assert mm.require(Mode.LIVE, LIVE_CONFIRMATION) is Mode.LIVE
    assert current_cap(store) == settings.gate.live_capital_cap


@pytest.mark.parametrize("breaker", ["short", "few_trades", "drift", "tests", "kill", "key", "recon", "gap"])
def test_each_check_can_block(settings, store, breaker):
    seed_good_paper(store, weeks=5 if breaker == "short" else 9, trades=20 if breaker == "few_trades" else 60)
    runner = (lambda: (False, "1 failed")) if breaker == "tests" else (lambda: (True, "ok"))
    if breaker == "drift":
        store.set("backtest.expectations", {"expectancy_pct": 0.05, "oos_expectancy_pct": 0.003})
    if breaker == "kill":
        store.delete("kill.tested")
    if breaker == "key":
        store.set("keycheck.result", {"can_trade": True, "can_withdraw": True, "ip_restricted": True})
    if breaker == "recon":
        store.add_recon_error("PAPER", {"x": 1})
    now = NOW + (8 * 3600 if breaker == "gap" else 0)  # paper stopped 8h ago (> 6h allowed gap)
    assert not evaluate_gate(settings, store, now=now, test_runner=runner).passed


def test_config_change_invalidates_gate(settings, store):
    seed_good_paper(store)
    record_gate(settings, store, evaluate_gate(settings, store, now=NOW, test_runner=lambda: (True, "ok")))
    changed = settings.model_copy(update={"risk": settings.risk.model_copy(update={"risk_per_trade_pct": 0.02})})
    ok, why = ModeManager(changed, store).live_unlocked()
    assert not ok and "config changed" in why


def test_kill_and_halt_block_modes(settings, store):
    mm = ModeManager(settings, store)
    store.set(KILL_KEY, {"ts": 1})
    with pytest.raises(KillSwitchEngaged):
        mm.require(Mode.PAPER)
    store.delete(KILL_KEY)
    store.set(HALT_KEY, {"kind": "daily_loss", "reason": "x", "ts": 1})
    with pytest.raises(TradingHalted):
        mm.require(Mode.PAPER)
    assert mm.require(Mode.BACKTEST) is Mode.BACKTEST


def test_capital_cap_rules(settings, store):
    seed_good_paper(store)
    record_gate(settings, store, evaluate_gate(settings, store, now=NOW, test_runner=lambda: (True, "ok")))
    cap0 = current_cap(store)
    c = store.get("live.capital_cap")
    ok, msg = raise_cap(settings, store, cap0 * 3, now=c["ts"] + 10 * WEEK)
    assert not ok and "step too large" in msg
    ok, msg = raise_cap(settings, store, cap0 * 2, now=c["ts"] + 3600)
    assert not ok and "too soon" in msg
    ok, msg = raise_cap(settings, store, cap0 * 2, now=c["ts"] + 10 * WEEK)
    assert not ok and "live trades" in msg
    for i in range(12):
        store.record_trade("LIVE", {"strategy": "s", "symbol": "BTC/USDT", "side": "long", "qty": 0.01,
                                    "entry_ts": c["ts"] + i * 100, "exit_ts": c["ts"] + i * 100 + 50,
                                    "entry_price": 50_000, "exit_price": 50_100, "pnl": 1.0, "fees": 0.1,
                                    "return_pct": 0.0022})
    ok, msg = raise_cap(settings, store, cap0 * 2, now=c["ts"] + 10 * WEEK)
    assert ok and current_cap(store) == cap0 * 2
    ok, _ = raise_cap(settings, store, 100, now=NOW)
    assert ok and current_cap(store) == 100  # lowering is always allowed
