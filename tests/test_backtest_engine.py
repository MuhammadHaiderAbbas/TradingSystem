"""Known synthetic datasets with hand-computed results: fill timing, fees, slippage, stops."""

import numpy as np
import pandas as pd
import pytest

from sentinel.backtest.engine import run_backtest
from sentinel.backtest.metrics import compute_metrics, max_drawdown_duration, trade_stats
from sentinel.config import CostConfig, RiskConfig
from sentinel.data.timeframes import epoch_seconds
from sentinel.features import indicators as ind
from tests.conftest import make_bars, settings_for

FEE = 0.001


def flat_costs(**kw):
    base = dict(maker_fee=FEE, taker_fee=FEE, spread_bps=0.0, slippage_bps=0.0, slippage_vol_k=0.0,
                funding_rate_8h=0.0, max_participation=1.0)
    base.update(kw)
    return CostConfig(**base)


def scenario(opens, entry_bar, exit_bar, **cost_kw):
    df = make_bars(opens)
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[entry_bar:exit_bar] = 1.0
    s = settings_for(["T"], costs=flat_costs(**cost_kw), risk=RiskConfig(stop_atr_mult=2.0))
    return df, sig, run_backtest({"T": df}, s, signals={"T": sig}, source="test")


def base_opens(n=60):
    return [100.0 + 0.2 * np.sin(i) for i in range(n)]


def test_fill_on_next_bar_open_never_signal_bar():
    opens = base_opens()
    opens[31] = 101.0  # entry bar (signal at 30's close)
    opens[41] = 102.5  # exit bar (signal flat at 40's close)
    df, sig, r = scenario(opens, 30, 40)
    t = r.trades.iloc[0]
    assert t["entry_ts"] == epoch_seconds(df.index[[31]])[0]
    assert t["entry_price"] == pytest.approx(101.0)  # bar 31 OPEN, not bar 30 close
    assert t["exit_ts"] == epoch_seconds(df.index[[41]])[0]
    assert t["exit_price"] == pytest.approx(102.5)


def test_position_size_fees_and_pnl_exact():
    opens = base_opens()
    opens[31], opens[41] = 101.0, 102.5
    df, sig, r = scenario(opens, 30, 40)
    atr30 = ind.atr(df["high"], df["low"], df["close"], 14).iloc[30]
    close30 = df["close"].iloc[30]
    stop_dist = 2.0 * atr30
    qty = 10_000 * 0.01 / stop_dist                     # 1% risk between entry ref and stop
    qty = min(qty, 0.5 * 10_000 / close30)              # per-asset exposure cap (50%)
    t = r.trades.iloc[0]
    assert t["qty"] == pytest.approx(qty, rel=1e-9)
    fees = qty * 101.0 * FEE + qty * 102.5 * FEE
    assert t["fees"] == pytest.approx(fees, rel=1e-9)
    assert t["pnl"] == pytest.approx(qty * (102.5 - 101.0) - fees, rel=1e-9)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + t["pnl"], rel=1e-9)
    assert r.metrics["fees_paid"] == pytest.approx(fees, rel=1e-9)


def test_spread_and_slippage_are_adverse():
    opens = base_opens()
    opens[31], opens[41] = 101.0, 102.5
    _, _, r = scenario(opens, 30, 40, spread_bps=10.0, slippage_bps=5.0)
    t = r.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(101.0 * (1 + 0.0005 + 0.0005))
    assert t["exit_price"] == pytest.approx(102.5 * (1 - 0.0005 - 0.0005))


def test_volatility_slippage():
    opens = base_opens()
    opens[31] = 101.0
    df, _, r = scenario(opens, 30, 40, slippage_vol_k=0.1)
    bar = df.iloc[31]
    expected = 101.0 * (1 + 0.1 * (bar["high"] - bar["low"]) / bar["open"])
    assert r.trades.iloc[0]["entry_price"] == pytest.approx(expected)


def test_stop_loss_hit_intrabar_and_gap():
    opens = base_opens()
    opens[31] = 101.0
    df = make_bars(opens)
    df.iloc[35, df.columns.get_loc("low")] = 80.0  # crash through the stop intrabar
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[30:50] = 1.0
    s = settings_for(["T"], costs=flat_costs())
    r = run_backtest({"T": df}, s, signals={"T": sig}, source="t")
    t = r.trades.iloc[0]
    atr30 = ind.atr(df["high"], df["low"], df["close"], 14).iloc[30]
    stop = df["close"].iloc[30] - 2 * atr30
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(stop)
    assert t["exit_ts"] == epoch_seconds(df.index[[35]])[0]
    # no immediate re-entry on the same stale signal after a stop-out
    assert len(r.trades) == 1 and r.open_positions == {}

    df2 = df.copy()
    df2.iloc[35, df2.columns.get_loc("open")] = 85.0  # gap below the stop: fill at the (worse) open
    df2.iloc[35, df2.columns.get_loc("high")] = 86.0
    r2 = run_backtest({"T": df2}, s, signals={"T": sig}, source="t")
    assert r2.trades.iloc[0]["exit_price"] == pytest.approx(85.0)


def test_benchmark_and_metrics_present():
    opens = list(np.linspace(100, 150, 200))
    df = make_bars(opens)
    sig = pd.Series(0.0, index=df.index)
    s = settings_for(["T"], costs=flat_costs())
    r = run_backtest({"T": df}, s, signals={"T": sig}, source="t")
    assert r.metrics["trades"] == 0 and r.equity.iloc[-1] == 10_000
    assert r.metrics["benchmark_total_return"] == pytest.approx(150 / opens[1] * (1 - FEE) - 1, rel=1e-3)


def test_metrics_known_values():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    eq = pd.Series([100, 110, 99, 120, 108], index=idx, dtype=float)
    m = compute_metrics(eq, None, 365)
    assert m["total_return"] == pytest.approx(0.08)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert max_drawdown_duration(eq) == pd.Timedelta(days=2)
    trades = pd.DataFrame({"pnl": [10, -5, 20, -5], "return_pct": [0.1, -0.05, 0.2, -0.05], "entry_ts": [0] * 4,
                           "exit_ts": [3600] * 4, "fees": [1] * 4})
    st = trade_stats(trades)
    assert st["win_rate"] == 0.5 and st["profit_factor"] == pytest.approx(3.0) and st["expectancy"] == 5.0


def test_daily_loss_halt_blocks_entries_in_backtest():
    opens = base_opens(120)
    df = make_bars(opens)
    sig = pd.Series(0.0, index=df.index)
    s = settings_for(["T"], costs=flat_costs(), risk=RiskConfig(max_daily_loss_pct=0.0001, max_weekly_loss_pct=0.06))
    sig.iloc[30:35] = 1.0
    sig.iloc[40:45] = 1.0
    df.iloc[33, df.columns.get_loc("low")] = 50.0
    r = run_backtest({"T": df}, s, signals={"T": sig}, source="t")
    assert any(h["kind"] == "daily_loss" for h in r.halts)
    assert any("halted" in v["reason"] for v in r.vetoes)
