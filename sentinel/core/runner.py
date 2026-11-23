"""Paper / live runner: feed -> signals -> trader -> risk -> order manager -> broker.

The runner owns the event loop for PAPER and LIVE modes:

* on every closed bar: update the bar history, let the broker process resting orders,
  mark to market, update the risk manager's equity guards, compute the strategy signal
  on the trailing window (the same code the backtester uses), and hand it to the Trader;
* every ``reconcile_interval_min``: reconcile the internal ledger with the venue;
* every bar: write heartbeat, equity snapshot, broker state (crash recovery);
* honours the kill switch / halts written by the CLI or dashboard (shared state store).
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import pandas as pd

from sentinel.backtest.engine import vol_zscore
from sentinel.config import Settings
from sentinel.core.modes import KILL_KEY
from sentinel.core.state import StateStore
from sentinel.core.trader import Trader
from sentinel.core.types import Bar, Mode
from sentinel.data.timeframes import tf_offset, tf_seconds
from sentinel.execution.broker import Broker
from sentinel.execution.order_manager import OrderManager
from sentinel.execution.reconcile import reconcile
from sentinel.execution.sim import SimBroker
from sentinel.features import indicators as ind
from sentinel.monitoring.alerts import Alerter
from sentinel.monitoring.logging import get_logger
from sentinel.risk.manager import RiskManager
from sentinel.strategies.combine import SignalPipeline

log = get_logger("runner")

STATE_KEY = "paper.broker_state"


class Runner:
    def __init__(self, settings: Settings, broker: Broker, store: StateStore, mode: Mode,
                 history: dict[str, pd.DataFrame] | None = None, alerter: Alerter | None = None,
                 clock: Callable[[], float] = time.time, window: int = 1500, ml_filter=None) -> None:
        if mode is Mode.BACKTEST:
            raise ValueError("use run_backtest for BACKTEST mode")
        self.s = settings
        self.mode = mode
        self.store = store
        self.broker = broker
        self.clock = clock
        self.bar_s = tf_seconds(settings.market.timeframe)
        self.window = window
        self.alerter = alerter
        self.ml_filter = ml_filter
        self.risk = RiskManager(settings.risk, store, mode, clock=clock, bar_seconds=self.bar_s)
        self.om = OrderManager(broker, self.risk, settings.execution, store, mode,
                               initial_cash=broker.balance()["cash"], native_brackets=isinstance(broker, SimBroker))
        self.trader = Trader(settings, broker, self.om, self.risk, mode)
        self.pipeline = SignalPipeline.from_settings(settings)
        self.hist: dict[str, deque] = {s: deque(maxlen=window) for s in settings.market.symbols}
        for s, df in (history or {}).items():
            for ts, row in df.tail(window).iterrows():
                self.hist.setdefault(s, deque(maxlen=window)).append(
                    (ts, row["open"], row["high"], row["low"], row["close"], row["volume"]))
        self.last_bar_time: dict[str, float] = {}
        self.last_reconcile = 0.0
        self.bars_processed = 0
        self.om.trade_listeners.append(self._on_trade)
        self._recover()
        from sentinel.funds import Funds

        self.funds = Funds(store, mode.value, settings.funds)
        self.funds.ensure_opening(self.broker.balance()["equity"], ts=self.clock())
        self.trader.reserved_provider = self.funds.reserved
        self.last_funds_sync = 0.0

    # ---- recovery -------------------------------------------------------------------
    def _recover(self) -> None:
        if isinstance(self.broker, SimBroker):
            st = self.store.get(f"{STATE_KEY}.{self.mode.value}")
            if st:
                self.broker.restore_state(st)
                # rebuild the internal ledger from the venue snapshot (same process boundary)
                self.om.ledger.cash = self.broker.portfolio.cash
                for sym, pos in self.broker.portfolio.positions.items():
                    import copy

                    self.om.ledger.positions[sym] = copy.deepcopy(pos)
                log.info("recovered paper broker state", extra={"positions": self.broker.positions()})
        n = self.om.recover()
        self.store.log_event("startup", f"{self.mode.value} runner started; {n} open orders refreshed",
                             mode=self.mode.value)

    def _persist(self) -> None:
        if isinstance(self.broker, SimBroker):
            self.store.set(f"{STATE_KEY}.{self.mode.value}", self.broker.to_state())

    # ---- events ---------------------------------------------------------------------
    def _on_trade(self, trade: dict) -> None:
        if self.alerter:
            self.alerter.send("trade_filled", f"{trade['symbol']} {trade['side']} closed pnl={trade['pnl']:.2f} "
                                              f"({trade['exit_reason']})")

    def frame(self, symbol: str) -> pd.DataFrame:
        rows = list(self.hist[symbol])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
        df.index = pd.DatetimeIndex(df.index)
        return df

    def returns_window(self) -> dict[str, np.ndarray]:
        out = {}
        for s in self.hist:
            if len(self.hist[s]) > 2:
                c = np.array([r[4] for r in self.hist[s]])
                out[s] = np.diff(np.log(c))[-self.s.risk.correlation_lookback:]
        return out

    def on_bar(self, symbol: str, bar: Bar) -> None:
        """Handle one CLOSED bar (bar.ts = open time)."""
        ts = pd.Timestamp(int(bar.ts * 1000), unit="ms", tz="UTC").floor(tf_offset(self.s.market.timeframe))
        if self.hist[symbol] and ts <= self.hist[symbol][-1][0]:
            return  # duplicate / out of order
        self.hist[symbol].append((ts, bar.open, bar.high, bar.low, bar.close, bar.volume))
        self.last_bar_time[symbol] = bar.ts
        if isinstance(self.broker, SimBroker):
            self.broker.on_bar(symbol, bar)
        close_ts = bar.ts + self.bar_s
        self.process_funds(force_sync=bool(self.funds.flows(("requested", "ready"), "withdrawal")))
        bal = self.broker.balance()
        self.risk.on_equity(close_ts, bal["equity"])
        self.store.record_equity(self.mode.value, close_ts, bal["equity"], bal["cash"],
                                 getattr(getattr(self.broker, "portfolio", None), "exposure_pct", lambda: 0.0)())
        self.store.heartbeat("runner", {"mode": self.mode.value, "symbol": symbol, "bar": str(ts),
                                        "equity": bal["equity"], "pid": os.getpid()}, mode=self.mode.value)
        if self.check_kill():
            return
        df = self.frame(symbol)
        if len(df) < self.pipeline.warmup():
            return
        res = self.pipeline.run(df)
        sig = res.target
        disabled = set(self.store.get("strategies.disabled", []) or [])
        if disabled & set(res.per_strategy):
            active = {k: v for k, v in res.per_strategy.items() if k not in disabled}
            if not active:
                sig = sig * 0.0
            elif len(active) == 1:
                sig = next(iter(active.values()))
            else:
                from sentinel.strategies.combine import combine_signals

                w = {e.name: e.weight for e in self.s.strategies}
                method = self.s.combiner.method if self.s.combiner.method != "none" else "vote"
                sig = combine_signals(active, w, method, self.s.combiner.threshold)
        if self.ml_filter is not None:
            sig = self.ml_filter(sig, df)
        atr = ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1]
        vz = vol_zscore(df["close"]).iloc[-1]
        data_age = max(0.0, self.clock() - (bar.ts + self.bar_s)) if self.mode is Mode.LIVE else 0.0
        feats = {"close": bar.close, "atr": float(atr), "vol_z": float(vz) if np.isfinite(vz) else 0.0,
                 **{k: float(v.iloc[-1]) for k, v in res.per_strategy.items()}}
        self.store.heartbeat(f"signal.{symbol}", {"target": float(sig.iloc[-1]), "source": res.source,
                                                  "per_strategy": {k: float(v.iloc[-1]) for k, v in
                                                                   res.per_strategy.items()}})
        self.trader.on_bar_close(symbol, close_ts, bar.close, float(sig.iloc[-1]), float(atr),
                                 vol_z=float(vz) if np.isfinite(vz) else 0.0, data_age_s=data_age,
                                 returns=self.returns_window, features=feats, source=res.source)
        self.bars_processed += 1
        self._persist()
        self.maybe_reconcile()

    def process_funds(self, force_sync: bool = False) -> None:
        """Read the venue's funding history (read-only), apply deposits/withdrawals to the accounting,
        and mark reserved withdrawals ready once their cash is free."""
        ex = getattr(self.broker, "ex", None)
        now = self.clock()
        if ex is not None and (force_sync or now - self.last_funds_sync >= self.s.funds.sync_interval_min * 60):
            self.last_funds_sync = now
            try:
                self.funds.sync_from_exchange(ex)
            except Exception as e:  # never let bookkeeping break trading
                log.warning("funds sync failed", extra={"err": str(e)[:200]})
        applied = self.funds.apply_pending(self.broker, self.risk, self.om.ledger, ts=now)
        if applied:
            self._persist()
            if self.alerter:
                for f in applied:
                    self.alerter.send("funds", f"{f['kind']} of {f['amount']:,.2f} {f['currency']} recorded")
        self.funds.refresh_ready(self.broker.balance()["cash"])

    def tick(self) -> None:
        """Periodic housekeeping between bars (live loop: every few seconds)."""
        self.check_kill()
        self.process_funds()

    def check_kill(self) -> bool:
        """Execute a kill requested via CLI/dashboard (shared state store). True if locked."""
        k = self.store.get(KILL_KEY)
        if not k:
            return False
        if not k.get("executed"):
            res = self.om.kill(close_positions=k.get("close_positions", True), reason=k.get("reason", "kill"))
            k.update(executed=True, executed_ts=self.clock(), result=res)
            self.store.set(KILL_KEY, k)
            if k.get("test"):
                self.store.set("kill.tested", {"ts": time.time(), "mode": self.mode.value, "result": res})
            self._persist()
            if self.alerter:
                self.alerter.send("kill", f"kill switch executed: {res}")
        return True

    def stop(self) -> None:
        """Mark the runner as stopped so the CLI kill switch acts directly instead of waiting."""
        self._persist()
        hb = self.store.heartbeats().get("runner", {})
        hb.pop("ts", None)
        self.store.heartbeat("runner", hb | {"stopped": True})

    def maybe_reconcile(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self.last_reconcile < self.s.execution.reconcile_interval_min * 60:
            return
        self.last_reconcile = now
        internal = {s: p.qty for s, p in self.om.ledger.open_positions().items()}
        reconcile(internal, self.broker.positions(), self.store, self.risk, self.mode,
                  self.s.execution.reconcile_tolerance)

    def check_feed(self, feed_heartbeat_age_s: float) -> None:
        """Called periodically by the live loop: stale feed -> pause entries (risk) + alert."""
        if feed_heartbeat_age_s > self.s.risk.circuit_breakers.stale_data_seconds:
            self.store.log_event("feed_stale", f"no market data for {feed_heartbeat_age_s:.0f}s", level="WARNING",
                                 mode=self.mode.value)
            if self.alerter:
                self.alerter.send("error", f"data feed stale for {feed_heartbeat_age_s:.0f}s")
