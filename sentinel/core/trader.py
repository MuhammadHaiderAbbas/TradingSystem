"""Trader: turns a target-exposure signal into order intents.

This exact class is used by the backtester, the paper runner and the live runner,
so what you backtest is what you trade.

Rules:
* Decisions are made at bar CLOSE; orders execute from the next bar.
* Target sign changes -> exit the current position (reduce-only market order).
* Flat and target != 0 -> entry with a mandatory ATR stop (and optional take-profit /
  trailing stop). Size is decided by the risk manager, scaled by |target|.
* After a stop-loss or take-profit exit, re-entry in the same direction waits until
  the signal resets (goes flat or flips), to avoid churning on a stale signal.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from sentinel.config import Settings
from sentinel.core.types import Mode, OrderPurpose, Side
from sentinel.execution.broker import Broker
from sentinel.execution.order_manager import OrderManager
from sentinel.risk.manager import OrderIntent, RiskContext, RiskManager

ReturnsProvider = Callable[[], dict[str, np.ndarray]]


class Trader:
    def __init__(self, settings: Settings, broker: Broker, om: OrderManager, risk: RiskManager,
                 mode: Mode = Mode.BACKTEST) -> None:
        self.s = settings
        self.broker = broker
        self.om = om
        self.risk = risk
        self.mode = mode
        self.stopped_out: dict[str, int] = {}
        self.capital_cap: float | None = None  # LIVE: equity the bot may use (readiness-gate cap)
        om.trade_listeners.append(self._on_trade)

    def _on_trade(self, trade: dict) -> None:
        if trade.get("exit_reason") in ("stop_loss", "take_profit"):
            self.stopped_out[trade["symbol"]] = 1 if trade["side"] == "long" else -1

    def context(self, ts: float, symbol: str, data_age_s: float = 0.0, vol_z: float = 0.0,
                returns: ReturnsProvider | None = None) -> RiskContext:
        bal = self.broker.balance()
        pos = self.broker.positions()
        prices = {}
        for s in set(pos) | {symbol}:
            try:
                prices[s] = self.broker.last_price(s)
            except KeyError:
                pass
        equity, cash = bal["equity"], bal["cash"]
        if self.capital_cap is not None:
            gross = sum(abs(q) * prices.get(s, 0.0) for s, q in pos.items())
            equity = min(equity, self.capital_cap)
            cash = min(cash, max(self.capital_cap - gross, 0.0))
        pending = sum(1 for o in self.broker.open_orders() if o.purpose is OrderPurpose.ENTRY and o.symbol != symbol)
        return RiskContext(ts=ts, equity=equity, cash=cash, positions=pos, prices=prices,
                           returns=returns() if returns else None, data_age_s=data_age_s, vol_z=vol_z,
                           pending_entries=pending, rules=self.broker.market_rules(symbol),
                           taker_fee=self.s.costs.taker_fee)

    def on_bar_close(self, symbol: str, ts: float, close: float, signal: float, atr: float | None,
                     vol_z: float = 0.0, data_age_s: float = 0.0, returns: ReturnsProvider | None = None,
                     features: dict | None = None, source: str = "") -> None:
        if not math.isfinite(signal):
            signal = 0.0
        pos = self.broker.positions().get(symbol, 0.0)
        cur = int(np.sign(pos)) if abs(pos) > 1e-12 else 0
        want = int(np.sign(signal))
        so = self.stopped_out.get(symbol)
        if so is not None and want != so:
            del self.stopped_out[symbol]
            so = None
        if cur != 0 and want != cur:
            ctx = self.context(ts, symbol, data_age_s, vol_z)
            self.om.exit_position(symbol, f"signal {signal:+.2f}", ctx, key=f"{source}|{symbol}|{ts:.0f}|exit")
            return
        if want == 0 or cur != 0 or so == want:
            return
        if any(o.purpose is OrderPurpose.ENTRY for o in self.broker.open_orders(symbol)):
            return
        if atr is None or not math.isfinite(atr) or atr <= 0:
            return
        r = self.s.risk
        side = Side.BUY if want > 0 else Side.SELL
        stop = close - want * r.stop_atr_mult * atr
        if stop <= 0:
            return
        tp = close + want * r.take_profit_atr_mult * atr if r.take_profit_atr_mult else None
        intent = OrderIntent(
            symbol=symbol, side=side, purpose=OrderPurpose.ENTRY, price=close, stop_price=stop, take_profit=tp,
            trail_pct=r.trailing_stop_pct, strength=abs(signal), atr=atr, strategy=source,
            reason=f"{source} signal {signal:+.2f}", features=features or {},
            key=f"{source}|{symbol}|{ts:.0f}|entry",
        )
        self.om.submit(intent, self.context(ts, symbol, data_age_s, vol_z, returns))
