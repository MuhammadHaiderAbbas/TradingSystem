"""Order manager: the single path from a trading decision to the broker.

* Every intent goes through ``RiskManager.evaluate``; vetoes are logged, approvals
  are passed to ``Broker.place_order`` (which verifies them).
* **Idempotent order IDs**: ``client_id`` is derived deterministically from the intent
  key (strategy | symbol | bar time | purpose). Re-submitting the same decision
  after a crash or reconnect returns the existing order instead of creating a duplicate.
* Retries with exponential backoff on transient broker errors, checking the venue for
  the client id before each retry (the request may have landed even if the response
  was lost).
* Order state is persisted (StateStore) so a restart can resume tracking.
* Maintains an *internal* position ledger from fills, used by reconciliation.
* Places protective stops itself for venues without native brackets.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

from sentinel.config import ExecutionConfig
from sentinel.core.errors import RiskVeto, TransientBrokerError
from sentinel.core.modes import KILL_KEY
from sentinel.core.state import StateStore
from sentinel.core.types import Fill, Mode, Order, OrderPurpose, OrderStatus, OrderType, Side
from sentinel.execution.broker import Broker
from sentinel.monitoring.logging import get_logger
from sentinel.portfolio.portfolio import Portfolio
from sentinel.risk.manager import OrderIntent, RiskContext, RiskDecision, RiskManager

log = get_logger("execution.orders")


def client_id_for(key: str, prefix: str = "snt") -> str:
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:24]}"


class OrderManager:
    def __init__(self, broker: Broker, risk: RiskManager, cfg: ExecutionConfig | None = None,
                 store: StateStore | None = None, mode: Mode = Mode.BACKTEST, initial_cash: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep, native_brackets: bool = True) -> None:
        self.broker = broker
        self.risk = risk
        self.cfg = cfg or ExecutionConfig()
        self.store = store
        self.mode = mode
        self.sleep = sleep
        self.native_brackets = native_brackets
        broker.bind_risk(risk)
        self.ledger = Portfolio(initial_cash)  # internal view, built only from our own fills
        self.decisions: list[RiskDecision] = []
        self.trade_listeners: list[Callable[[dict], None]] = []
        self._known: set[str] = set()
        self.api_errors = 0
        if hasattr(broker, "listeners"):
            broker.listeners.append(self._on_fill)

    def _persist(self) -> bool:
        return self.store is not None and self.mode is not Mode.BACKTEST

    # ---- submission ------------------------------------------------------------------
    def submit(self, intent: OrderIntent, ctx: RiskContext) -> Order | None:
        cid = client_id_for(intent.key) if intent.key else None
        if cid and (cid in self._known or (self._persist() and self.store.get_order(cid))):
            existing = self.broker.get_order(client_id=cid)
            log.info("duplicate intent ignored (idempotent)", extra={"client_id": cid})
            return existing
        decision = self.risk.evaluate(intent, ctx)
        self.decisions.append(decision)
        if not decision.approved:
            return None
        order = Order(
            symbol=intent.symbol, side=intent.side, qty=decision.qty, type=intent.order_type,
            limit_price=intent.limit_price, purpose=intent.purpose,
            reduce_only=intent.purpose is not OrderPurpose.ENTRY, strategy=intent.strategy,
        )
        if cid:
            order.client_id = cid
        if intent.purpose is OrderPurpose.ENTRY:
            order.stop_loss = intent.stop_price
            order.take_profit = intent.take_profit
            order.trail_pct = intent.trail_pct
        risk_amt = decision.qty * abs(intent.price - intent.stop_price) if intent.stop_price else 0.0
        order.meta.update({"reason": intent.reason, "features": intent.features, "risk_amount": risk_amt,
                           "risk_checks": [list(c) for c in decision.checks], "strategy": intent.strategy})
        return self._place(order, decision)

    def _place(self, order: Order, decision: RiskDecision) -> Order | None:
        attempt = 0
        while True:
            try:
                placed = self.broker.place_order(order, decision.approval)
                break
            except TransientBrokerError as e:
                self.api_errors += 1
                self.risk.on_api_error(str(e))
                found = self.broker.get_order(client_id=order.client_id)
                if found is not None:  # the request landed; don't duplicate
                    placed = found
                    break
                if attempt >= self.cfg.order_retry_max:
                    log.error("order failed after retries", extra={"client_id": order.client_id, "err": str(e)})
                    return None
                self.sleep(self.cfg.backoff_base_s * (2**attempt))
                attempt += 1
            except RiskVeto:
                raise
        self._known.add(placed.client_id)
        self._record_order(placed)
        if placed.status is OrderStatus.REJECTED:
            log.warning("order rejected by venue", extra={"client_id": placed.client_id, "why": placed.reject_reason})
        return placed

    def exit_position(self, symbol: str, reason: str, ctx: RiskContext, key: str = "",
                      purpose: OrderPurpose = OrderPurpose.EXIT) -> Order | None:
        qty = self.broker.positions().get(symbol, 0.0)
        if abs(qty) <= 1e-12:
            return None
        self.broker.cancel_all(symbol)
        side = Side.SELL if qty > 0 else Side.BUY
        intent = OrderIntent(symbol=symbol, side=side, purpose=purpose, price=ctx.prices.get(symbol, 0.0),
                             qty=abs(qty), reason=reason, key=key)
        return self.submit(intent, ctx)

    # ---- kill switch -----------------------------------------------------------------
    def kill(self, close_positions: bool = True, reason: str = "manual kill switch") -> dict:
        """Cancel all orders, optionally flatten, and lock the bot."""
        if self.store is not None:
            self.store.set(KILL_KEY, {"ts": time.time(), "reason": reason, "close_positions": close_positions})
            self.store.log_event("kill", reason, level="CRITICAL", mode=self.mode.value,
                                 data={"close_positions": close_positions})
        self.risk.halt("kill", reason)
        cancelled = self.broker.cancel_all()
        closed = []
        if close_positions:
            for sym, qty in list(self.broker.positions().items()):
                if abs(qty) <= 1e-12:
                    continue
                side = Side.SELL if qty > 0 else Side.BUY
                o = Order(symbol=sym, side=side, qty=abs(qty), purpose=OrderPurpose.KILL, reduce_only=True,
                          strategy="kill", client_id=client_id_for(f"kill|{sym}|{time.time()}"))
                o.meta["reason"] = "kill switch"
                ap = self.risk.approve_exit(sym, side, abs(qty), OrderPurpose.KILL)
                try:
                    self._known.add(o.client_id)
                    self._record_order(self.broker.place_order(o, ap))
                    closed.append(sym)
                except Exception as e:  # keep going: cancel/close everything we can
                    log.error("kill: failed to close position", extra={"symbol": sym, "err": str(e)})
        log.critical("KILL SWITCH ENGAGED", extra={"cancelled": cancelled, "closed": closed})
        return {"cancelled": cancelled, "closed": closed}

    # ---- fills -----------------------------------------------------------------------
    def _on_fill(self, fill: Fill, order: Order, venue_trade: dict | None) -> None:
        trade = self.ledger.apply_fill(fill, order.meta | ({"stop_price": order.stop_loss}
                                                          if order.purpose is OrderPurpose.ENTRY else {}))
        self._record_order(order)
        if self._persist():
            self.store.record_fill(self.mode.value, {
                "client_id": fill.client_id, "symbol": fill.symbol, "side": fill.side.value, "qty": fill.qty,
                "price": fill.price, "fee": fill.fee, "ts": fill.ts, "liquidity": fill.liquidity,
                "purpose": fill.purpose.value, "strategy": fill.strategy})
        if (not self.native_brackets and order.purpose is OrderPurpose.ENTRY and order.stop_loss
                and order.status is OrderStatus.FILLED):
            self._place_protective_stop(order)
        if trade is not None:
            self.risk.on_trade_closed(trade)
            if self._persist():
                self.store.record_trade(self.mode.value, trade)
            for cb in self.trade_listeners:
                cb(trade)

    def _place_protective_stop(self, parent: Order) -> None:
        side = parent.side.opposite()
        o = Order(symbol=parent.symbol, side=side, qty=parent.filled_qty, type=OrderType.STOP,
                  stop_price=parent.stop_loss, purpose=OrderPurpose.STOP_LOSS, reduce_only=True,
                  strategy=parent.strategy, client_id=f"{parent.client_id}-sl", oco_group=f"oco-{parent.id}")
        ap = self.risk.approve_exit(parent.symbol, side, parent.filled_qty, OrderPurpose.STOP_LOSS)
        try:
            self._record_order(self.broker.place_order(o, ap))
        except Exception as e:
            # An unprotected position is unacceptable: flatten it.
            log.critical("protective stop failed; flattening", extra={"symbol": parent.symbol, "err": str(e)})
            self.risk.halt("protection_failed", f"could not place stop for {parent.symbol}: {e}")
            flat = Order(symbol=parent.symbol, side=side, qty=parent.filled_qty, purpose=OrderPurpose.EXIT,
                         reduce_only=True, client_id=f"{parent.client_id}-flat")
            self.broker.place_order(flat, self.risk.approve_exit(parent.symbol, side, parent.filled_qty))

    def _record_order(self, order: Order) -> None:
        if self._persist():
            self.store.upsert_order(self.mode.value, order.to_dict())

    # ---- recovery --------------------------------------------------------------------
    def recover(self) -> int:
        """After a restart: reload known client ids and refresh state of open orders from the venue."""
        if self.store is None:
            return 0
        n = 0
        for d in self.store.orders(self.mode.value, open_only=False, limit=10_000):
            self._known.add(d["client_id"])
        for d in self.store.orders(self.mode.value, open_only=True):
            live = self.broker.get_order(client_id=d["client_id"])
            if live is not None:
                self._record_order(live)
                n += 1
        return n
