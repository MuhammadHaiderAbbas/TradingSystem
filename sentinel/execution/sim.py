"""Simulated broker: the fill model shared by the backtester and the paper broker.

Fill rules (conservative by design):

* Orders placed after bar t's close become active on bar t+1. Market orders fill at
  bar t+1's **open**, never on the signal bar.
* Market/stop fills pay half the spread + slippage (fixed bps + ``slippage_vol_k`` x
  bar range) and the taker fee. Resting limits/take-profits fill at their price with
  the maker fee, only if price trades *through* the limit (touching isn't enough).
* Stops that gap through fill at the (worse) open. Fill prices are capped to the
  bar's high/low.
* When a stop and a take-profit could both trigger in the same bar, the stop wins.
* Partial fills: at most ``max_participation`` x bar volume per bar per symbol; the
  rest stays open. Zero-volume bars fill nothing.
* Brackets: an entry with ``stop_loss``/``take_profit``/``trail_pct`` creates
  reduce-only OCO children when it fills. The stop child is live for the rest of the
  entry bar (worst case); the take-profit becomes live on the next bar.
* Funding: short positions pay ``funding_rate_8h`` pro-rata every bar.
* Exchange rules: qty rounded down to the step, prices to the tick, min qty /
  min notional rejected.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from sentinel.config import CostConfig
from sentinel.core.types import (
    Bar,
    Fill,
    MarketRules,
    Order,
    OrderPurpose,
    OrderStatus,
    OrderType,
    Side,
    new_id,
)
from sentinel.execution.broker import Broker
from sentinel.execution.precision import round_price, round_qty, validate
from sentinel.portfolio.portfolio import Portfolio

EPS = 1e-12
_PRIORITY = {OrderType.MARKET: 0, OrderType.STOP: 1, OrderType.TRAILING_STOP: 1, OrderType.STOP_LIMIT: 1,
             OrderType.LIMIT: 2, OrderType.TAKE_PROFIT: 2}

FillListener = Callable[[Fill, Order, dict | None], None]


class SimBroker(Broker):
    venue = "sim"

    def __init__(self, cash: float, costs: CostConfig, rules: MarketRules | dict[str, MarketRules] | None = None,
                 bar_seconds: int = 3600) -> None:
        super().__init__()
        self.costs = costs
        self._rules = rules or MarketRules()
        self.bar_seconds = bar_seconds
        self.portfolio = Portfolio(cash)
        self.open: dict[str, Order] = {}
        self.all_orders: dict[str, Order] = {}
        self.by_client: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.listeners: list[FillListener] = []
        self.last_bar: dict[str, Bar] = {}
        self.clock: float = 0.0  # time of the most recent event (bar open ts in backtests)

    # ---- helpers -------------------------------------------------------------------
    def market_rules(self, symbol: str) -> MarketRules:
        return self._rules.get(symbol, MarketRules()) if isinstance(self._rules, dict) else self._rules

    def last_price(self, symbol: str) -> float:
        if symbol in self.portfolio.marks:
            return self.portfolio.marks[symbol]
        if symbol in self.last_bar:
            return self.last_bar[symbol].close
        raise KeyError(f"no price for {symbol}")

    def _half_spread(self) -> float:
        return self.costs.spread_bps / 2 / 1e4

    def _slip(self, bar: Bar | None) -> float:
        s = self.costs.slippage_bps / 1e4
        if bar is not None and bar.open > 0:
            s += self.costs.slippage_vol_k * (bar.high - bar.low) / bar.open
        return s

    # ---- Broker API ----------------------------------------------------------------
    def _submit(self, order: Order) -> Order:
        if order.client_id in self.by_client:  # idempotent on client id
            return self.by_client[order.client_id]
        rules = self.market_rules(order.symbol)
        order.qty = round_qty(order.qty, rules)
        for attr in ("limit_price", "stop_price", "stop_loss", "take_profit"):
            v = getattr(order, attr)
            if v is not None:
                setattr(order, attr, round_price(v, rules))
        if order.reduce_only:
            pos = self.portfolio.qty(order.symbol)
            if pos == 0 or (pos > 0) == (order.side is Side.BUY):
                return self._reject(order, "reduce-only order with no position to reduce")
            order.qty = min(order.qty, round_qty(abs(pos), rules) or abs(pos))
        ref = order.limit_price or order.stop_price or self._ref_price(order.symbol)
        reason = validate(order.qty, ref, rules) if not order.reduce_only else (
            None if order.qty > 0 else "quantity must be positive")
        if reason:
            return self._reject(order, reason)
        if order.type is OrderType.TRAILING_STOP:
            if not order.trail_pct:
                return self._reject(order, "trailing stop needs trail_pct")
            order._trail_ref = self._ref_price(order.symbol)
        order.status = OrderStatus.OPEN
        order.meta.setdefault("active_after", self.clock)
        order.updated_ts = self.clock
        self.open[order.id] = order
        self.all_orders[order.id] = order
        self.by_client[order.client_id] = order
        self._after_submit(order)
        return order

    def _after_submit(self, order: Order) -> None:
        """Hook for the paper broker (immediate fills against live quotes)."""

    def _ref_price(self, symbol: str) -> float:
        try:
            return self.last_price(symbol)
        except KeyError:
            return 0.0

    def _reject(self, order: Order, reason: str) -> Order:
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        self.all_orders[order.id] = order
        self.by_client[order.client_id] = order
        return order

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        o = self.open.pop(order_id, None)
        if o is None:
            return False
        o.status = OrderStatus.CANCELED
        o.updated_ts = self.clock
        return True

    def cancel_all(self, symbol: str | None = None) -> int:
        ids = [i for i, o in self.open.items() if symbol is None or o.symbol == symbol]
        for i in ids:
            self.cancel_order(i)
        return len(ids)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [o for o in self.open.values() if symbol is None or o.symbol == symbol]

    def get_order(self, order_id: str | None = None, client_id: str | None = None) -> Order | None:
        if order_id:
            return self.all_orders.get(order_id)
        return self.by_client.get(client_id) if client_id else None

    def positions(self) -> dict[str, float]:
        return {s: p.qty for s, p in self.portfolio.open_positions().items()}

    def balance(self) -> dict[str, float]:
        return {"cash": self.portfolio.cash, "equity": self.portfolio.equity()}

    # ---- matching ------------------------------------------------------------------
    def _match(self, o: Order, bar: Bar, after_open: bool) -> tuple[float, str] | None:
        hs, slip = self._half_spread(), self._slip(bar)
        buy = o.side is Side.BUY
        op = bar.open
        if o.type is OrderType.MARKET:
            px = op * (1 + hs + slip) if buy else op * (1 - hs - slip)
            return (min(px, bar.high * (1 + hs + slip)) if buy else max(px, bar.low * (1 - hs - slip))), "taker"
        if o.type is OrderType.STOP_LIMIT and not o._triggered:
            if (buy and bar.high >= o.stop_price) or (not buy and bar.low <= o.stop_price):
                o._triggered = True  # becomes a resting limit from the next bar (conservative)
            return None
        if o.type in (OrderType.LIMIT, OrderType.TAKE_PROFIT, OrderType.STOP_LIMIT):
            lim = o.limit_price if o.type is not OrderType.TAKE_PROFIT else (o.limit_price or o.stop_price)
            if buy:
                if not after_open and op <= lim:
                    return min(op * (1 + hs), lim), "taker"
                if bar.low < lim:
                    return lim, "maker"
            else:
                if not after_open and op >= lim:
                    return max(op * (1 - hs), lim), "taker"
                if bar.high > lim:
                    return lim, "maker"
            return None
        if o.type in (OrderType.STOP, OrderType.TRAILING_STOP):
            stop = o.stop_price
            if o.type is OrderType.TRAILING_STOP:
                stop = o._trail_ref * (1 + o.trail_pct) if buy else o._trail_ref * (1 - o.trail_pct)
            if buy:
                if not after_open and op >= stop:
                    return min(op * (1 + hs + slip), bar.high), "taker"
                if bar.high >= stop:
                    return min(stop * (1 + hs + slip), bar.high), "taker"
            else:
                if not after_open and op <= stop:
                    return max(op * (1 - hs - slip), bar.low), "taker"
                if bar.low <= stop:
                    return max(stop * (1 - hs - slip), bar.low), "taker"
            return None
        return None

    def on_bar(self, symbol: str, bar: Bar) -> list[Fill]:
        """Process all active orders for ``symbol`` against ``bar``; then funding + marks."""
        self.clock = bar.ts
        self.last_bar[symbol] = bar
        fills: list[Fill] = []
        budget = self.costs.max_participation * bar.volume if bar.volume > 0 else 0.0

        def eligible(after_open_pass: bool) -> list[Order]:
            if after_open_pass:
                cand = [o for o in self.open.values() if o.symbol == symbol
                        and o.meta.get("same_bar") == bar.ts and o.type in
                        (OrderType.STOP, OrderType.TRAILING_STOP)]
            else:
                cand = [o for o in self.open.values() if o.symbol == symbol
                        and o.meta.get("active_after", -1) < bar.ts and o.meta.get("same_bar") != bar.ts]
            return sorted(cand, key=lambda o: (_PRIORITY[o.type], o.created_ts))

        for after_open_pass in (False, True):
            for o in eligible(after_open_pass):
                if o.id not in self.open:
                    continue  # cancelled by OCO meanwhile
                m = self._match(o, bar, after_open_pass)
                if m is None:
                    continue
                px, liq = m
                qty = min(o.remaining, budget)
                if o.reduce_only:
                    qty = min(qty, abs(self.portfolio.qty(symbol)))
                qty = round_qty(qty, self.market_rules(symbol))
                if qty <= EPS:
                    continue
                fills.append(self._fill(o, qty, px, liq, bar.ts))
                budget -= qty
        # trailing-stop watermarks update AFTER the trigger check (uses the prior extreme)
        for o in self.open.values():
            if o.symbol == symbol and o.type is OrderType.TRAILING_STOP:
                o._trail_ref = min(o._trail_ref, bar.low) if o.side is Side.BUY else max(o._trail_ref, bar.high)
        # funding on shorts
        q = self.portfolio.qty(symbol)
        if q < 0 and self.costs.funding_rate_8h:
            self.portfolio.apply_funding(symbol, abs(q) * bar.close * self.costs.funding_rate_8h
                                         * self.bar_seconds / 28800)
        self.portfolio.mark(symbol, bar.close)
        return fills

    def _fill(self, o: Order, qty: float, px: float, liq: str, ts: float) -> Fill:
        fee_rate = self.costs.maker_fee if liq == "maker" else self.costs.taker_fee
        fee = qty * px * fee_rate
        o.avg_fill_price = (o.avg_fill_price * o.filled_qty + px * qty) / (o.filled_qty + qty)
        o.filled_qty += qty
        o.fees_paid += fee
        o.updated_ts = ts
        o.status = OrderStatus.FILLED if o.remaining <= EPS else OrderStatus.PARTIALLY_FILLED
        f = Fill(o.id, o.client_id, o.symbol, o.side, qty, px, fee, ts, liq, o.purpose, o.strategy)
        self.fills.append(f)
        trade = self.portfolio.apply_fill(f, o.meta | {"stop_price": o.stop_loss} if o.purpose is OrderPurpose.ENTRY
                                          else o.meta)
        if o.status is OrderStatus.FILLED:
            self.open.pop(o.id, None)
            if o.oco_group:
                for sib in [s for s in self.open.values() if s.oco_group == o.oco_group]:
                    self.cancel_order(sib.id)
        if o.purpose is OrderPurpose.ENTRY and (o.stop_loss or o.take_profit or o.trail_pct):
            self._sync_brackets(o, ts)
        self._clamp_reduce_only(o.symbol)
        for cb in self.listeners:
            cb(f, o, trade)
        return f

    def _sync_brackets(self, parent: Order, ts: float) -> None:
        children = parent.meta.setdefault("children", {})
        exit_side = parent.side.opposite()
        group = f"oco-{parent.id}"
        specs = []
        if parent.trail_pct:
            specs.append(("sl", OrderType.TRAILING_STOP, None, None, OrderPurpose.STOP_LOSS))
        elif parent.stop_loss:
            specs.append(("sl", OrderType.STOP, None, parent.stop_loss, OrderPurpose.STOP_LOSS))
        if parent.take_profit:
            specs.append(("tp", OrderType.TAKE_PROFIT, parent.take_profit, None, OrderPurpose.TAKE_PROFIT))
        for key, typ, lim, stop, purpose in specs:
            cid = children.get(key)
            child = self.all_orders.get(cid) if cid else None
            if child is not None and child.id in self.open:
                child.qty = parent.filled_qty
                continue
            child = Order(symbol=parent.symbol, side=exit_side, qty=parent.filled_qty, type=typ,
                          limit_price=lim, stop_price=stop, trail_pct=parent.trail_pct if key == "sl" else None,
                          purpose=purpose, reduce_only=True, oco_group=group, strategy=parent.strategy,
                          client_id=f"{parent.client_id}-{key}", id=new_id("b"))
            child.created_ts = ts
            child.status = OrderStatus.OPEN
            if typ is OrderType.TRAILING_STOP:
                child._trail_ref = parent.avg_fill_price
            # stop live for the rest of this bar; TP from the next bar
            child.meta = {"active_after": ts, "parent": parent.id, "reason": purpose.value}
            if key == "sl":
                child.meta["same_bar"] = ts
            self.open[child.id] = child
            self.all_orders[child.id] = child
            self.by_client[child.client_id] = child
            children[key] = child.id

    def _clamp_reduce_only(self, symbol: str) -> None:
        pos = abs(self.portfolio.qty(symbol))
        for o in list(self.open.values()):
            if o.symbol == symbol and o.reduce_only:
                if pos <= EPS:
                    self.cancel_order(o.id)
                elif o.remaining > pos + EPS:
                    o.qty = o.filled_qty + pos

    # ---- persistence (crash recovery for paper mode) -------------------------------
    def to_state(self) -> dict:
        p = self.portfolio
        return {
            "cash": p.cash, "initial_cash": p.initial_cash, "marks": p.marks, "fees_paid": p.fees_paid,
            "funding_paid": p.funding_paid, "realized_pnl": p.realized_pnl, "clock": self.clock,
            "positions": {s: copy.deepcopy(vars(pos)) for s, pos in p.positions.items()},
            "open_orders": [o.to_dict() | {"meta": o.meta, "_trail_ref": o._trail_ref,
                                           "_triggered": o._triggered} for o in self.open.values()],
        }

    def restore_state(self, st: dict) -> None:
        from sentinel.portfolio.portfolio import Position

        p = self.portfolio
        p.cash, p.initial_cash, p.marks = st["cash"], st["initial_cash"], dict(st["marks"])
        p.fees_paid, p.funding_paid, p.realized_pnl = st["fees_paid"], st["funding_paid"], st["realized_pnl"]
        p.positions = {s: Position(**d) for s, d in st["positions"].items()}
        self.clock = st.get("clock", 0.0)
        self.open.clear()
        for d in st["open_orders"]:
            o = order_from_dict(d)
            self.open[o.id] = o
            self.all_orders[o.id] = o
            self.by_client[o.client_id] = o


def order_from_dict(d: dict) -> Order:
    o = Order(symbol=d["symbol"], side=Side(d["side"]), qty=d["qty"], type=OrderType(d["type"]),
              limit_price=d.get("limit_price"), stop_price=d.get("stop_price"), trail_pct=d.get("trail_pct"),
              purpose=OrderPurpose(d["purpose"]), reduce_only=d.get("reduce_only", False),
              oco_group=d.get("oco_group"), stop_loss=d.get("stop_loss"), take_profit=d.get("take_profit"),
              strategy=d.get("strategy", ""), client_id=d["client_id"], id=d["id"])
    o.status = OrderStatus(d["status"])
    o.filled_qty = d.get("filled_qty", 0.0)
    o.avg_fill_price = d.get("avg_fill_price", 0.0)
    o.fees_paid = d.get("fees_paid", 0.0)
    o.created_ts = d.get("created_ts", 0.0)
    o.meta = d.get("meta", {})
    o._trail_ref = d.get("_trail_ref")
    o._triggered = d.get("_triggered", False)
    return o
