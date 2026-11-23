"""Smart execution for paper/live: liquidity check, limit-first with market fallback, TWAP.

All child orders are placed through ``Broker.place_order`` with the *parent's* risk
approval, whose qty budget caps the total across children -- splitting an order can
never exceed what the risk manager approved.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from sentinel.config import ExecutionConfig
from sentinel.core.types import Order, OrderStatus, OrderType, Side, new_id
from sentinel.execution.broker import Broker
from sentinel.execution.precision import round_price, round_qty
from sentinel.monitoring.logging import get_logger
from sentinel.risk.manager import Approval

log = get_logger("execution.smart")


@dataclass
class ExecReport:
    filled_qty: float
    avg_price: float
    children: list[str]
    used_fallback: bool
    skipped_reason: str = ""


def liquidity_ok(bar_volume_quote: float, cfg: ExecutionConfig) -> bool:
    return bar_volume_quote >= cfg.min_bar_volume_quote


def twap_slices(qty: float, price: float, cfg: ExecutionConfig) -> list[float]:
    if qty * price < cfg.twap_threshold_notional:
        return [qty]
    n = cfg.twap_slices
    base = qty / n
    return [base] * (n - 1) + [qty - base * (n - 1)]


class SmartExecutor:
    def __init__(self, broker: Broker, cfg: ExecutionConfig, quote: Callable[[str], tuple[float, float]],
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time) -> None:
        self.broker = broker
        self.cfg = cfg
        self.quote = quote
        self.sleep = sleep
        self.now = now

    def _child(self, parent: Order, qty: float, typ: OrderType, limit: float | None, idx: int) -> Order:
        return Order(symbol=parent.symbol, side=parent.side, qty=qty, type=typ, limit_price=limit,
                     purpose=parent.purpose, reduce_only=parent.reduce_only, strategy=parent.strategy,
                     stop_loss=parent.stop_loss, take_profit=parent.take_profit, trail_pct=parent.trail_pct,
                     client_id=f"{parent.client_id}-{idx}{typ.value[0]}", id=new_id("x"), meta=dict(parent.meta))

    def execute(self, parent: Order, approval: Approval) -> ExecReport:
        rules = self.broker.market_rules(parent.symbol)
        bid, ask = self.quote(parent.symbol)
        ref = ask if parent.side is Side.BUY else bid
        slices = twap_slices(parent.qty, ref, self.cfg)
        filled, notional, children, fallback = 0.0, 0.0, [], False
        for k, q in enumerate(slices):
            q = round_qty(q, rules)
            if q <= 0:
                continue
            got, px, ids, fb = self._limit_then_market(parent, q, approval, k)
            filled += got
            notional += got * px
            children += ids
            fallback |= fb
            if k < len(slices) - 1:
                self.sleep(self.cfg.twap_interval_s)
        return ExecReport(filled, notional / filled if filled else 0.0, children, fallback)

    def _limit_then_market(self, parent: Order, qty: float, ap: Approval, k: int) -> tuple[float, float, list, bool]:
        ids: list[str] = []
        filled, notional = 0.0, 0.0
        if self.cfg.limit_first:
            bid, ask = self.quote(parent.symbol)
            off = self.cfg.limit_offset_bps / 1e4
            # Passive price: join the best bid (buy) / ask (sell), slightly inside the spread.
            px = bid * (1 + off) if parent.side is Side.BUY else ask * (1 - off)
            px = min(px, ask) if parent.side is Side.BUY else max(px, bid)
            px = round_price(px, self.broker.market_rules(parent.symbol))
            lim = self.broker.place_order(self._child(parent, qty, OrderType.LIMIT, px, k), ap)
            ids.append(lim.client_id)
            deadline = self.now() + self.cfg.limit_timeout_s
            while self.now() < deadline:
                cur = self.broker.get_order(client_id=lim.client_id) or lim
                if cur.status is OrderStatus.FILLED:
                    break
                self.sleep(1.0)
            cur = self.broker.get_order(client_id=lim.client_id) or lim
            if cur.status is not OrderStatus.FILLED:
                self.broker.cancel_and_release(cur, ap)
                cur = self.broker.get_order(client_id=lim.client_id) or cur
            filled += cur.filled_qty
            notional += cur.filled_qty * (cur.avg_fill_price or px)
        rest = round_qty(qty - filled, self.broker.market_rules(parent.symbol))
        used_fb = False
        if rest > 0:
            used_fb = self.cfg.limit_first
            mkt = self.broker.place_order(self._child(parent, rest, OrderType.MARKET, None, k), ap)
            ids.append(mkt.client_id)
            mkt = self.broker.get_order(client_id=mkt.client_id) or mkt
            filled += mkt.filled_qty
            notional += mkt.filled_qty * mkt.avg_fill_price
        return filled, (notional / filled if filled else 0.0), ids, used_fb
