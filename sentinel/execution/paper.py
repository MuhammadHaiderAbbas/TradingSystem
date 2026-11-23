"""Paper broker: the simulator, driven by *live* prices.

* Market orders fill immediately against the live quote (ask for buys, bid for
  sells) plus fixed slippage and the taker fee.
* Resting limit / stop / trailing orders are checked against every quote update
  and every closed bar.
* Quotes come from a ``quote_source(symbol) -> (bid, ask)`` callable, typically a
  public (key-less) ccxt ticker, so paper trading needs no API keys.
"""

from __future__ import annotations

from collections.abc import Callable

from sentinel.core.types import Bar, OrderStatus, OrderType, Side
from sentinel.execution.precision import round_qty
from sentinel.execution.sim import EPS, SimBroker

QuoteSource = Callable[[str], tuple[float, float]]


class PaperBroker(SimBroker):
    venue = "paper"

    def __init__(self, *args, quote_source: QuoteSource | None = None, clock: Callable[[], float] | None = None,
                 **kw) -> None:
        super().__init__(*args, **kw)
        self.quote_source = quote_source
        self.quotes: dict[str, tuple[float, float]] = {}
        self._clock_fn = clock

    def _now(self) -> float:
        import time

        return self._clock_fn() if self._clock_fn else time.time()

    def refresh_quote(self, symbol: str) -> tuple[float, float] | None:
        if self.quote_source is None:
            return self.quotes.get(symbol)
        try:
            bid, ask = self.quote_source(symbol)
        except Exception:
            return self.quotes.get(symbol)
        self.on_quote(symbol, bid, ask)
        return bid, ask

    def _after_submit(self, order) -> None:
        if order.type is OrderType.MARKET:
            q = self.refresh_quote(order.symbol)
            if q:
                self._fill_market_on_quote(order, *q)

    def _fill_market_on_quote(self, o, bid: float, ask: float) -> None:
        slip = self.costs.slippage_bps / 1e4
        px = ask * (1 + slip) if o.side is Side.BUY else bid * (1 - slip)
        qty = o.remaining
        if o.reduce_only:
            qty = min(qty, abs(self.portfolio.qty(o.symbol)))
        qty = round_qty(qty, self.market_rules(o.symbol))
        if qty > EPS:
            self._fill(o, qty, px, "taker", self._now())

    def on_quote(self, symbol: str, bid: float, ask: float) -> None:
        if bid <= 0 or ask <= 0 or ask < bid:
            return
        self.quotes[symbol] = (bid, ask)
        self.portfolio.mark(symbol, (bid + ask) / 2)
        now = self._now()
        for o in sorted([o for o in self.open.values() if o.symbol == symbol], key=lambda o: o.created_ts):
            if o.id not in self.open or o.status not in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                continue
            buy = o.side is Side.BUY
            px = None
            liq = "taker"
            if o.type is OrderType.MARKET:
                self._fill_market_on_quote(o, bid, ask)
                continue
            if o.type in (OrderType.LIMIT, OrderType.TAKE_PROFIT) or (o.type is OrderType.STOP_LIMIT and o._triggered):
                lim = o.limit_price or o.stop_price
                if (buy and ask <= lim) or (not buy and bid >= lim):
                    px, liq = lim, "maker"
            elif o.type is OrderType.STOP_LIMIT:
                if (buy and ask >= o.stop_price) or (not buy and bid <= o.stop_price):
                    o._triggered = True
            elif o.type in (OrderType.STOP, OrderType.TRAILING_STOP):
                stop = o.stop_price
                if o.type is OrderType.TRAILING_STOP:
                    stop = o._trail_ref * (1 + o.trail_pct) if buy else o._trail_ref * (1 - o.trail_pct)
                slip = self.costs.slippage_bps / 1e4
                if buy and ask >= stop:
                    px = ask * (1 + slip)
                elif not buy and bid <= stop:
                    px = bid * (1 - slip)
                if o.type is OrderType.TRAILING_STOP and px is None:
                    o._trail_ref = min(o._trail_ref, ask) if buy else max(o._trail_ref, bid)
            if px is not None:
                qty = o.remaining
                if o.reduce_only:
                    qty = min(qty, abs(self.portfolio.qty(symbol)))
                qty = round_qty(qty, self.market_rules(symbol))
                if qty > EPS:
                    self._fill(o, qty, px, liq, now)

    def on_bar(self, symbol: str, bar: Bar):
        # Resting orders are also checked against the closed bar's range in case quotes were missed.
        return super().on_bar(symbol, bar)
