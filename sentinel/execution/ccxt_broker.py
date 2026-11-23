"""ccxt adapter (Binance, Kraken, Coinbase, ...), spot markets.

* Uses the exchange's testnet/sandbox when ``testnet`` is true.
* ``client_id`` is sent as the venue's client order id (``newClientOrderId`` on
  Binance, ``clientOrderId`` generically) so retries after a reconnect are idempotent.
* Positions for spot = free+used base balance **minus a baseline** captured the first
  time Sentinel runs on the account (so coins you already held aren't treated as bot
  positions). The baseline lives in the state store.
* Transient errors (network, rate limit, maintenance) are raised as
  ``TransientBrokerError`` for the order manager to retry.
* Protective stops are placed by the order manager after the entry fills
  (``native_brackets=False``), as ``stop_market``/``STOP_LOSS`` where supported.
"""

from __future__ import annotations

from typing import Any

from sentinel.core.errors import BrokerError, TransientBrokerError
from sentinel.core.state import StateStore
from sentinel.core.types import MarketRules, Order, OrderStatus, OrderType, Side
from sentinel.data.downloader import _is_transient
from sentinel.execution.broker import Broker
from sentinel.execution.precision import rules_from_ccxt_market
from sentinel.security.key_safety import KeyPermissions, fetch_permissions

_STATUS = {"open": OrderStatus.OPEN, "closed": OrderStatus.FILLED, "canceled": OrderStatus.CANCELED,
           "cancelled": OrderStatus.CANCELED, "expired": OrderStatus.EXPIRED, "rejected": OrderStatus.REJECTED}


class CcxtBroker(Broker):
    venue = "ccxt"

    def __init__(self, exchange: Any, symbols: list[str], store: StateStore | None = None,
                 quote_currency: str = "USDT") -> None:
        super().__init__()
        self.ex = exchange
        self.symbols = symbols
        self.store = store
        self.quote = quote_currency
        self._markets: dict | None = None
        self._orders: dict[str, Order] = {}

    def _call(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            if _is_transient(e):
                raise TransientBrokerError(str(e)) from e
            raise BrokerError(str(e)) from e

    @property
    def markets(self) -> dict:
        if self._markets is None:
            self._markets = self._call(self.ex.load_markets)
        return self._markets

    def market_rules(self, symbol: str) -> MarketRules:
        return rules_from_ccxt_market(self.markets[symbol])

    def key_permissions(self) -> KeyPermissions:
        return fetch_permissions(self.ex)

    def last_price(self, symbol: str) -> float:
        t = self._call(self.ex.fetch_ticker, symbol)
        return float(t.get("last") or t.get("close"))

    def quote_for(self, symbol: str) -> tuple[float, float]:
        t = self._call(self.ex.fetch_ticker, symbol)
        return float(t["bid"]), float(t["ask"])

    # ---- orders --------------------------------------------------------------------
    def _params(self, order: Order) -> dict:
        p: dict[str, Any] = {"clientOrderId": order.client_id}
        if getattr(self.ex, "id", "").startswith("binance"):
            p = {"newClientOrderId": order.client_id}
        if order.type in (OrderType.STOP, OrderType.STOP_LIMIT):
            p["stopPrice"] = order.stop_price
        if order.type is OrderType.TRAILING_STOP and order.trail_pct:
            p["trailingPercent"] = order.trail_pct * 100
        return p

    def _submit(self, order: Order) -> Order:
        typ = {OrderType.MARKET: "market", OrderType.LIMIT: "limit", OrderType.TAKE_PROFIT: "limit",
               OrderType.STOP: "market", OrderType.STOP_LIMIT: "limit", OrderType.TRAILING_STOP: "market"}[order.type]
        price = order.limit_price if typ == "limit" else None
        if order.type is OrderType.STOP:
            typ = "stop_market" if self.ex.has.get("createStopMarketOrder") else "market"
        amount = float(self.ex.amount_to_precision(order.symbol, order.qty))
        if price is not None:
            price = float(self.ex.price_to_precision(order.symbol, price))
        res = self._call(self.ex.create_order, order.symbol, typ, order.side.value, amount, price, self._params(order))
        self._update_from(order, res)
        self._orders[order.client_id] = order
        return order

    def _update_from(self, order: Order, res: dict) -> None:
        order.exchange_id = str(res.get("id"))
        order.status = _STATUS.get(res.get("status") or "open", OrderStatus.OPEN)
        order.filled_qty = float(res.get("filled") or 0.0)
        order.avg_fill_price = float(res.get("average") or res.get("price") or 0.0)
        fee = res.get("fee") or {}
        order.fees_paid = float(fee.get("cost") or 0.0)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        try:
            self._call(self.ex.cancel_order, order_id, symbol)
            return True
        except BrokerError:
            return False

    def cancel_all(self, symbol: str | None = None) -> int:
        n = 0
        for s in [symbol] if symbol else self.symbols:
            for o in self._call(self.ex.fetch_open_orders, s):
                if self.cancel_order(o["id"], s):
                    n += 1
        return n

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        out = []
        for s in [symbol] if symbol else self.symbols:
            for o in self._call(self.ex.fetch_open_orders, s):
                cid = o.get("clientOrderId") or ""
                known = self._orders.get(cid)
                if known:
                    self._update_from(known, o)
                    out.append(known)
        return out

    def get_order(self, order_id: str | None = None, client_id: str | None = None) -> Order | None:
        known = self._orders.get(client_id) if client_id else None
        try:
            if known and known.exchange_id:
                self._update_from(known, self._call(self.ex.fetch_order, known.exchange_id, known.symbol))
                return known
            if client_id and self.ex.has.get("fetchOpenOrders"):
                for s in self.symbols:  # search recent orders by client id (reconnect case)
                    for o in self._call(self.ex.fetch_orders, s, None, 50):
                        if o.get("clientOrderId") == client_id:
                            return known or _order_from_ccxt(o)
        except BrokerError:
            return known
        return known

    # ---- balances ------------------------------------------------------------------
    def _baseline(self, bal: dict) -> dict[str, float]:
        key = f"ccxt.baseline.{getattr(self.ex, 'id', 'x')}"
        base = self.store.get(key) if self.store else None
        if base is None:
            base = {s: float((bal.get(s.split('/')[0]) or {}).get("total") or 0.0) for s in self.symbols}
            if self.store:
                self.store.set(key, base)
        return base

    def positions(self) -> dict[str, float]:
        bal = self._call(self.ex.fetch_balance)
        base = self._baseline(bal)
        out = {}
        for s in self.symbols:
            total = float((bal.get(s.split("/")[0]) or {}).get("total") or 0.0)
            q = total - base.get(s, 0.0)
            if abs(q) > 1e-12:
                out[s] = q
        return out

    def balance(self) -> dict[str, float]:
        bal = self._call(self.ex.fetch_balance)
        cash = float((bal.get(self.quote) or {}).get("total") or 0.0)
        eq = cash + sum(q * self.last_price(s) for s, q in self.positions().items())
        return {"cash": cash, "equity": eq}

    def healthy(self) -> bool:
        try:
            st = self.ex.fetch_status()
            return (st or {}).get("status", "ok") == "ok"
        except Exception:
            return False


def _order_from_ccxt(o: dict) -> Order:
    od = Order(symbol=o["symbol"], side=Side(o["side"]), qty=float(o["amount"]),
               client_id=o.get("clientOrderId") or "", type=OrderType.LIMIT if o.get("type") == "limit" else OrderType.MARKET)
    od.exchange_id = str(o["id"])
    od.status = _STATUS.get(o.get("status") or "open", OrderStatus.OPEN)
    od.filled_qty = float(o.get("filled") or 0.0)
    od.avg_fill_price = float(o.get("average") or 0.0)
    return od
