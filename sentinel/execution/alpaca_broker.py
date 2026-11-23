"""Alpaca adapter (US stocks, paper endpoint by default) behind the same Broker interface.

Requires ``pip install 'sentinel-trader[stocks]'`` (alpaca-py). Untested against the
live API in CI (no network); covered by interface tests with a fake client.

Notes: stocks trade in sessions (not 24/7), have corporate actions (splits/dividends,
use adjusted data for research) and pattern-day-trader rules on small accounts.
Alpaca trading keys cannot withdraw funds, and Alpaca offers no IP whitelist.
"""

from __future__ import annotations

from typing import Any

from sentinel.core.errors import BrokerError, TransientBrokerError
from sentinel.core.types import MarketRules, Order, OrderStatus, OrderType, Side
from sentinel.execution.broker import Broker
from sentinel.security.key_safety import KeyPermissions

_STATUS = {"new": OrderStatus.OPEN, "accepted": OrderStatus.OPEN, "partially_filled": OrderStatus.PARTIALLY_FILLED,
           "filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELED, "expired": OrderStatus.EXPIRED,
           "rejected": OrderStatus.REJECTED, "pending_new": OrderStatus.NEW}


class AlpacaBroker(Broker):
    venue = "alpaca"

    def __init__(self, trading_client: Any | None = None, data_client: Any | None = None, api_key: str = "",
                 api_secret: str = "", paper: bool = True) -> None:
        super().__init__()
        if trading_client is None:
            try:
                from alpaca.trading.client import TradingClient  # type: ignore
            except ImportError as e:
                raise RuntimeError("alpaca-py not installed: pip install 'sentinel-trader[stocks]'") from e
            trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.tc = trading_client
        self.dc = data_client
        self.paper = paper
        self._orders: dict[str, Order] = {}

    def _call(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("timeout", "429", "temporarily", "connection", "503", "502")):
                raise TransientBrokerError(str(e)) from e
            raise BrokerError(str(e)) from e

    def key_permissions(self) -> KeyPermissions:
        return KeyPermissions(known=True, can_trade=True, can_withdraw=False, ip_restricted=False,
                              source="alpaca (trading keys cannot withdraw)")

    def market_rules(self, symbol: str) -> MarketRules:
        return MarketRules(min_qty=1.0, step_size=1.0, tick_size=0.01, min_notional=1.0)

    def _submit(self, order: Order) -> Order:
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore
        from alpaca.trading.requests import (  # type: ignore
            LimitOrderRequest,
            MarketOrderRequest,
            StopOrderRequest,
            TrailingStopOrderRequest,
        )

        common = dict(symbol=order.symbol, qty=order.qty, client_order_id=order.client_id,
                      side=OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL, time_in_force=TimeInForce.DAY)
        if order.type is OrderType.MARKET:
            req = MarketOrderRequest(**common)
        elif order.type in (OrderType.LIMIT, OrderType.TAKE_PROFIT):
            req = LimitOrderRequest(limit_price=order.limit_price, **common)
        elif order.type is OrderType.STOP:
            req = StopOrderRequest(stop_price=order.stop_price, **common)
        elif order.type is OrderType.TRAILING_STOP:
            req = TrailingStopOrderRequest(trail_percent=(order.trail_pct or 0) * 100, **common)
        else:
            raise BrokerError(f"order type {order.type} not supported on Alpaca adapter")
        res = self._call(self.tc.submit_order, req)
        self._update(order, res)
        self._orders[order.client_id] = order
        return order

    @staticmethod
    def _update(order: Order, res: Any) -> None:
        order.exchange_id = str(getattr(res, "id", ""))
        order.status = _STATUS.get(str(getattr(res, "status", "new")).split(".")[-1].lower(), OrderStatus.OPEN)
        order.filled_qty = float(getattr(res, "filled_qty", 0) or 0)
        order.avg_fill_price = float(getattr(res, "filled_avg_price", 0) or 0)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        try:
            self._call(self.tc.cancel_order_by_id, order_id)
            return True
        except BrokerError:
            return False

    def cancel_all(self, symbol: str | None = None) -> int:
        res = self._call(self.tc.cancel_orders)
        return len(res or [])

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [o for o in self._orders.values() if not o.status.is_terminal and (symbol is None or o.symbol == symbol)]

    def get_order(self, order_id: str | None = None, client_id: str | None = None) -> Order | None:
        if client_id:
            known = self._orders.get(client_id)
            try:
                res = self._call(self.tc.get_order_by_client_id, client_id)
            except BrokerError:
                return known
            if known:
                self._update(known, res)
                return known
        return None

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self._call(self.tc.get_all_positions)}

    def balance(self) -> dict[str, float]:
        a = self._call(self.tc.get_account)
        return {"cash": float(a.cash), "equity": float(a.equity)}

    def last_price(self, symbol: str) -> float:
        if self.dc is None:
            raise BrokerError("no Alpaca data client configured")
        from alpaca.data.requests import StockLatestTradeRequest  # type: ignore

        res = self._call(self.dc.get_stock_latest_trade, StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(res[symbol].price)
