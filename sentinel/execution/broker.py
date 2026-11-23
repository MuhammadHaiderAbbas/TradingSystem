"""Unified broker interface.

Every adapter (simulator, paper, ccxt, Alpaca) implements the same interface.

**Risk cannot be bypassed:** ``place_order`` is a template method defined *here*. It
requires an approval issued by the bound :class:`~sentinel.risk.manager.RiskManager`
and verifies it before calling the adapter's private ``_submit``. A broker that has
no risk manager bound refuses all orders. Adapters must not override ``place_order``
(enforced by ``__init_subclass__``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sentinel.core.errors import RiskVeto
from sentinel.core.types import MarketRules, Order

if TYPE_CHECKING:
    from sentinel.risk.manager import Approval, RiskManager
    from sentinel.security.key_safety import KeyPermissions


class Broker(ABC):
    venue: str = "abstract"

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if "place_order" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not override Broker.place_order (risk gate)")

    def __init__(self) -> None:
        self._risk: RiskManager | None = None

    def bind_risk(self, risk: RiskManager) -> None:
        if self._risk is not None and self._risk is not risk:
            raise RuntimeError("a different risk manager is already bound to this broker")
        self._risk = risk

    def place_order(self, order: Order, approval: Approval | None) -> Order:
        """The ONLY entry point for new orders. Verifies the risk approval first."""
        if self._risk is None:
            raise RiskVeto("no risk manager bound to broker; refusing order")
        if approval is None or not self._risk.verify_and_consume(approval, order):
            raise RiskVeto(f"order {order.client_id} lacks a valid risk approval")
        try:
            return self._submit(order)
        except Exception:
            self._risk.refund(approval, order.qty)  # nothing reached the venue: restore the budget
            raise

    def cancel_and_release(self, order: Order, approval: Approval) -> bool:
        """Cancel an order and return its *unfilled* quantity to the approval's budget
        (e.g. a passive limit that timed out before the market-order fallback)."""
        ok = self.cancel_order(order.exchange_id or order.id, order.symbol)
        if ok and self._risk is not None:
            self._risk.refund(approval, order.remaining)
        return ok

    # ---- adapter API -----------------------------------------------------------------
    @abstractmethod
    def _submit(self, order: Order) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool: ...

    @abstractmethod
    def cancel_all(self, symbol: str | None = None) -> int: ...

    @abstractmethod
    def open_orders(self, symbol: str | None = None) -> list[Order]: ...

    @abstractmethod
    def get_order(self, order_id: str | None = None, client_id: str | None = None) -> Order | None: ...

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Signed base-asset quantity per symbol, as the venue sees it."""

    @abstractmethod
    def balance(self) -> dict[str, float]:
        """{'cash': quote currency free+used, 'equity': mark-to-market equity}."""

    @abstractmethod
    def last_price(self, symbol: str) -> float: ...

    def market_rules(self, symbol: str) -> MarketRules:
        return MarketRules()

    def key_permissions(self) -> KeyPermissions:
        from sentinel.security.key_safety import KeyPermissions

        return KeyPermissions(known=True, can_trade=True, can_withdraw=False, ip_restricted=True, source="sim")

    def healthy(self) -> bool:
        return True
