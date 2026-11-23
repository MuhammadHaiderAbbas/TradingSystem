"""Core domain types shared by every module."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class Mode(str, enum.Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, enum.Enum):
    NEW = "new"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED)


class OrderPurpose(str, enum.Enum):
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    KILL = "kill"


def new_id(prefix: str = "o") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    trail_pct: float | None = None
    purpose: OrderPurpose = OrderPurpose.ENTRY
    reduce_only: bool = False
    oco_group: str | None = None
    # Bracket children created when this (entry) order fills.
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy: str = ""
    client_id: str = field(default_factory=lambda: new_id("c"))
    id: str = field(default_factory=new_id)
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fees_paid: float = 0.0
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    exchange_id: str | None = None
    reject_reason: str = ""
    # internal: trailing-stop high/low watermark and stop-limit trigger state
    _trail_ref: float | None = None
    _triggered: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "client_id": self.client_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "type": self.type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "trail_pct": self.trail_pct,
            "purpose": self.purpose.value,
            "reduce_only": self.reduce_only,
            "oco_group": self.oco_group,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy": self.strategy,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "fees_paid": self.fees_paid,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "exchange_id": self.exchange_id,
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True)
class Fill:
    order_id: str
    client_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    ts: float
    liquidity: str = "taker"  # 'maker' | 'taker'
    purpose: OrderPurpose = OrderPurpose.ENTRY
    strategy: str = ""


@dataclass
class Bar:
    ts: float  # bar OPEN time, epoch seconds UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketRules:
    """Exchange trading rules for a symbol."""

    min_qty: float = 0.0
    step_size: float = 0.0
    tick_size: float = 0.0
    min_notional: float = 0.0
