"""Positions, PnL and accounting.

Cash-based accounting for spot markets (shorts are modelled as negative quantity
with cash credited on sale, i.e. a simplified fully-collateralised short):

* ``cash``    -= signed_qty * price + fee for every fill
* ``equity``   = cash + sum(qty * mark_price)
* A round-trip *trade* is recorded when a position returns to zero (or flips). Its
  net PnL includes entry fees, exit fees and funding paid while open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentinel.core.types import Fill, OrderPurpose


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    opened_ts: float = 0.0
    strategy: str = ""
    # round-trip accumulators
    gross_realized: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    max_abs_qty: float = 0.0
    exit_value: float = 0.0
    exit_qty: float = 0.0
    entry_reason: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    risk_checks: list = field(default_factory=list)
    stop_price: float | None = None
    risk_amount: float = 0.0

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "short" if self.qty < 0 else "flat"

    def to_dict(self, mark: float | None = None) -> dict:
        d = {"symbol": self.symbol, "qty": self.qty, "avg_price": self.avg_price, "side": self.side,
             "opened_ts": self.opened_ts, "strategy": self.strategy, "stop_price": self.stop_price}
        if mark is not None:
            d["mark"] = mark
            d["unrealized"] = (mark - self.avg_price) * self.qty
        return d


class Portfolio:
    def __init__(self, cash: float) -> None:
        self.initial_cash = cash
        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.marks: dict[str, float] = {}
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.realized_pnl = 0.0
        self.closed_trades: list[dict] = []

    # ---- queries -------------------------------------------------------------------
    def qty(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.qty if p else 0.0

    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if abs(p.qty) > 1e-12}

    def mark(self, symbol: str, price: float) -> None:
        self.marks[symbol] = price

    def equity(self) -> float:
        return self.cash + sum(p.qty * self.marks.get(s, p.avg_price) for s, p in self.positions.items())

    def gross_exposure(self) -> float:
        return sum(abs(p.qty) * self.marks.get(s, p.avg_price) for s, p in self.positions.items())

    def exposure_pct(self) -> float:
        eq = self.equity()
        return self.gross_exposure() / eq if eq > 0 else 0.0

    # ---- mutations -----------------------------------------------------------------
    def apply_funding(self, symbol: str, amount: float) -> None:
        """Positive amount = cost paid."""
        self.cash -= amount
        self.funding_paid += amount
        if symbol in self.positions:
            self.positions[symbol].funding += amount

    def apply_fill(self, f: Fill, meta: dict | None = None) -> dict | None:
        """Apply a fill; return a closed-trade dict if this fill closed a round trip."""
        meta = meta or {}
        signed = f.qty * f.side.sign
        self.cash -= signed * f.price + f.fee
        self.fees_paid += f.fee
        self.marks.setdefault(f.symbol, f.price)
        pos = self.positions.get(f.symbol)
        if pos is None or abs(pos.qty) < 1e-12:
            pos = Position(f.symbol, opened_ts=f.ts, strategy=f.strategy or meta.get("strategy", ""),
                           entry_reason=meta.get("reason", ""), features=meta.get("features", {}),
                           risk_checks=meta.get("risk_checks", []), stop_price=meta.get("stop_price"),
                           risk_amount=meta.get("risk_amount", 0.0))
            self.positions[f.symbol] = pos
        closed: dict | None = None
        same_dir = pos.qty == 0 or (pos.qty > 0) == (signed > 0)
        if same_dir:
            new_qty = pos.qty + signed
            pos.avg_price = (pos.avg_price * abs(pos.qty) + f.price * abs(signed)) / abs(new_qty)
            pos.qty = new_qty
            pos.fees += f.fee
            pos.max_abs_qty = max(pos.max_abs_qty, abs(new_qty))
            if meta.get("stop_price") is not None:
                pos.stop_price = meta["stop_price"]
            return None
        # reducing / closing / flipping
        close_qty = min(abs(signed), abs(pos.qty))
        direction = 1 if pos.qty > 0 else -1
        gross = (f.price - pos.avg_price) * close_qty * direction
        pos.gross_realized += gross
        self.realized_pnl += gross
        fee_close = f.fee * close_qty / abs(signed)
        pos.fees += fee_close
        pos.exit_value += f.price * close_qty
        pos.exit_qty += close_qty
        pos.qty += signed if abs(signed) <= abs(pos.qty) else -pos.qty
        if abs(pos.qty) < 1e-12:
            pos.qty = 0.0
            closed = self._close_trade(pos, f, meta)
        remainder = abs(signed) - close_qty
        if remainder > 1e-12:  # flip into a new position with the rest
            rest = Position(f.symbol, qty=remainder * (1 if signed > 0 else -1), avg_price=f.price, opened_ts=f.ts,
                            strategy=f.strategy, fees=f.fee - fee_close, max_abs_qty=remainder,
                            entry_reason=meta.get("reason", ""))
            self.positions[f.symbol] = rest
        return closed

    def _close_trade(self, pos: Position, f: Fill, meta: dict) -> dict:
        net = pos.gross_realized - pos.fees - pos.funding
        entry_notional = pos.avg_price * pos.max_abs_qty
        trade = {
            "strategy": pos.strategy, "symbol": pos.symbol,
            "side": "long" if (f.side.value == "sell") else "short",
            "qty": pos.max_abs_qty, "entry_ts": pos.opened_ts, "exit_ts": f.ts,
            "entry_price": pos.avg_price, "exit_price": pos.exit_value / pos.exit_qty if pos.exit_qty else f.price,
            "gross_pnl": pos.gross_realized, "fees": pos.fees, "funding": pos.funding, "pnl": net,
            "return_pct": net / entry_notional if entry_notional else 0.0,
            "entry_reason": pos.entry_reason,
            "exit_reason": meta.get("reason") or (f.purpose.value if isinstance(f.purpose, OrderPurpose) else ""),
            "features": pos.features, "risk_checks": pos.risk_checks, "risk_amount": pos.risk_amount,
        }
        self.closed_trades.append(trade)
        del self.positions[pos.symbol]
        return trade

    def snapshot(self) -> dict:
        return {
            "cash": self.cash, "equity": self.equity(), "exposure_pct": self.exposure_pct(),
            "fees_paid": self.fees_paid, "funding_paid": self.funding_paid, "realized_pnl": self.realized_pnl,
            "positions": {s: p.to_dict(self.marks.get(s)) for s, p in self.open_positions().items()},
        }
