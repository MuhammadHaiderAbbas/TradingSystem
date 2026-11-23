"""Exchange precision handling: tick size, lot (step) size, min qty, min notional."""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from sentinel.core.types import MarketRules


def _round_to(value: float, step: float, mode=ROUND_DOWN) -> float:
    if not step:
        return value
    q = (Decimal(str(value)) / Decimal(str(step))).quantize(Decimal(1), rounding=mode)
    return float(q * Decimal(str(step)))


def round_qty(qty: float, rules: MarketRules) -> float:
    """Round quantity DOWN to the lot step (never buy more than approved)."""
    return _round_to(abs(qty), rules.step_size, ROUND_DOWN) * (1 if qty >= 0 else -1)


def round_price(price: float, rules: MarketRules) -> float:
    return _round_to(price, rules.tick_size, ROUND_HALF_UP)


def validate(qty: float, price: float, rules: MarketRules) -> str | None:
    """Return a rejection reason, or None if the order satisfies the rules."""
    q = abs(qty)
    if q <= 0 or not math.isfinite(q):
        return "quantity must be positive"
    if rules.min_qty and q < rules.min_qty - 1e-15:
        return f"qty {q} below min_qty {rules.min_qty}"
    if rules.min_notional and q * price < rules.min_notional - 1e-9:
        return f"notional {q * price:.4f} below min_notional {rules.min_notional}"
    if rules.step_size and abs(round_qty(q, rules) - q) > 1e-12:
        return f"qty {q} not a multiple of step {rules.step_size}"
    return None


def rules_from_ccxt_market(m: dict) -> MarketRules:
    """Build MarketRules from a ccxt market dict (precision may be decimals or tick sizes)."""
    prec = m.get("precision", {}) or {}
    lim = m.get("limits", {}) or {}

    def to_step(p):
        if p is None:
            return 0.0
        return float(p) if p < 1 and p != 0 else (10 ** (-int(p)) if float(p).is_integer() else float(p))

    return MarketRules(
        min_qty=float((lim.get("amount") or {}).get("min") or 0),
        step_size=to_step(prec.get("amount")),
        tick_size=to_step(prec.get("price")),
        min_notional=float((lim.get("cost") or {}).get("min") or 0),
    )
