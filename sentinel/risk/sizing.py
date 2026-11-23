"""Position sizing: fixed fractional, volatility (ATR) based, and capped fractional Kelly.

All sizers return a base-asset quantity *before* exposure caps. They need a stop
distance because every position must have a stop-loss.
"""

from __future__ import annotations

import math


def fixed_fractional(equity: float, risk_pct: float, entry: float, stop: float) -> float:
    """Risk ``risk_pct`` of equity between entry and stop."""
    d = abs(entry - stop)
    if d <= 0 or equity <= 0:
        return 0.0
    return equity * risk_pct / d


def atr_target(equity: float, target_vol_pct: float, atr: float) -> float:
    """Size so that a 1-ATR move equals ``target_vol_pct`` of equity."""
    if atr <= 0 or equity <= 0 or not math.isfinite(atr):
        return 0.0
    return equity * target_vol_pct / atr


def kelly_fraction(win_rate: float, payoff: float) -> float:
    """Full-Kelly fraction f* = W - (1 - W) / R (may be negative = no edge)."""
    if payoff <= 0:
        return 0.0
    return win_rate - (1 - win_rate) / payoff


def kelly_size(equity: float, entry: float, stop: float, win_rate: float, payoff: float,
               fraction: float = 0.25, cap_pct: float = 0.02) -> float:
    """Fractional Kelly expressed as the risk fraction, capped at ``cap_pct``; 0 if no edge."""
    f = kelly_fraction(win_rate, payoff) * fraction
    f = max(0.0, min(f, cap_pct))
    return fixed_fractional(equity, f, entry, stop) if f > 0 else 0.0
