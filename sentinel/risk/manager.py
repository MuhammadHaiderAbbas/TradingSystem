"""Risk manager: an independent module every order must pass through.

``evaluate(intent, ctx)`` returns a :class:`RiskDecision`. Approved decisions carry an
:class:`Approval` (HMAC-signed with a per-process secret, single use, qty-budgeted,
short-lived) that ``Broker.place_order`` verifies; without it no order reaches a venue.

Checks for entries (exits that only reduce risk are always allowed):

* kill switch / active halt / pause conditions (stale data, extreme volatility)
* mandatory stop-loss on the correct side of the price
* sizing (fixed fractional / ATR / capped fractional Kelly) x signal strength
* max risk per trade, max open positions, per-asset and total exposure, leverage,
  available cash (spot), exchange minimums
* correlation with existing positions (no stacking the same bet)

Account-level guards (``on_equity``/``on_trade_closed``/``on_api_error``): daily loss,
weekly loss, max drawdown, N consecutive losses, API error bursts -> **halt**.
A halt requires a manual re-enable after ``cooldown_minutes`` (``sentinel unlock``).
In BACKTEST mode halts clear automatically at the next day/week/cooldown, except the
drawdown halt which stays for the rest of the run.

Every veto is logged with its reason (and stored as a ``veto`` event).
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sentinel.config import RiskConfig
from sentinel.core.modes import HALT_KEY, KILL_KEY
from sentinel.core.state import StateStore
from sentinel.core.types import MarketRules, Mode, Order, OrderPurpose, OrderType, Side, new_id
from sentinel.monitoring.logging import get_logger
from sentinel.risk import sizing

log = get_logger("risk")

APPROVAL_TTL_S = 300.0
EXIT_PURPOSES = (OrderPurpose.EXIT, OrderPurpose.STOP_LOSS, OrderPurpose.TAKE_PROFIT, OrderPurpose.KILL)


@dataclass
class OrderIntent:
    symbol: str
    side: Side
    purpose: OrderPurpose
    price: float                      # reference price (last close / quote)
    stop_price: float | None = None   # mandatory for entries
    take_profit: float | None = None
    trail_pct: float | None = None
    strength: float = 1.0             # |target exposure| in (0, 1]
    atr: float | None = None
    qty: float | None = None          # exits: qty to close; entries: computed by sizing
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    strategy: str = ""
    reason: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    key: str = ""                     # idempotency key (e.g. strategy|symbol|bar_ts|purpose)


@dataclass
class RiskContext:
    ts: float
    equity: float
    cash: float
    positions: dict[str, float]               # signed qty
    prices: dict[str, float]
    returns: dict[str, np.ndarray] | None = None  # recent log returns per symbol (aligned, same length)
    data_age_s: float = 0.0
    vol_z: float = 0.0
    pending_entries: int = 0
    rules: MarketRules = field(default_factory=MarketRules)
    taker_fee: float = 0.001


@dataclass(frozen=True)
class Approval:
    nonce: str
    symbol: str
    side: str
    max_qty: float
    purpose: str
    expires: float
    sig: str


@dataclass
class RiskDecision:
    approved: bool
    qty: float = 0.0
    reason: str = ""
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    approval: Approval | None = None


class RiskManager:
    def __init__(self, cfg: RiskConfig, store: StateStore | None = None, mode: Mode = Mode.PAPER,
                 clock: Callable[[], float] | None = None, bar_seconds: int = 3600) -> None:
        self.cfg = cfg
        self.store = store
        self.mode = mode
        self.clock = clock or time.time
        self.bar_seconds = bar_seconds
        self._key = os.urandom(32)
        self._budgets: dict[str, float] = {}
        self._lock = threading.Lock()
        self.auto_reenable = mode is Mode.BACKTEST
        # account state
        self.peak_equity: float | None = None
        self.day_key: int | None = None
        self.day_start_equity: float | None = None
        self.week_key: int | None = None
        self.week_start_equity: float | None = None
        self.consecutive_losses = 0
        self.api_errors: list[float] = []
        self.trade_returns: list[float] = []  # R-multiples for Kelly
        self._halt: dict | None = None
        self.vetoes: list[dict] = []
        self._veto_logged: dict[tuple[str, str], float] = {}
        self._restore()

    # ---- persistence -----------------------------------------------------------------
    def _persist(self) -> bool:
        return self.store is not None and self.mode is not Mode.BACKTEST

    def _restore(self) -> None:
        if not self._persist():
            return
        st = self.store.get(f"risk.state.{self.mode.value}") or {}
        for k in ("peak_equity", "day_key", "day_start_equity", "week_key", "week_start_equity",
                  "consecutive_losses"):
            if k in st:
                setattr(self, k, st[k])
        self.trade_returns = st.get("trade_returns", [])

    def _save(self) -> None:
        if self._persist():
            self.store.set(f"risk.state.{self.mode.value}", {
                "peak_equity": self.peak_equity, "day_key": self.day_key, "day_start_equity": self.day_start_equity,
                "week_key": self.week_key, "week_start_equity": self.week_start_equity,
                "consecutive_losses": self.consecutive_losses, "trade_returns": self.trade_returns[-500:],
            })

    # ---- halts -----------------------------------------------------------------------
    @property
    def halt_state(self) -> dict | None:
        if self._persist():
            return self.store.get(HALT_KEY)
        return self._halt

    def kill_engaged(self) -> bool:
        return bool(self._persist() and self.store.get(KILL_KEY)) or bool(self._halt and self._halt.get("kind") == "kill")

    def halt(self, kind: str, reason: str, ts: float | None = None) -> None:
        ts = ts if ts is not None else self.clock()
        state = {"kind": kind, "reason": reason, "ts": ts}
        if self.halt_state and self.halt_state.get("kind") == "drawdown" and kind != "kill":
            return  # don't overwrite the most serious halt
        self._halt = state
        if self._persist():
            self.store.set(HALT_KEY, state)
            self.store.log_event("halt", reason, level="CRITICAL", mode=self.mode.value, data=state, ts=ts)
        log.critical("TRADING HALTED", extra=state)

    def reenable(self, note: str = "", force_cooldown: bool = True) -> None:
        h = self.halt_state
        if not h:
            return
        if force_cooldown and self.clock() - h["ts"] < self.cfg.cooldown_minutes * 60:
            left = self.cfg.cooldown_minutes * 60 - (self.clock() - h["ts"])
            raise RuntimeError(f"cooldown active: {left / 60:.1f} minutes remaining")
        self._halt = None
        self.consecutive_losses = 0
        self.api_errors.clear()
        if self._persist():
            self.store.delete(HALT_KEY)
            self.store.log_event("reenable", f"trading re-enabled: {note}", mode=self.mode.value, data=h)
        self._save()

    def _maybe_auto_clear(self, ts: float) -> None:
        if not self.auto_reenable or not self._halt:
            return
        h = self._halt
        kind = h["kind"]
        if kind == "daily_loss" and int(ts // 86400) != int(h["ts"] // 86400):
            self._halt = None
        elif kind == "weekly_loss" and _week(ts) != _week(h["ts"]):
            self._halt = None
        elif kind in ("consecutive_losses", "api_errors") and ts - h["ts"] >= self.cfg.cooldown_minutes * 60:
            self._halt = None
            self.consecutive_losses = 0

    # ---- account guards ----------------------------------------------------------------
    def on_equity(self, ts: float, equity: float) -> None:
        self._maybe_auto_clear(ts)
        dk, wk = int(ts // 86400), _week(ts)
        if self.day_key != dk:
            self.day_key, self.day_start_equity = dk, equity
        if self.week_key != wk:
            self.week_key, self.week_start_equity = wk, equity
        self.peak_equity = max(self.peak_equity or equity, equity)
        c = self.cfg
        if self.day_start_equity and (self.day_start_equity - equity) / self.day_start_equity >= c.max_daily_loss_pct:
            if not self.halt_state:
                self.halt("daily_loss", f"daily loss limit {c.max_daily_loss_pct:.1%} breached", ts)
        if self.week_start_equity and (self.week_start_equity - equity) / self.week_start_equity >= c.max_weekly_loss_pct:
            if not self.halt_state or self.halt_state["kind"] == "daily_loss":
                self.halt("weekly_loss", f"weekly loss limit {c.max_weekly_loss_pct:.1%} breached", ts)
        if self.peak_equity and 1 - equity / self.peak_equity >= c.max_drawdown_pct:
            if not self.halt_state or self.halt_state["kind"] != "drawdown":
                self.halt("drawdown", f"max drawdown {c.max_drawdown_pct:.1%} breached", ts)
        self._save()

    def on_trade_closed(self, trade: dict) -> None:
        pnl = trade.get("pnl", 0.0)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        risk_amt = trade.get("risk_amount")
        if risk_amt:
            self.trade_returns.append(pnl / risk_amt)
        n = self.cfg.circuit_breakers.max_consecutive_losses
        if self.consecutive_losses >= n and not self.halt_state:
            self.halt("consecutive_losses", f"{self.consecutive_losses} losing trades in a row",
                      trade.get("exit_ts"))
        self._save()

    def on_api_error(self, err: str = "") -> None:
        now = self.clock()
        self.api_errors = [t for t in self.api_errors if now - t < 600] + [now]
        if len(self.api_errors) >= self.cfg.circuit_breakers.max_api_errors and not self.halt_state:
            self.halt("api_errors", f"{len(self.api_errors)} API errors in 10 min (exchange outage?): {err[:100]}")

    # ---- approvals ---------------------------------------------------------------------
    def _sign(self, nonce: str, symbol: str, side: str, max_qty: float, purpose: str, expires: float) -> str:
        msg = f"{nonce}|{symbol}|{side}|{max_qty:.12g}|{purpose}|{expires:.3f}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def _issue(self, symbol: str, side: Side, qty: float, purpose: OrderPurpose) -> Approval:
        nonce = new_id("a")
        expires = self.clock() + APPROVAL_TTL_S
        sig = self._sign(nonce, symbol, side.value, qty, purpose.value, expires)
        with self._lock:
            self._budgets[nonce] = qty
        return Approval(nonce, symbol, side.value, qty, purpose.value, expires, sig)

    def verify_and_consume(self, ap: Approval, order: Order) -> bool:
        good = hmac.compare_digest(ap.sig, self._sign(ap.nonce, ap.symbol, ap.side, ap.max_qty, ap.purpose, ap.expires))
        if not good or self.clock() > ap.expires:
            return False
        if ap.symbol != order.symbol or ap.side != order.side.value:
            return False
        is_exit_ap = OrderPurpose(ap.purpose) in EXIT_PURPOSES
        if is_exit_ap and not order.reduce_only:
            return False
        if not is_exit_ap and order.purpose in EXIT_PURPOSES:
            return False
        with self._lock:
            left = self._budgets.get(ap.nonce)
            if left is None or order.qty > left * (1 + 1e-9) + 1e-12:
                return False
            left -= order.qty
            if left <= 1e-12:
                self._budgets.pop(ap.nonce, None)
            else:
                self._budgets[ap.nonce] = left
        return True

    def refund(self, ap: Approval, qty: float) -> None:
        """Restore budget after a submission that provably failed before reaching the venue."""
        if self.clock() > ap.expires:
            return
        with self._lock:
            self._budgets[ap.nonce] = min(self._budgets.get(ap.nonce, 0.0) + qty, ap.max_qty)

    # ---- evaluation --------------------------------------------------------------------
    def _veto(self, intent: OrderIntent, reason: str, checks: list, ctx: RiskContext) -> RiskDecision:
        rec = {"ts": ctx.ts, "symbol": intent.symbol, "side": intent.side.value, "purpose": intent.purpose.value,
               "strategy": intent.strategy, "reason": reason}
        self.vetoes.append(rec)
        key = (intent.symbol, reason)
        last = self._veto_logged.get(key)
        if last is not None and abs(ctx.ts - last) < 3600:
            return RiskDecision(False, 0.0, reason, checks)  # counted, but don't spam the log/event store
        self._veto_logged[key] = ctx.ts
        log.info("risk veto", extra=rec)
        if self._persist():
            self.store.log_event("veto", reason, level="WARNING", mode=self.mode.value, data=rec, ts=ctx.ts)
        return RiskDecision(False, 0.0, reason, checks)

    def approve_exit(self, symbol: str, side: Side, qty: float, purpose: OrderPurpose = OrderPurpose.EXIT) -> Approval:
        """Exits / kill-switch flattening only reduce risk: always approved (reduce-only enforced)."""
        if purpose not in EXIT_PURPOSES:
            raise ValueError("approve_exit is only for risk-reducing orders")
        return self._issue(symbol, side, qty, purpose)

    def evaluate(self, intent: OrderIntent, ctx: RiskContext) -> RiskDecision:
        checks: list[tuple[str, bool, str]] = []

        def chk(name: str, ok: bool, detail: str = "") -> bool:
            checks.append((name, ok, detail))
            return ok

        if intent.purpose in EXIT_PURPOSES:
            qty = abs(intent.qty or ctx.positions.get(intent.symbol, 0.0))
            if qty <= 0:
                return self._veto(intent, "nothing to exit", checks, ctx)
            chk("exit_reduces_risk", True)
            return RiskDecision(True, qty, "exit", checks, self._issue(intent.symbol, intent.side, qty, intent.purpose))

        c, cb = self.cfg, self.cfg.circuit_breakers
        self._maybe_auto_clear(ctx.ts)
        if not chk("kill_switch", not self.kill_engaged()):
            return self._veto(intent, "kill switch engaged", checks, ctx)
        h = self.halt_state
        if not chk("not_halted", not h, (h or {}).get("reason", "")):
            return self._veto(intent, f"halted: {h['reason']}", checks, ctx)
        if not chk("data_fresh", ctx.data_age_s <= cb.stale_data_seconds, f"{ctx.data_age_s:.0f}s"):
            return self._veto(intent, f"data feed stale ({ctx.data_age_s:.0f}s)", checks, ctx)
        if not chk("volatility_normal", abs(ctx.vol_z) < cb.extreme_vol_zscore, f"z={ctx.vol_z:.2f}"):
            return self._veto(intent, f"extreme volatility (z={ctx.vol_z:.1f})", checks, ctx)
        if not chk("price_valid", ctx.equity > 0 and intent.price > 0 and math.isfinite(intent.price)):
            return self._veto(intent, "invalid price or equity", checks, ctx)
        stop = intent.stop_price
        if intent.trail_pct and stop is None:
            stop = intent.price * (1 - intent.trail_pct) if intent.side is Side.BUY else intent.price * (1 + intent.trail_pct)
        stop_ok = stop is not None and ((intent.side is Side.BUY and stop < intent.price) or
                                        (intent.side is Side.SELL and stop > intent.price))
        if not chk("stop_loss_present", stop_ok, f"stop={stop}"):
            return self._veto(intent, "mandatory stop-loss missing or on the wrong side", checks, ctx)

        open_pos = {s: q for s, q in ctx.positions.items() if abs(q) > 1e-12}
        if not chk("max_open_positions", len(open_pos) + ctx.pending_entries < c.max_open_positions,
                   f"{len(open_pos)}+{ctx.pending_entries}/{c.max_open_positions}"):
            return self._veto(intent, "max open positions reached", checks, ctx)
        if intent.symbol in open_pos:
            return self._veto(intent, "position already open in symbol (no pyramiding)", checks, ctx)

        # correlation
        if ctx.returns and intent.symbol in ctx.returns:
            r0 = ctx.returns[intent.symbol][-c.correlation_lookback:]
            for s, q in open_pos.items():
                if s not in ctx.returns:
                    continue
                r1 = ctx.returns[s][-c.correlation_lookback:]
                n = min(len(r0), len(r1))
                if n < 20:
                    continue
                a, b = r0[-n:], r1[-n:]
                if np.std(a) == 0 or np.std(b) == 0:
                    continue
                rho = float(np.corrcoef(a, b)[0, 1])
                same_dir = (q > 0) == (intent.side is Side.BUY)
                eff = rho if same_dir else -rho
                if not chk(f"correlation_{s}", eff <= c.max_correlation, f"rho={rho:.2f}"):
                    return self._veto(intent, f"too correlated with open {s} position (rho={rho:.2f})", checks, ctx)

        # sizing
        eq = ctx.equity
        strength = min(max(abs(intent.strength), 0.0), 1.0)
        base = sizing.fixed_fractional(eq, c.risk_per_trade_pct, intent.price, stop)
        if c.sizing == "atr" and intent.atr:
            qty = min(sizing.atr_target(eq, c.atr_target_vol_pct, intent.atr), base)
        elif c.sizing == "kelly":
            if len(self.trade_returns) >= 30:
                r = np.array(self.trade_returns[-200:])
                wins, losses = r[r > 0], r[r <= 0]
                w = len(wins) / len(r)
                payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() else 0.0
                qty = min(sizing.kelly_size(eq, intent.price, stop, w, payoff, c.kelly_fraction, c.kelly_cap_pct), base)
            else:
                qty = base * 0.5  # not enough history for Kelly: half the fixed-fractional size
        else:
            qty = base
        qty *= strength
        chk("risk_per_trade", True, f"risk={qty * abs(intent.price - stop) / eq:.4%} of equity")

        # exposure caps
        prices = ctx.prices
        gross = sum(abs(q) * prices.get(s, 0.0) for s, q in open_pos.items())
        cap_asset = c.max_exposure_per_asset_pct * eq / intent.price
        cap_total = max(c.max_total_exposure_pct * eq - gross, 0.0) / intent.price
        cap_lev = max(c.max_leverage * eq - gross, 0.0) / intent.price
        caps = {"asset_exposure": cap_asset, "total_exposure": cap_total, "leverage": cap_lev}
        if intent.side is Side.BUY and c.max_leverage <= 1.0:
            caps["cash"] = max(ctx.cash, 0.0) / (intent.price * (1 + ctx.taker_fee * 2))
        for name, cap in caps.items():
            if qty > cap:
                chk(f"cap_{name}", True, f"reduced {qty:.6g}->{cap:.6g}")
                qty = cap
        rules = ctx.rules
        if rules.step_size:
            from sentinel.execution.precision import round_qty

            qty = round_qty(qty, rules)
        notional = qty * intent.price
        min_ok = qty > 0 and notional >= rules.min_notional and qty >= rules.min_qty
        if not chk("exchange_minimums", min_ok, f"notional={notional:.2f}"):
            return self._veto(intent, f"size after caps below exchange minimum (notional {notional:.2f})", checks, ctx)
        ap = self._issue(intent.symbol, intent.side, qty, intent.purpose)
        return RiskDecision(True, qty, "approved", checks, ap)


def _week(ts: float) -> int:
    """ISO-week bucket (weeks start Monday 00:00 UTC; epoch day 0 was a Thursday)."""
    return int((ts // 86400 + 3) // 7)
