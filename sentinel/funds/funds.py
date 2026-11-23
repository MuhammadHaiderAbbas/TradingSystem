"""Funds: deposits, withdrawals and honest profit accounting.

Design rules
------------
* **Money never moves through Sentinel.** Your funds sit at the exchange/broker. Deposits arrive
  there from your bank (UK: Faster Payments), and withdrawals are sent from there to your bank by
  *you*, in the venue's app/website, approved with 2FA. Sentinel keeps using trade-only API keys
  and still refuses any key that can withdraw.
* Sentinel **records** every cash flow (manually, or read-only from the venue's funding history),
  **reserves** cash for a requested withdrawal so the bot stops trading it, and **separates
  invested capital from profit** so deposits never look like gains and withdrawals never look
  like losses (which would otherwise trip the daily-loss / drawdown halts).
* In PAPER mode deposits/withdrawals are simulated against the paper broker, so the whole
  workflow can be rehearsed with no real money.

Withdrawal lifecycle: ``requested`` (cash reserved; bot won't open new trades with it) ->
``ready`` (enough free cash, i.e. not tied up in open positions) -> ``completed`` (you sent it
on the exchange; confirmed manually or detected via the venue's withdrawal history) or
``cancelled`` (reservation released).
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel.config import FundsConfig
from sentinel.core.state import StateStore
from sentinel.monitoring.logging import get_logger

log = get_logger("funds")

OPEN_WITHDRAWAL = ("requested", "ready")


class FundsError(ValueError):
    """A funds request that can't be honoured (amount too large, wrong state, ...)."""


@dataclass
class WithdrawalCheck:
    amount: float
    free_cash: float
    withdrawable_now: float
    withdrawable_profit: float
    ready: bool
    message: str


def _sign(kind: str) -> float:
    return -1.0 if kind == "withdrawal" else 1.0


class Funds:
    def __init__(self, store: StateStore, mode: str, cfg: FundsConfig | None = None) -> None:
        self.store = store
        self.mode = mode
        self.cfg = cfg or FundsConfig()

    # ------------------------------------------------------------------ ledger
    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.store._q(sql, args)]

    def flows(self, status: str | tuple[str, ...] | None = None, kind: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM cash_flows WHERE mode=?", [self.mode]
        if status:
            st = (status,) if isinstance(status, str) else status
            sql += f" AND status IN ({','.join('?' * len(st))})"
            args += list(st)
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        return self._rows(sql + " ORDER BY ts, id", tuple(args))

    def get(self, flow_id: int) -> dict:
        rows = self._rows("SELECT * FROM cash_flows WHERE id=? AND mode=?", (flow_id, self.mode))
        if not rows:
            raise FundsError(f"no cash flow #{flow_id} in {self.mode}")
        return rows[0]

    def record(self, kind: str, amount: float, status: str = "completed", source: str = "manual",
               external_id: str | None = None, note: str = "", ts: float | None = None,
               applied: bool = False) -> int | None:
        if kind not in ("opening", "deposit", "withdrawal"):
            raise FundsError(f"unknown cash-flow kind {kind!r}")
        if not amount or amount <= 0:
            raise FundsError("amount must be positive")
        now = ts if ts is not None else time.time()
        with self.store._lock:
            cur = self.store._conn.execute(
                "INSERT OR IGNORE INTO cash_flows(ts,mode,kind,amount,currency,status,source,external_id,applied,note,"
                "updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (now, self.mode, kind, float(amount), self.cfg.currency, status, source, external_id, int(applied),
                 note, now))
            new_id = cur.lastrowid if cur.rowcount else None
        if new_id:
            self.store.log_event("funds", f"{kind} {amount:,.2f} {self.cfg.currency} ({status}, {source})",
                                 mode=self.mode, data={"id": new_id, "note": note})
        return new_id

    def _update(self, flow_id: int, **fields: Any) -> None:
        fields["updated_ts"] = fields.get("updated_ts", time.time())
        sets = ", ".join(f"{k}=?" for k in fields)
        self.store._x(f"UPDATE cash_flows SET {sets} WHERE id=? AND mode=?", (*fields.values(), flow_id, self.mode))

    def ensure_opening(self, equity: float, ts: float | None = None) -> None:
        """Record the opening balance the first time a mode runs: the current equity minus any flows that
        are already reflected in it (call this *before* applying pending flows)."""
        if self.flows(kind="opening"):
            return
        already = sum(_sign(f["kind"]) * f["amount"] for f in self.flows("completed") if f["applied"])
        opening = equity - already
        if opening > 0:
            self.record("opening", opening, "completed", "system", note="opening balance", ts=ts, applied=True)

    # ------------------------------------------------------------------ deposits
    def deposit(self, amount: float, source: str = "manual", note: str = "", external_id: str | None = None,
                ts: float | None = None) -> int | None:
        """Record money that arrived at the venue (PAPER: simulated deposit). Applied to accounting on the
        bot's next tick (or immediately by the CLI if the bot isn't running)."""
        return self.record("deposit", amount, "completed", source, external_id, note, ts)

    # ------------------------------------------------------------------ withdrawals
    def reserved(self) -> float:
        return sum(f["amount"] for f in self.flows(OPEN_WITHDRAWAL, "withdrawal"))

    def check_withdrawal(self, amount: float, equity: float, cash: float, profit_only: bool = False) -> WithdrawalCheck:
        s = self.summary(equity, cash)
        free = max(cash - self.reserved() - self.cfg.keep_minimum_balance, 0.0)
        if amount < self.cfg.min_withdrawal:
            raise FundsError(f"minimum withdrawal is {self.cfg.min_withdrawal:,.2f} {self.cfg.currency}")
        limit = s["withdrawable_profit"] if profit_only else max(equity - self.reserved() - self.cfg.keep_minimum_balance, 0)
        if amount > limit + 1e-9:
            what = "profit available" if profit_only else "account value available"
            raise FundsError(f"requested {amount:,.2f} but only {limit:,.2f} {self.cfg.currency} {what}")
        ready = amount <= free + 1e-9
        msg = ("enough free cash: you can send it now" if ready else
               f"only {free:,.2f} {self.cfg.currency} is free cash; the rest is in open positions. "
               "The bot will stop opening new trades with the reserved amount; it becomes ready as positions close "
               "(or close them yourself / use `sentinel kill`).")
        return WithdrawalCheck(amount, free, s["withdrawable_now"], s["withdrawable_profit"], ready, msg)

    def request_withdrawal(self, amount: float, equity: float, cash: float, profit_only: bool = False,
                           note: str = "", ts: float | None = None) -> tuple[int, WithdrawalCheck]:
        chk = self.check_withdrawal(amount, equity, cash, profit_only)
        fid = self.record("withdrawal", amount, "ready" if chk.ready else "requested", "manual",
                          note=note or ("profit withdrawal" if profit_only else "withdrawal"), ts=ts)
        return fid, chk

    def refresh_ready(self, cash: float) -> None:
        """Promote requested withdrawals to ready once enough cash is free (oldest first)."""
        free = max(cash - self.cfg.keep_minimum_balance, 0.0)
        for f in self.flows(OPEN_WITHDRAWAL, "withdrawal"):
            if f["status"] == "requested" and f["amount"] <= free + 1e-9:
                self._update(f["id"], status="ready")
            free -= f["amount"]

    def confirm_withdrawal(self, flow_id: int, external_id: str | None = None, ts: float | None = None) -> None:
        """You sent the money to your bank on the exchange (or it was detected in the venue history)."""
        f = self.get(flow_id)
        if f["kind"] != "withdrawal" or f["status"] not in OPEN_WITHDRAWAL:
            raise FundsError(f"#{flow_id} is not an open withdrawal (status {f['status']})")
        self._update(flow_id, status="completed", external_id=external_id or f["external_id"],
                     updated_ts=ts if ts is not None else time.time())
        self.store.log_event("funds", f"withdrawal #{flow_id} of {f['amount']:,.2f} {f['currency']} completed",
                             mode=self.mode)

    def cancel_withdrawal(self, flow_id: int) -> None:
        f = self.get(flow_id)
        if f["kind"] != "withdrawal" or f["status"] not in OPEN_WITHDRAWAL:
            raise FundsError(f"#{flow_id} is not an open withdrawal")
        self._update(flow_id, status="cancelled")

    # ------------------------------------------------------------------ applying flows to accounting
    def apply_pending(self, broker=None, risk=None, ledger=None, ts: float | None = None) -> list[dict]:
        """Apply completed-but-unapplied flows: shift the risk manager's reference equities (deposits and
        withdrawals are not P&L) and, in PAPER mode, move the simulated cash."""
        from sentinel.execution.sim import SimBroker

        done = []
        for f in self._rows("SELECT * FROM cash_flows WHERE mode=? AND status='completed' AND applied=0 "
                            "AND kind!='opening' ORDER BY id", (self.mode,)):
            signed = _sign(f["kind"]) * f["amount"]
            if isinstance(broker, SimBroker):  # paper/backtest: simulate the money moving
                broker.portfolio.cash += signed
                broker.portfolio.initial_cash += signed
            if ledger is not None:
                ledger.cash += signed
            if risk is not None:
                risk.on_cash_flow(signed)
            self._update(f["id"], applied=1, updated_ts=ts if ts is not None else time.time())
            done.append(f)
        return done

    # ------------------------------------------------------------------ accounting
    def summary(self, equity: float | None, cash: float | None, equity_curve: list[dict] | None = None) -> dict:
        flows = self.flows("completed")
        opening = sum(f["amount"] for f in flows if f["kind"] == "opening")
        deposits = sum(f["amount"] for f in flows if f["kind"] == "deposit")
        withdrawals = sum(f["amount"] for f in flows if f["kind"] == "withdrawal")
        net = opening + deposits - withdrawals
        eq = equity if equity is not None else net
        cash = cash if cash is not None else eq
        profit = eq - net
        reserved = self.reserved()
        withdrawable_now = max(cash - reserved - self.cfg.keep_minimum_balance, 0.0)
        withdrawable_profit = max(min(withdrawable_now, profit - reserved), 0.0)
        suggestion = None
        if self.cfg.profit_take_pct and net > 0 and profit > net * self.cfg.profit_take_pct and withdrawable_profit >= \
                self.cfg.min_withdrawal:
            suggestion = (f"Profit is {profit / net:.1%} of invested capital (target {self.cfg.profit_take_pct:.0%}). "
                          f"Consider withdrawing up to {withdrawable_profit:,.2f} {self.cfg.currency}.")
        return {
            "currency": self.cfg.currency, "opening": opening, "deposits": deposits, "withdrawals": withdrawals,
            "net_invested": net, "equity": eq, "cash": cash, "profit": profit,
            "profit_pct": profit / net if net > 0 else 0.0,
            "time_weighted_return": self.twr(equity_curve) if equity_curve else None,
            "reserved_for_withdrawal": reserved, "withdrawable_now": withdrawable_now,
            "withdrawable_profit": withdrawable_profit, "suggestion": suggestion, "bank_label": self.cfg.bank_label,
        }

    def twr(self, equity_curve: list[dict]) -> float | None:
        """Time-weighted return: performance of the strategy independent of when you added/removed money."""
        if not equity_curve or len(equity_curve) < 2:
            return None
        flows = [(f["updated_ts"], _sign(f["kind"]) * f["amount"]) for f in self.flows("completed")
                 if f["kind"] != "opening" and f["applied"]]
        growth = 1.0
        for a, b in zip(equity_curve[:-1], equity_curve[1:], strict=True):
            f_amt = sum(amt for t, amt in flows if a["ts"] < t <= b["ts"])
            if a["equity"] > 0:
                growth *= (b["equity"] - f_amt) / a["equity"]
        return growth - 1.0

    # ------------------------------------------------------------------ venue sync (read-only)
    def sync_from_exchange(self, exchange, since_ms: int | None = None) -> dict:
        """Read the venue's deposit/withdrawal history (needs only *query funds* permission, never
        withdrawal permission). New deposits are recorded; withdrawals are matched to open requests."""
        key = f"funds.sync_since.{self.mode}"
        since = since_ms if since_ms is not None else self.store.get(key)
        cur = self.cfg.currency
        added = matched = external = 0
        latest = since or 0
        for d in exchange.fetch_deposits(cur, since) or []:
            latest = max(latest, int(d.get("timestamp") or 0))
            if d.get("status") not in ("ok", "complete", "completed"):
                continue
            if self.record("deposit", float(d["amount"]), "completed", "exchange", f"dep:{d['id']}",
                           note=f"{exchange.id} deposit", ts=(d.get("timestamp") or 0) / 1000 or None):
                added += 1
        open_w = self.flows(OPEN_WITHDRAWAL, "withdrawal")
        for w in exchange.fetch_withdrawals(cur, since) or []:
            latest = max(latest, int(w.get("timestamp") or 0))
            if w.get("status") not in ("ok", "complete", "completed"):
                continue
            ext = f"wd:{w['id']}"
            if self._rows("SELECT id FROM cash_flows WHERE mode=? AND external_id=?", (self.mode, ext)):
                continue
            amt = float(w["amount"]) + float((w.get("fee") or {}).get("cost") or 0)
            match = next((f for f in open_w if abs(f["amount"] - amt) <= max(0.02 * f["amount"], 1.0)), None)
            if match:
                self.confirm_withdrawal(match["id"], ext)
                open_w.remove(match)
                matched += 1
            else:  # made outside Sentinel: record it so accounting stays right
                self.record("withdrawal", amt, "completed", "exchange", ext, note="withdrawal made outside Sentinel")
                external += 1
        self.store.set(key, latest)
        return {"deposits_added": added, "withdrawals_matched": matched, "withdrawals_external": external}

    # ------------------------------------------------------------------ guidance & reporting
    def instructions(self, venue: str, amount: float) -> str:
        bank = self.cfg.bank_label or "your verified UK bank account"
        cur = self.cfg.currency
        common = [
            f"1. Log in to {venue.capitalize()} yourself (app or website). Sentinel cannot and will not send money.",
            f"2. Open Funding / Withdraw, choose {cur}, and pick {bank} (Faster Payments for GBP).",
            f"3. Enter {amount:,.2f} {cur}. Check the venue's withdrawal fee and limits on that screen.",
            "4. Approve with your 2FA / email confirmation.",
            "5. Sentinel marks the request completed when it sees the withdrawal in the venue's history "
            "(read-only sync), or run `sentinel funds confirm <id>`.",
        ]
        if venue in ("kraken", "coinbase", "coinbaseadvanced"):
            common.insert(1, f"   ({venue.capitalize()} supports GBP withdrawals to UK bank accounts; menu names can change.)")
        return "\n".join(common)

    def deposit_instructions(self, venue: str) -> str:
        cur = self.cfg.currency
        return "\n".join([
            f"1. In {venue.capitalize()}, open Funding / Deposit and choose {cur}; it shows the bank details and any "
            "reference you must include (Faster Payments for GBP).",
            "2. Send the money from your own bank account (same name as your exchange account).",
            "3. Once it arrives, Sentinel's read-only sync records it automatically, or run "
            "`sentinel funds deposit <amount>` to record it yourself.",
            "4. New money is NOT profit: it raises your invested capital, and the risk limits adjust.",
        ])

    def export_csv(self, path: str | Path, year: int | None = None) -> Path:
        """Cash flows + closed trades in one CSV, for your own records / tax software (not tax advice)."""
        rows = []
        for f in self.flows("completed"):
            rows.append({"date": _iso(f["updated_ts"] or f["ts"]), "type": f["kind"], "symbol": "", "side": "",
                         "qty": "", "price": "", "amount": f["amount"], "fees": "", "pnl": "",
                         "currency": f["currency"], "reference": f["external_id"] or f["note"]})
        for t in self.store.trades(self.mode):
            rows.append({"date": _iso(t["exit_ts"]), "type": "trade", "symbol": t["symbol"], "side": t["side"],
                         "qty": t["qty"], "price": t["exit_price"], "amount": t["qty"] * t["exit_price"],
                         "fees": t["fees"], "pnl": t["pnl"], "currency": self.cfg.currency,
                         "reference": f"entry {_iso(t['entry_ts'])} @ {t['entry_price']}"})
        if year:
            rows = [r for r in rows if r["date"].startswith(str(year))]
        rows.sort(key=lambda r: r["date"])
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["date", "type", "symbol", "side", "qty", "price", "amount", "fees", "pnl",
                                               "currency", "reference"])
            w.writeheader()
            w.writerows(rows)
        return p


def _iso(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts or 0))
