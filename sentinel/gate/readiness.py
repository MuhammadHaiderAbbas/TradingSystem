"""LIVE readiness gate. Every check must pass before LIVE can be enabled.

Checks (thresholds in ``gate`` config):

1. >= ``min_paper_weeks`` of *continuous* paper trading (wall-clock heartbeats, no gap
   longer than ``max_heartbeat_gap_hours``)
2. >= ``min_paper_trades`` closed paper trades
3. paper per-trade expectancy within ``backtest_tolerance`` (relative) of the saved
   backtest expectation, and the same sign
4. paper max drawdown below ``max_paper_drawdown_pct``
5. positive expectancy after fees in paper AND in the saved out-of-sample backtest
6. zero unresolved reconciliation errors
7. the full test suite passes (runs pytest)
8. kill switch tested at least once in paper (``sentinel kill --test``)
9. API key verified: trade-only, no withdrawal, IP whitelisted (``sentinel keycheck``)

On success the pass is stored with the config fingerprint (see ``core.modes``); any
later change to strategy/risk/cost config invalidates it.

LIVE then starts with ``live_capital_cap``. The cap can only be raised manually, by at
most ``cap_step_max_multiplier`` per step, at least ``min_weeks_between_cap_raises``
apart, and only while live results track paper results.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentinel.config import Settings
from sentinel.core.modes import GATE_KEY, config_fingerprint
from sentinel.core.state import StateStore
from sentinel.core.types import Mode

WEEK = 7 * 86400
EXPECTATIONS_KEY = "backtest.expectations"
KILL_TESTED_KEY = "kill.tested"
KEYCHECK_KEY = "keycheck.result"
CAP_KEY = "live.capital_cap"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name, bool(passed), detail))

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.__dict__ for c in self.checks]}


def _max_dd(equity: list[float]) -> float:
    if len(equity) < 2:
        return 0.0
    e = np.asarray(equity, dtype=float)
    return float(1 - (e / np.maximum.accumulate(e)).min())


def continuous_span(ts: list[float], max_gap_s: float) -> float:
    """Length of the most recent run of heartbeats with no gap > max_gap_s."""
    if len(ts) < 2:
        return 0.0
    t = np.asarray(ts)
    gaps = np.where(np.diff(t) > max_gap_s)[0]
    start = t[gaps[-1] + 1] if len(gaps) else t[0]
    return float(t[-1] - start)


def run_tests(project_root: Path | None = None) -> tuple[bool, str]:
    root = project_root or Path(__file__).resolve().parents[2]
    if not (root / "tests").exists():
        return False, "tests directory not found"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(root / "tests")],
                       capture_output=True, text=True, cwd=root, timeout=1800,
                       env={"SENTINEL_IN_GATE": "1", **__import__("os").environ})
    tail = (p.stdout or p.stderr).strip().splitlines()[-1:] or [""]
    return p.returncode == 0, tail[0]


def evaluate_gate(settings: Settings, store: StateStore, now: float | None = None, run_test_suite: bool = True,
                  test_runner=run_tests) -> GateReport:
    g = settings.gate
    now = now or time.time()
    rep = GateReport()
    mode = Mode.PAPER.value

    hb = store.heartbeat_log(mode)
    span = continuous_span(hb, g.max_heartbeat_gap_hours * 3600)
    recent = bool(hb) and now - hb[-1] < g.max_heartbeat_gap_hours * 3600
    rep.add("paper_duration", span >= g.min_paper_weeks * WEEK and recent,
            f"{span / WEEK:.1f} continuous weeks (need {g.min_paper_weeks}); last heartbeat "
            f"{(now - hb[-1]) / 3600:.1f}h ago" if hb else "no paper heartbeats recorded")

    trades = store.trades(mode)
    rep.add("paper_trades", len(trades) >= g.min_paper_trades, f"{len(trades)} closed trades (need {g.min_paper_trades})")

    exp_paper = float(np.mean([t["pnl"] / max(t["entry_price"] * t["qty"], 1e-9) for t in trades])) if trades else 0.0
    bt = store.get(EXPECTATIONS_KEY)
    if bt and trades:
        e_bt = bt.get("expectancy_pct", 0.0)
        rel = abs(exp_paper - e_bt) / max(abs(e_bt), 1e-6)
        ok = rel <= g.backtest_tolerance and np.sign(exp_paper) == np.sign(e_bt)
        rep.add("paper_matches_backtest", ok, f"paper expectancy {exp_paper:.4%}/trade vs backtest {e_bt:.4%} "
                                               f"(rel diff {rel:.0%}, tolerance {g.backtest_tolerance:.0%})")
    else:
        rep.add("paper_matches_backtest", False, "no saved backtest expectations (run `sentinel backtest --save-expectations`)"
                if not bt else "no paper trades to compare")

    eq = [e["equity"] for e in store.equity_curve(mode)]
    dd = _max_dd(eq)
    rep.add("paper_drawdown", bool(eq) and dd < g.max_paper_drawdown_pct,
            f"paper max drawdown {dd:.1%} (limit {g.max_paper_drawdown_pct:.0%})")

    oos_exp = (bt or {}).get("oos_expectancy_pct")
    rep.add("positive_expectancy", bool(trades) and exp_paper > 0 and oos_exp is not None and oos_exp > 0,
            f"paper {exp_paper:.4%}/trade, out-of-sample backtest "
            f"{'n/a' if oos_exp is None else f'{oos_exp:.4%}'}/trade (both must be > 0 after fees)")

    unresolved = store.unresolved_recon_errors()
    rep.add("reconciliation_clean", not unresolved, f"{len(unresolved)} unresolved reconciliation errors")

    if run_test_suite:
        ok, tail = test_runner()
        rep.add("tests_pass", ok, tail)
    else:
        rep.add("tests_pass", False, "test suite not run")

    kt = store.get(KILL_TESTED_KEY)
    rep.add("kill_switch_tested", bool(kt), f"tested at {time.strftime('%Y-%m-%d %H:%M', time.gmtime(kt['ts']))} UTC"
            if kt else "never tested (run `sentinel kill --test` while paper trading)")

    kc = store.get(KEYCHECK_KEY) or {}
    key_ok = kc.get("can_trade") and kc.get("can_withdraw") is False and kc.get("ip_restricted") is True
    rep.add("api_key_safe", bool(key_ok), f"trade={kc.get('can_trade')} withdraw={kc.get('can_withdraw')} "
                                          f"ip_whitelist={kc.get('ip_restricted')} ({kc.get('source', 'not checked')})")
    return rep


def record_gate(settings: Settings, store: StateStore, rep: GateReport) -> None:
    store.set(GATE_KEY, {"passed": rep.passed, "ts": time.time(), "fingerprint": config_fingerprint(settings),
                         "report": rep.to_dict()})
    store.log_event("gate", f"readiness gate {'PASSED' if rep.passed else 'failed'}",
                    level="WARNING" if rep.passed else "INFO", data=rep.to_dict())
    if rep.passed and store.get(CAP_KEY) is None:
        store.set(CAP_KEY, {"cap": settings.gate.live_capital_cap, "ts": time.time(), "history": []})


# ---------------------------------------------------------------- capital cap
def current_cap(store: StateStore) -> float | None:
    c = store.get(CAP_KEY)
    return float(c["cap"]) if c else None


def raise_cap(settings: Settings, store: StateStore, new_cap: float, now: float | None = None) -> tuple[bool, str]:
    now = now or time.time()
    g = settings.gate
    c = store.get(CAP_KEY)
    if not c:
        return False, "no live capital cap yet (gate has not passed)"
    old = float(c["cap"])
    if new_cap <= old:
        c["history"].append({"ts": now, "from": old, "to": new_cap, "reason": "lowered"})
        c.update(cap=new_cap, ts=now)
        store.set(CAP_KEY, c)
        return True, f"cap lowered to {new_cap}"
    if new_cap > old * g.cap_step_max_multiplier:
        return False, f"step too large: max {old * g.cap_step_max_multiplier:.2f} (x{g.cap_step_max_multiplier})"
    if now - c["ts"] < g.min_weeks_between_cap_raises * WEEK:
        return False, f"too soon: wait {g.min_weeks_between_cap_raises} weeks between raises"
    live = store.trades(Mode.LIVE.value, since=c["ts"])
    paper = store.trades(Mode.PAPER.value)
    if len(live) < 10:
        return False, f"only {len(live)} live trades since the last change (need 10)"
    e_live = float(np.mean([t["return_pct"] for t in live]))
    e_paper = float(np.mean([t["return_pct"] for t in paper])) if paper else 0.0
    if e_live <= 0 or abs(e_live - e_paper) / max(abs(e_paper), 1e-6) > g.backtest_tolerance:
        return False, f"live expectancy {e_live:.4%} does not match paper {e_paper:.4%}"
    c["history"].append({"ts": now, "from": old, "to": new_cap})
    c.update(cap=new_cap, ts=now)
    store.set(CAP_KEY, c)
    store.log_event("cap_raise", f"live capital cap {old} -> {new_cap}", level="WARNING")
    return True, f"cap raised {old} -> {new_cap}"
