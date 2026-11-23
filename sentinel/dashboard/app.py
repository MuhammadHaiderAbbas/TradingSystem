"""Local dashboard (FastAPI + a single static page, no build step).

* Bound to 127.0.0.1 by default; refuses other hosts unless explicitly allowed
  (e.g. inside Docker, where the host port is itself bound to 127.0.0.1).
* HTTP Basic auth against an argon2 hash (``sentinel dashboard --set-password``).
  Refuses to start without a password.
* Read-mostly: it reads the shared state store written by the runner. The only
  write actions are the KILL SWITCH and strategy enable/disable toggles, both of which
  are executed by the runner on its next tick and logged.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path

import numpy as np
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from sentinel.config import Settings
from sentinel.core.modes import GATE_KEY, HALT_KEY, KILL_KEY
from sentinel.core.state import StateStore
from sentinel.gate.readiness import CAP_KEY, EXPECTATIONS_KEY
from sentinel.monitoring.health import system_health

PW_KEY = "dashboard.password_hash"
STATIC = Path(__file__).parent / "static"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_ph = PasswordHasher()


def set_password(store: StateStore, password: str) -> None:
    if len(password) < 10:
        raise ValueError("dashboard password must be at least 10 characters")
    store.set(PW_KEY, _ph.hash(password))


class DepositRequest(BaseModel):
    amount: float
    note: str = ""


class WithdrawRequest(BaseModel):
    amount: float
    profit_only: bool = False
    note: str = ""


class KillRequest(BaseModel):
    confirm: str
    close_positions: bool = True
    test: bool = False


def _trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "pnl": 0.0, "win_rate": None, "expectancy_pct": None}
    pnl = np.array([t["pnl"] for t in trades])
    return {"trades": len(trades), "pnl": float(pnl.sum()), "win_rate": float((pnl > 0).mean()),
            "expectancy_pct": float(np.mean([t["return_pct"] for t in trades]))}


def _max_dd(eq: list[float]) -> float:
    if len(eq) < 2:
        return 0.0
    e = np.asarray(eq)
    return float(1 - (e / np.maximum.accumulate(e)).min())


def create_app(settings: Settings, store: StateStore) -> FastAPI:
    app = FastAPI(title="Sentinel", docs_url=None, redoc_url=None, openapi_url=None)
    security = HTTPBasic()

    def auth(creds: HTTPBasicCredentials = Depends(security)) -> str:
        h = store.get(PW_KEY)
        user_ok = secrets.compare_digest(creds.username.encode(), settings.dashboard.username.encode())
        try:
            pw_ok = bool(h) and _ph.verify(h, creds.password)
        except VerifyMismatchError:
            pw_ok = False
        if not (user_ok and pw_ok):
            time.sleep(0.5)  # slow down brute force
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unauthorized", {"WWW-Authenticate": "Basic"})
        return creds.username

    @app.middleware("http")
    async def headers(request, call_next):
        resp = await call_next(request)
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                                   "script-src 'self' 'unsafe-inline'; img-src 'self' data:")
        return resp

    def mode_now() -> str:
        return store.get("mode.current") or settings.app.mode.value

    @app.get("/", response_class=HTMLResponse)
    def index(_: str = Depends(auth)):
        return (STATIC / "index.html").read_text()

    @app.get("/api/overview")
    def overview(_: str = Depends(auth)):
        mode = mode_now()
        eq = store.equity_curve(mode)
        day_ago = time.time() - 86400
        today = [e for e in eq if e["ts"] >= (eq[-1]["ts"] - 86400 if eq else day_ago)]
        pnl_today = (today[-1]["equity"] - today[0]["equity"]) if len(today) > 1 else 0.0
        bstate = store.get(f"paper.broker_state.{mode}") or {}
        positions = []
        for sym, p in (bstate.get("positions") or {}).items():
            mark = (bstate.get("marks") or {}).get(sym, p.get("avg_price"))
            positions.append({"symbol": sym, "qty": p["qty"], "avg_price": p["avg_price"], "mark": mark,
                              "unrealized": (mark - p["avg_price"]) * p["qty"], "stop": p.get("stop_price"),
                              "strategy": p.get("strategy")})
        return {"mode": mode, "equity": eq[-1]["equity"] if eq else None, "pnl_today": pnl_today,
                "positions": positions, "open_orders": store.orders(mode, open_only=True),
                "halt": store.get(HALT_KEY), "kill": store.get(KILL_KEY),
                "gate": (store.get(GATE_KEY) or {}).get("passed", False), "capital_cap": store.get(CAP_KEY)}

    @app.get("/api/equity")
    def equity(mode: str | None = None, _: str = Depends(auth)):
        return store.equity_curve(mode or mode_now())

    @app.get("/api/compare")
    def compare(_: str = Depends(auth)):
        bt = store.get(EXPECTATIONS_KEY) or {}
        out = {"backtest": {k: bt.get(k) for k in ("expectancy_pct", "oos_expectancy_pct", "win_rate", "max_drawdown",
                                                   "trades_per_week", "sharpe")}}
        for m in ("PAPER", "LIVE"):
            tr = store.trades(m)
            st = _trade_stats(tr)
            eq = [e["equity"] for e in store.equity_curve(m)]
            span_w = ((tr[-1]["exit_ts"] - tr[0]["entry_ts"]) / (7 * 86400)) if len(tr) > 1 else 0
            st.update(max_drawdown=-_max_dd(eq), trades_per_week=(len(tr) / span_w) if span_w else None)
            out[m.lower()] = st
        e_bt, e_p = out["backtest"].get("expectancy_pct"), out["paper"].get("expectancy_pct")
        out["drift_warning"] = bool(e_bt and e_p is not None and (np.sign(e_bt) != np.sign(e_p) or
                                                                  abs(e_p - e_bt) / max(abs(e_bt), 1e-9) >
                                                                  settings.gate.backtest_tolerance))
        return out

    @app.get("/api/risk")
    def risk(_: str = Depends(auth)):
        mode = mode_now()
        r = settings.risk
        st = store.get(f"risk.state.{mode}") or {}
        eq = store.equity_curve(mode)
        cur = eq[-1]["equity"] if eq else None
        exposure = eq[-1]["exposure"] if eq else 0.0

        def loss(ref):
            return (ref - cur) / ref if (cur and ref) else 0.0

        dd = 1 - cur / st["peak_equity"] if cur and st.get("peak_equity") else 0.0
        positions = (store.get(f"paper.broker_state.{mode}") or {}).get("positions") or {}
        return {
            "exposure_pct": exposure, "max_total_exposure_pct": r.max_total_exposure_pct,
            "open_positions": len(positions), "max_open_positions": r.max_open_positions,
            "daily_loss": loss(st.get("day_start_equity")), "max_daily_loss_pct": r.max_daily_loss_pct,
            "weekly_loss": loss(st.get("week_start_equity")), "max_weekly_loss_pct": r.max_weekly_loss_pct,
            "drawdown": dd, "max_drawdown_pct": r.max_drawdown_pct,
            "consecutive_losses": st.get("consecutive_losses", 0),
            "max_consecutive_losses": r.circuit_breakers.max_consecutive_losses,
            "leverage_cap": r.max_leverage, "risk_per_trade_pct": r.risk_per_trade_pct,
            "halt": store.get(HALT_KEY), "vetoes": store.events(kind="veto", limit=20),
        }

    @app.get("/api/strategies")
    def strategies(_: str = Depends(auth)):
        mode = mode_now()
        trades = store.trades(mode)
        disabled = set(store.get("strategies.disabled", []) or [])
        hb = store.heartbeats()
        signals = {k.split(".", 1)[1]: v for k, v in hb.items() if k.startswith("signal.")}
        rows = []
        for e in settings.strategies:
            mine = [t for t in trades if e.name in (t.get("strategy") or "")]
            rows.append({"name": e.name, "configured": e.enabled, "enabled": e.enabled and e.name not in disabled,
                         "params": e.params, "regimes": e.regimes, **_trade_stats(mine)})
        return {"strategies": rows, "signals": signals}

    @app.post("/api/strategies/{name}/toggle")
    def toggle(name: str, user: str = Depends(auth)):
        names = {e.name for e in settings.strategies}
        if name not in names:
            raise HTTPException(404, "unknown strategy")
        disabled = set(store.get("strategies.disabled", []) or [])
        now_disabled = name not in disabled
        (disabled.add if now_disabled else disabled.discard)(name)
        store.set("strategies.disabled", sorted(disabled))
        store.log_event("strategy_toggle", f"{name} {'disabled' if now_disabled else 'enabled'} by {user}")
        return {"name": name, "enabled": not now_disabled}

    @app.get("/api/journal")
    def journal(limit: int = 200, _: str = Depends(auth)):
        return list(reversed(store.trades(mode_now())))[:limit]

    @app.get("/api/health")
    def health(_: str = Depends(auth)):
        return system_health(store, settings.data.stale_feed_seconds)

    @app.get("/api/events")
    def events(kind: str | None = None, limit: int = 100, _: str = Depends(auth)):
        return store.events(kind=kind, limit=limit)

    # ---- funds: money stays at the exchange; these endpoints only record / reserve ----
    def _funds():
        from sentinel.funds import Funds

        return Funds(store, mode_now(), settings.funds)

    def _bal():
        mode = mode_now()
        st = store.get(f"paper.broker_state.{mode}") if mode == "PAPER" else None
        if st:
            return (st["cash"] + sum(p["qty"] * st["marks"].get(k, p["avg_price"]) for k, p in st["positions"].items()),
                    st["cash"])
        eq = store.equity_curve(mode)
        return (eq[-1]["equity"], eq[-1]["cash"]) if eq else (None, None)

    @app.get("/api/funds")
    def funds(_: str = Depends(auth)):
        f = _funds()
        eq, cash = _bal()
        out = f.summary(eq, cash, store.equity_curve(mode_now()))
        out.update(mode=mode_now(), flows=list(reversed(f.flows()))[:100], venue=settings.market.venue,
                   open_withdrawals=f.flows(("requested", "ready"), "withdrawal"))
        return out

    @app.post("/api/funds/deposit")
    def funds_deposit(req: DepositRequest, user: str = Depends(auth)):
        from sentinel.funds import FundsError

        try:
            fid = _funds().deposit(req.amount, "paper" if mode_now() == "PAPER" else "manual",
                                   req.note or f"recorded by {user}")
        except FundsError as e:
            raise HTTPException(400, str(e)) from e
        return {"id": fid, "message": "deposit recorded; the bot applies it on its next tick"}

    @app.post("/api/funds/withdraw")
    def funds_withdraw(req: WithdrawRequest, user: str = Depends(auth)):
        from sentinel.funds import FundsError

        eq, cash = _bal()
        if eq is None:
            raise HTTPException(400, "no account balance yet")
        f = _funds()
        try:
            fid, chk = f.request_withdrawal(req.amount, eq, cash, req.profit_only, req.note)
        except FundsError as e:
            raise HTTPException(400, str(e)) from e
        return {"id": fid, "ready": chk.ready, "message": chk.message,
                "instructions": f.instructions(settings.market.venue, req.amount) if mode_now() != "PAPER" else
                "PAPER mode: press Confirm to simulate the bank payout."}

    @app.post("/api/funds/{flow_id}/{action}")
    def funds_action(flow_id: int, action: str, user: str = Depends(auth)):
        from sentinel.funds import FundsError

        f = _funds()
        try:
            if action == "confirm":
                f.confirm_withdrawal(flow_id)
            elif action == "cancel":
                f.cancel_withdrawal(flow_id)
            else:
                raise HTTPException(404, "unknown action")
        except FundsError as e:
            raise HTTPException(400, str(e)) from e
        store.log_event("funds", f"withdrawal #{flow_id} {action} by {user}", mode=mode_now())
        return {"ok": True}

    @app.post("/api/kill")
    def kill(req: KillRequest, user: str = Depends(auth)):
        if req.confirm != "KILL":
            raise HTTPException(400, "type KILL to confirm")
        state = {"ts": time.time(), "reason": f"dashboard kill by {user}", "close_positions": req.close_positions,
                 "executed": False, "test": req.test}
        store.set(KILL_KEY, state)
        store.log_event("kill", state["reason"], level="CRITICAL", data=state)
        return {"ok": True, "message": "kill switch engaged; the runner executes it on its next tick",
                "state": state}

    return app


def check_bind(host: str, allow_remote: bool) -> None:
    if host not in LOOPBACK and not allow_remote:
        raise SystemExit(f"refusing to bind the dashboard to {host}: use 127.0.0.1 and an SSH tunnel / VPN "
                         "(or --allow-remote inside Docker with the port published on 127.0.0.1 only)")
