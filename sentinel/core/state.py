"""Durable state store (SQLite, WAL mode).

The single source of truth that the bot, the CLI, the dashboard and the readiness
gate share: kill-switch lock, halts, orders, fills, round-trip trades, equity
snapshots, heartbeats, reconciliation errors and the event/veto log.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, mode TEXT, kind TEXT NOT NULL,
    level TEXT NOT NULL, message TEXT NOT NULL, data TEXT);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS ix_events_kind ON events(kind);
CREATE TABLE IF NOT EXISTS orders (
    client_id TEXT PRIMARY KEY, order_id TEXT, mode TEXT NOT NULL, symbol TEXT, status TEXT,
    data TEXT NOT NULL, created_ts REAL NOT NULL, updated_ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(mode, status);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT, mode TEXT NOT NULL, symbol TEXT,
    side TEXT, qty REAL, price REAL, fee REAL, ts REAL NOT NULL, data TEXT);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, mode TEXT NOT NULL, strategy TEXT, symbol TEXT,
    side TEXT, qty REAL, entry_ts REAL, exit_ts REAL, entry_price REAL, exit_price REAL,
    pnl REAL, fees REAL, return_pct REAL, entry_reason TEXT, exit_reason TEXT,
    features TEXT, risk_checks TEXT);
CREATE INDEX IF NOT EXISTS ix_trades_mode ON trades(mode, exit_ts);
CREATE TABLE IF NOT EXISTS equity (
    mode TEXT NOT NULL, ts REAL NOT NULL, equity REAL NOT NULL, cash REAL, exposure REAL,
    PRIMARY KEY (mode, ts));
CREATE TABLE IF NOT EXISTS heartbeats (
    component TEXT PRIMARY KEY, ts REAL NOT NULL, data TEXT);
CREATE TABLE IF NOT EXISTS heartbeat_log (
    mode TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_hblog ON heartbeat_log(mode, ts);
CREATE TABLE IF NOT EXISTS recon_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, mode TEXT, details TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0, resolved_ts REAL, resolution TEXT);
"""


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        if str(path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, args)
            return cur.lastrowid or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- kv ------------------------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        self._x(
            "INSERT INTO kv(key,value,updated_ts) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts",
            (key, json.dumps(value, default=str), time.time()),
        )

    def get(self, key: str, default: Any = None) -> Any:
        rows = self._q("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def delete(self, key: str) -> None:
        self._x("DELETE FROM kv WHERE key=?", (key,))

    # ---- events --------------------------------------------------------------------
    def log_event(self, kind: str, message: str, *, level: str = "INFO", mode: str | None = None,
                  data: dict | None = None, ts: float | None = None) -> int:
        return self._x(
            "INSERT INTO events(ts,mode,kind,level,message,data) VALUES(?,?,?,?,?,?)",
            (ts or time.time(), mode, kind, level, message, json.dumps(data or {}, default=str)),
        )

    def events(self, kind: str | None = None, limit: int = 200, since: float | None = None) -> list[dict]:
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if since is not None:
            sql += " AND ts>=?"
            args.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) | {"data": json.loads(r["data"] or "{}")} for r in self._q(sql, tuple(args))]

    def count_events(self, kind: str) -> int:
        return int(self._q("SELECT COUNT(*) c FROM events WHERE kind=?", (kind,))[0]["c"])

    # ---- orders / fills ------------------------------------------------------------
    def upsert_order(self, mode: str, order: dict) -> None:
        now = time.time()
        self._x(
            "INSERT INTO orders(client_id,order_id,mode,symbol,status,data,created_ts,updated_ts) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(client_id) DO UPDATE SET order_id=excluded.order_id, "
            "status=excluded.status, data=excluded.data, updated_ts=excluded.updated_ts",
            (order["client_id"], order.get("id"), mode, order.get("symbol"), order.get("status"),
             json.dumps(order, default=str), order.get("created_ts", now), now),
        )

    def get_order(self, client_id: str) -> dict | None:
        rows = self._q("SELECT data FROM orders WHERE client_id=?", (client_id,))
        return json.loads(rows[0]["data"]) if rows else None

    def orders(self, mode: str, open_only: bool = False, limit: int = 500) -> list[dict]:
        sql = "SELECT data FROM orders WHERE mode=?"
        if open_only:
            sql += " AND status IN ('new','open','partially_filled')"
        sql += " ORDER BY updated_ts DESC LIMIT ?"
        return [json.loads(r["data"]) for r in self._q(sql, (mode, limit))]

    def record_fill(self, mode: str, fill: dict) -> None:
        self._x(
            "INSERT INTO fills(client_id,mode,symbol,side,qty,price,fee,ts,data) VALUES(?,?,?,?,?,?,?,?,?)",
            (fill["client_id"], mode, fill["symbol"], fill["side"], fill["qty"], fill["price"], fill["fee"],
             fill["ts"], json.dumps(fill, default=str)),
        )

    # ---- trades --------------------------------------------------------------------
    def record_trade(self, mode: str, t: dict) -> int:
        return self._x(
            "INSERT INTO trades(mode,strategy,symbol,side,qty,entry_ts,exit_ts,entry_price,exit_price,pnl,"
            "fees,return_pct,entry_reason,exit_reason,features,risk_checks) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mode, t.get("strategy"), t.get("symbol"), t.get("side"), t.get("qty"), t.get("entry_ts"),
             t.get("exit_ts"), t.get("entry_price"), t.get("exit_price"), t.get("pnl"), t.get("fees"),
             t.get("return_pct"), t.get("entry_reason"), t.get("exit_reason"),
             json.dumps(t.get("features") or {}, default=str), json.dumps(t.get("risk_checks") or [], default=str)),
        )

    def trades(self, mode: str, limit: int = 100_000, since: float | None = None) -> list[dict]:
        sql, args = "SELECT * FROM trades WHERE mode=?", [mode]
        if since is not None:
            sql += " AND exit_ts>=?"
            args.append(since)
        sql += " ORDER BY exit_ts ASC LIMIT ?"
        args.append(limit)
        out = []
        for r in self._q(sql, tuple(args)):
            d = dict(r)
            d["features"] = json.loads(d["features"] or "{}")
            d["risk_checks"] = json.loads(d["risk_checks"] or "[]")
            out.append(d)
        return out

    # ---- equity --------------------------------------------------------------------
    def record_equity(self, mode: str, ts: float, equity: float, cash: float, exposure: float) -> None:
        self._x(
            "INSERT OR REPLACE INTO equity(mode,ts,equity,cash,exposure) VALUES(?,?,?,?,?)",
            (mode, ts, equity, cash, exposure),
        )

    def equity_curve(self, mode: str, since: float | None = None) -> list[dict]:
        sql, args = "SELECT ts,equity,cash,exposure FROM equity WHERE mode=?", [mode]
        if since is not None:
            sql += " AND ts>=?"
            args.append(since)
        return [dict(r) for r in self._q(sql + " ORDER BY ts", tuple(args))]

    # ---- heartbeats ----------------------------------------------------------------
    def heartbeat(self, component: str, data: dict | None = None, mode: str | None = None,
                  ts: float | None = None) -> None:
        now = ts or time.time()
        self._x(
            "INSERT INTO heartbeats(component,ts,data) VALUES(?,?,?) "
            "ON CONFLICT(component) DO UPDATE SET ts=excluded.ts, data=excluded.data",
            (component, now, json.dumps(data or {}, default=str)),
        )
        if mode:
            self._x("INSERT INTO heartbeat_log(mode,ts) VALUES(?,?)", (mode, now))

    def heartbeats(self) -> dict[str, dict]:
        return {r["component"]: {"ts": r["ts"], **json.loads(r["data"] or "{}")}
                for r in self._q("SELECT * FROM heartbeats")}

    def heartbeat_log(self, mode: str) -> list[float]:
        return [r["ts"] for r in self._q("SELECT ts FROM heartbeat_log WHERE mode=? ORDER BY ts", (mode,))]

    # ---- reconciliation ------------------------------------------------------------
    def add_recon_error(self, mode: str, details: dict) -> int:
        return self._x("INSERT INTO recon_errors(ts,mode,details) VALUES(?,?,?)",
                       (time.time(), mode, json.dumps(details, default=str)))

    def unresolved_recon_errors(self, mode: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM recon_errors WHERE resolved=0", []
        if mode:
            sql += " AND mode=?"
            args.append(mode)
        return [dict(r) for r in self._q(sql, tuple(args))]

    def resolve_recon_error(self, err_id: int, resolution: str) -> None:
        self._x("UPDATE recon_errors SET resolved=1, resolved_ts=?, resolution=? WHERE id=?",
                (time.time(), resolution, err_id))
