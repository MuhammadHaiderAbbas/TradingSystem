"""Operations helpers: encrypted backups, NTP time-sync check, periodic performance reports."""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import sqlite3
import struct
import tarfile
import tempfile
import time
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet

from sentinel.backtest.metrics import compute_metrics, trade_stats
from sentinel.backtest.report import CSS, _df_table, _kv_table, line_chart
from sentinel.core.state import StateStore
from sentinel.security.vault import derive_key

NTP_EPOCH_DELTA = 2208988800


# ---------------------------------------------------------------- time sync
def ntp_offset(server: str = "pool.ntp.org", timeout: float = 3.0) -> float:
    """Clock offset (server - local) in seconds via a single SNTP query."""
    pkt = b"\x1b" + 47 * b"\0"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        t0 = time.time()
        s.sendto(pkt, (server, 123))
        data, _ = s.recvfrom(48)
        t3 = time.time()
    secs, frac = struct.unpack("!II", data[40:48])
    server_time = secs - NTP_EPOCH_DELTA + frac / 2**32
    return server_time - (t0 + t3) / 2


def check_time_sync(max_offset_s: float = 1.0, servers: tuple[str, ...] = ("pool.ntp.org", "time.cloudflare.com")) -> dict:
    for srv in servers:
        try:
            off = ntp_offset(srv)
            return {"server": srv, "offset_s": off, "ok": abs(off) <= max_offset_s}
        except OSError:
            continue
    return {"server": None, "offset_s": None, "ok": False, "error": "no NTP server reachable"}


# ---------------------------------------------------------------- backups
def _sqlite_snapshot(src: Path) -> bytes:
    """Consistent copy of a live SQLite DB using the online backup API."""
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / src.name
        with sqlite3.connect(src) as a, sqlite3.connect(dst) as b:
            a.backup(b)
        return dst.read_bytes()


def create_backup(paths: list[Path], out_dir: Path, password: str) -> Path:
    """tar.gz of configs + consistent DB snapshots, encrypted with an argon2id-derived Fernet key.
    Secrets in the vault stay encrypted (the vault file is included as-is)."""
    if len(password) < 8:
        raise ValueError("backup password must be at least 8 characters")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and not f.name.endswith(("-wal", "-shm")):
                        _add(tar, f)
            else:
                _add(tar, p)
    salt = os.urandom(16)
    token = Fernet(derive_key(password, salt)).encrypt(buf.getvalue())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"sentinel-backup-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.enc"
    out.write_bytes(json.dumps({"v": 1, "salt": base64.b64encode(salt).decode()}).encode() + b"\n" + token)
    os.chmod(out, 0o600)
    return out


def _add(tar: tarfile.TarFile, f: Path) -> None:
    data = _sqlite_snapshot(f) if f.suffix == ".db" else f.read_bytes()
    info = tarfile.TarInfo(str(f))
    info.size = len(data)
    info.mtime = int(f.stat().st_mtime)
    tar.addfile(info, io.BytesIO(data))


def restore_backup(path: Path, password: str, dest: Path) -> list[str]:
    raw = Path(path).read_bytes()
    header, token = raw.split(b"\n", 1)
    salt = base64.b64decode(json.loads(header)["salt"])
    data = Fernet(derive_key(password, salt)).decrypt(token)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
        return tar.getnames()


def prune_backups(out_dir: Path, keep: int = 14) -> int:
    files = sorted(Path(out_dir).glob("sentinel-backup-*.enc"))
    for f in files[:-keep]:
        f.unlink()
    return max(0, len(files) - keep)


# ---------------------------------------------------------------- performance report
def performance_report(store: StateStore, mode: str, period_days: int, out: Path, title: str | None = None) -> Path:
    since = time.time() - period_days * 86400
    eq = store.equity_curve(mode, since=since)
    trades = pd.DataFrame(store.trades(mode, since=since))
    title = title or f"Sentinel {mode} report — last {period_days} days"
    sections = [f"<h1>{title}</h1><p class='muted'>Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
                "Net of fees and slippage. Includes losing periods; past results do not guarantee future results.</p>"]
    if len(eq) > 1:
        s = pd.Series([e["equity"] for e in eq], index=pd.to_datetime([e["ts"] for e in eq], unit="s", utc=True))
        m = compute_metrics(s, trades if len(trades) else None, 365 * 24)
        sections.append("<h2>Equity</h2>" + line_chart({"equity": s}))
        sections.append("<h2>Metrics</h2>" + _kv_table({k: m.get(k) for k in (
            "total_return", "max_drawdown", "sharpe", "trades", "win_rate", "profit_factor", "expectancy")}))
    else:
        sections.append("<p>No equity data in this period.</p>")
    if len(trades):
        st = trade_stats(trades)
        sections.append("<h2>Trades</h2>" + _kv_table(st))
        t = trades.copy()
        t["exit"] = pd.to_datetime(t["exit_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M")
        sections.append(_df_table(t[["exit", "symbol", "side", "pnl", "fees", "return_pct", "exit_reason"]].tail(200),
                                  pct_cols={"return_pct"}))
    halts = store.events(kind="halt", since=since)
    vetoes = store.events(kind="veto", since=since, limit=10_000)
    sections.append(f"<h2>Risk events</h2><p>{len(halts)} halts, {len(vetoes)} vetoes in this period.</p>")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
           f"initial-scale=1'><title>{title}</title><style>{CSS}</style></head><body><main>{''.join(sections)}"
           "</main></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out
