"""OHLCV storage: SQLite (default, time-indexed WITHOUT ROWID table) + Parquet export.

Timestamps are stored as integer milliseconds since epoch, UTC, bar OPEN time.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pandas as pd

from sentinel.data.timeframes import to_utc_index

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, ts INTEGER NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
    flags INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, timeframe, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, rate REAL NOT NULL, PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS orderbook (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, bids TEXT NOT NULL, asks TEXT NOT NULL,
    PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS ingest_meta (
    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, last_update REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe));
"""

FLAG_FILLED = 1
FLAG_OUTLIER = 2


def _ms(ts: str | pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // 1_000_000)


class OHLCVStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def write(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        idx = to_utc_index(df.index)
        ts_ms = (idx.as_unit("ns").asi8 // 1_000_000).astype("int64")
        flags = df["flags"].astype(int).to_numpy() if "flags" in df else [0] * len(df)
        rows = list(
            zip(
                [symbol] * len(df), [timeframe] * len(df), ts_ms.tolist(),
                df["open"].astype(float).tolist(), df["high"].astype(float).tolist(),
                df["low"].astype(float).tolist(), df["close"].astype(float).tolist(),
                df["volume"].astype(float).tolist(), list(map(int, flags)), strict=True,
            )
        )
        with self._lock, self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self._conn.execute(
                "INSERT OR REPLACE INTO ingest_meta VALUES (?,?,strftime('%s','now'))", (symbol, timeframe)
            )
        return len(rows)

    def load(self, symbol: str, timeframe: str, start: str | pd.Timestamp | None = None,
             end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        sql = "SELECT ts,open,high,low,close,volume,flags FROM ohlcv WHERE symbol=? AND timeframe=?"
        args: list = [symbol, timeframe]
        if start is not None:
            sql += " AND ts>=?"
            args.append(_ms(start))
        if end is not None:
            sql += " AND ts<?"
            args.append(_ms(end))
        with self._lock:
            df = pd.read_sql_query(sql + " ORDER BY ts", self._conn, params=args)
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        df.index.name = "ts"
        return df

    def last_ts(self, symbol: str, timeframe: str) -> pd.Timestamp | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?", (symbol, timeframe)
            ).fetchone()
        return pd.Timestamp(row[0], unit="ms", tz="UTC") if row and row[0] is not None else None

    def coverage(self) -> pd.DataFrame:
        with self._lock:
            df = pd.read_sql_query(
                "SELECT o.symbol, o.timeframe, MIN(o.ts) first_ts, MAX(o.ts) last_ts, COUNT(*) bars, "
                "SUM(o.flags & 1) filled, SUM((o.flags & 2) > 0) outliers, m.last_update "
                "FROM ohlcv o LEFT JOIN ingest_meta m ON m.symbol=o.symbol AND m.timeframe=o.timeframe "
                "GROUP BY o.symbol, o.timeframe ORDER BY o.symbol, o.timeframe",
                self._conn,
            )
        for c in ("first_ts", "last_ts"):
            df[c] = pd.to_datetime(df[c], unit="ms", utc=True)
        df["last_update"] = pd.to_datetime(df["last_update"].astype("float"), unit="s", utc=True)
        return df

    def symbols(self) -> list[tuple[str, str]]:
        with self._lock:
            return [tuple(r) for r in self._conn.execute("SELECT DISTINCT symbol,timeframe FROM ohlcv")]

    # ---- optional datasets -----------------------------------------------------------
    def write_funding(self, symbol: str, rows: list[tuple[int, float]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO funding VALUES (?,?,?)", [(symbol, t, r) for t, r in rows])

    def load_funding(self, symbol: str) -> pd.Series:
        with self._lock:
            df = pd.read_sql_query("SELECT ts, rate FROM funding WHERE symbol=? ORDER BY ts", self._conn,
                                   params=[symbol])
        return pd.Series(df["rate"].to_numpy(), index=pd.to_datetime(df["ts"], unit="ms", utc=True), name="funding")

    def write_orderbook(self, symbol: str, ts_ms: int, bids: str, asks: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO orderbook VALUES (?,?,?,?)", (symbol, ts_ms, bids, asks))

    def export_parquet(self, symbol: str, timeframe: str, out_dir: str | Path) -> Path:
        df = self.load(symbol, timeframe)
        out = Path(out_dir) / f"{symbol.replace('/', '-')}_{timeframe}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        return out
