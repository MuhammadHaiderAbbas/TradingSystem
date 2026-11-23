"""Historical OHLCV downloader (ccxt) with pagination, rate limiting and resume.

* Resume: starts from the last stored bar for (symbol, timeframe), so re-running is
  cheap and interrupted downloads continue where they stopped.
* Pagination: walks forward with ``since`` until it reaches ``until`` / now.
* Rate limits: honours ``exchange.rateLimit`` between calls and backs off
  exponentially on rate-limit / network errors.
* The still-forming current candle is never stored.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from sentinel.data.quality import check, repair
from sentinel.data.storage import OHLCVStore
from sentinel.data.timeframes import tf_seconds
from sentinel.monitoring.logging import get_logger

log = get_logger("data.downloader")

TRANSIENT_ERRORS = ("RateLimitExceeded", "DDoSProtection", "NetworkError", "RequestTimeout",
                    "ExchangeNotAvailable", "OnMaintenance")


def _is_transient(e: Exception) -> bool:
    return any(cls.__name__ in TRANSIENT_ERRORS for cls in type(e).__mro__)


def make_exchange(venue: str, testnet: bool = True, credentials: dict | None = None) -> Any:
    try:
        import ccxt  # type: ignore
    except ImportError as e:
        raise RuntimeError("ccxt not installed: pip install 'sentinel-trader[exchanges]'") from e
    cls = getattr(ccxt, venue)
    ex = cls({"enableRateLimit": True, **(credentials or {})})
    if testnet and hasattr(ex, "set_sandbox_mode"):
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            log.warning("venue has no sandbox for this market type; public data only", extra={"venue": venue})
    return ex


class HistoricalDownloader:
    def __init__(self, exchange: Any, store: OHLCVStore, limit: int = 1000, max_retries: int = 6,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.ex = exchange
        self.store = store
        self.limit = limit
        self.max_retries = max_retries
        self.sleep = sleep

    def _fetch(self, symbol: str, timeframe: str, since_ms: int) -> list[list[float]]:
        delay = max(getattr(self.ex, "rateLimit", 200), 50) / 1000
        for attempt in range(self.max_retries + 1):
            try:
                return self.ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=self.limit)
            except Exception as e:
                if not _is_transient(e) or attempt == self.max_retries:
                    raise
                backoff = delay * (2**attempt)
                log.warning("transient error, backing off", extra={"symbol": symbol, "err": type(e).__name__,
                                                                    "sleep_s": backoff})
                self.sleep(backoff)
        return []

    def download(self, symbol: str, timeframe: str, since: str | pd.Timestamp = "2021-01-01",
                 until: str | pd.Timestamp | None = None, now_ms: int | None = None) -> int:
        step_ms = tf_seconds(timeframe) * 1000
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        last = self.store.last_ts(symbol, timeframe)
        start = pd.Timestamp(since)
        start = start.tz_localize("UTC") if start.tzinfo is None else start
        since_ms = int(start.value // 1_000_000)
        if last is not None:
            since_ms = max(since_ms, int(last.value // 1_000_000) + step_ms)
        end_ms = now_ms
        if until is not None:
            u = pd.Timestamp(until)
            end_ms = min(end_ms, int((u.tz_localize("UTC") if u.tzinfo is None else u).value // 1_000_000))
        total = 0
        delay = max(getattr(self.ex, "rateLimit", 200), 50) / 1000
        while since_ms < end_ms:
            rows = self._fetch(symbol, timeframe, since_ms)
            if not rows:
                break
            # Only fully closed candles: open + step <= now
            rows = [r for r in rows if r[0] >= since_ms and r[0] + step_ms <= now_ms and r[0] < end_ms]
            if not rows:
                break
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
            df = repair(df, timeframe, now=pd.Timestamp(now_ms, unit="ms", tz="UTC"))
            total += self.store.write(symbol, timeframe, df)
            next_since = int(rows[-1][0]) + step_ms
            if next_since <= since_ms:
                break
            since_ms = next_since
            log.info("downloaded page", extra={"symbol": symbol, "tf": timeframe, "rows": len(rows),
                                               "through": str(df.index[-1])})
            self.sleep(delay)
        if total:
            check(self.store.load(symbol, timeframe), symbol, timeframe)
        return total

    def download_funding(self, symbol: str, since_ms: int) -> int:
        if not getattr(self.ex, "has", {}).get("fetchFundingRateHistory"):
            return 0
        rows = self.ex.fetch_funding_rate_history(symbol, since=since_ms)
        data = [(int(r["timestamp"]), float(r["fundingRate"])) for r in rows]
        self.store.write_funding(symbol, data)
        return len(data)

    def snapshot_order_book(self, symbol: str, depth: int = 20) -> None:
        import json

        ob = self.ex.fetch_order_book(symbol, limit=depth)
        self.store.write_orderbook(symbol, int(ob.get("timestamp") or time.time() * 1000),
                                   json.dumps(ob["bids"][:depth]), json.dumps(ob["asks"][:depth]))
