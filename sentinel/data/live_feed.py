"""Live market data feeds with automatic reconnect and heartbeat.

All feeds emit *closed* bars through ``on_bar(symbol, Bar)`` and update a heartbeat
timestamp on every message, so the risk engine can detect a stale feed.

* :class:`WebSocketFeed` -- ccxt.pro ``watch_ohlcv`` (WebSocket), reconnects with
  exponential backoff + jitter on any error.
* :class:`PollingFeed` -- REST fallback polling ``fetch_ohlcv``.
* :class:`ReplayFeed` -- replays a DataFrame (demo / synthetic paper trading / tests).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Iterable

import pandas as pd

from sentinel.core.types import Bar
from sentinel.data.timeframes import epoch_seconds, tf_seconds
from sentinel.monitoring.logging import get_logger

log = get_logger("data.feed")

BarHandler = Callable[[str, Bar], None]


class Feed:
    def __init__(self, symbols: Iterable[str], timeframe: str, on_bar: BarHandler) -> None:
        self.symbols = list(symbols)
        self.timeframe = timeframe
        self.on_bar = on_bar
        self.last_heartbeat: float = 0.0
        self.last_bar_ts: dict[str, float] = {}
        self.reconnects = 0
        self.errors = 0
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def beat(self) -> None:
        self.last_heartbeat = time.time()

    def seconds_since_heartbeat(self, now: float | None = None) -> float:
        return (now or time.time()) - self.last_heartbeat if self.last_heartbeat else float("inf")

    def _emit_closed(self, symbol: str, candles: list[list[float]], now_ms: float) -> None:
        step = tf_seconds(self.timeframe) * 1000
        for c in candles:
            ts_ms = c[0]
            if ts_ms + step > now_ms:  # still forming
                continue
            if ts_ms / 1000 <= self.last_bar_ts.get(symbol, -1):
                continue  # duplicate after reconnect
            self.last_bar_ts[symbol] = ts_ms / 1000
            self.on_bar(symbol, Bar(ts_ms / 1000, *map(float, c[1:6])))


class WebSocketFeed(Feed):
    def __init__(self, exchange_pro, symbols, timeframe, on_bar, max_backoff: float = 60.0) -> None:
        super().__init__(symbols, timeframe, on_bar)
        self.ex = exchange_pro
        self.max_backoff = max_backoff

    async def _watch(self, symbol: str) -> None:
        backoff = 1.0
        while not self._stop:
            try:
                candles = await self.ex.watch_ohlcv(symbol, self.timeframe)
                self.beat()
                backoff = 1.0
                # watch_ohlcv returns the live candle; emit the ones that are closed
                self._emit_closed(symbol, candles, time.time() * 1000)
            except Exception as e:
                self.errors += 1
                self.reconnects += 1
                sleep = min(self.max_backoff, backoff) * (0.5 + random.random())
                log.warning("websocket error, reconnecting", extra={"symbol": symbol, "err": str(e)[:200],
                                                                     "sleep_s": round(sleep, 2)})
                await asyncio.sleep(sleep)
                backoff *= 2

    async def run(self) -> None:
        try:
            await asyncio.gather(*(self._watch(s) for s in self.symbols))
        finally:
            close = getattr(self.ex, "close", None)
            if close:
                await close()


class PollingFeed(Feed):
    def __init__(self, exchange, symbols, timeframe, on_bar, poll_s: float = 10.0) -> None:
        super().__init__(symbols, timeframe, on_bar)
        self.ex = exchange
        self.poll_s = poll_s

    def poll_once(self, now_ms: float | None = None) -> None:
        for s in self.symbols:
            try:
                candles = self.ex.fetch_ohlcv(s, self.timeframe, limit=5)
                self.beat()
                self._emit_closed(s, candles, now_ms or time.time() * 1000)
            except Exception as e:
                self.errors += 1
                log.warning("poll error", extra={"symbol": s, "err": str(e)[:200]})

    def run(self) -> None:
        backoff = self.poll_s
        while not self._stop:
            before = self.errors
            self.poll_once()
            backoff = min(backoff * 2, 300) if self.errors > before else self.poll_s
            time.sleep(backoff)


class ReplayFeed(Feed):
    """Replays historical/synthetic bars in timestamp order across symbols."""

    def __init__(self, data: dict[str, pd.DataFrame], timeframe: str, on_bar: BarHandler,
                 delay_s: float = 0.0) -> None:
        super().__init__(data.keys(), timeframe, on_bar)
        self.data = data
        self.delay_s = delay_s

    def run(self, max_bars: int | None = None) -> int:
        events = []
        for sym, df in self.data.items():
            ts = epoch_seconds(df.index)
            arr = df[["open", "high", "low", "close", "volume"]].to_numpy()
            events.extend((t, sym, row) for t, row in zip(ts, arr, strict=True))
        events.sort(key=lambda e: (e[0], e[1]))
        n = 0
        for t, sym, row in events:
            if self._stop or (max_bars is not None and n >= max_bars * len(self.symbols)):
                break
            self.beat()
            self.last_bar_ts[sym] = t
            self.on_bar(sym, Bar(float(t), *map(float, row)))
            n += 1
            if self.delay_s:
                time.sleep(self.delay_s)
        return n
