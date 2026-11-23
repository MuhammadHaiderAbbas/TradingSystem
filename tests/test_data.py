import numpy as np
import pandas as pd
import pytest

from sentinel.core.types import Bar
from sentinel.data.downloader import HistoricalDownloader
from sentinel.data.live_feed import Feed, ReplayFeed
from sentinel.data.quality import check, repair
from sentinel.data.resample import align_higher_tf, resample_ohlcv
from sentinel.data.storage import FLAG_FILLED, FLAG_OUTLIER, OHLCVStore
from sentinel.data.synthetic import generate_ohlcv
from sentinel.data.timeframes import epoch_seconds, infer_timeframe


@pytest.fixture
def df():
    return generate_ohlcv(1000, seed=5)


def test_synthetic_is_valid_and_utc(df):
    assert str(df.index.tz) == "UTC"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    rep = check(df, "X", "1h", now=df.index[-1] + pd.Timedelta(hours=2))
    assert (rep.duplicates, rep.missing, rep.misaligned, rep.invalid_ohlc, rep.future) == (0, 0, 0, 0, 0)
    assert infer_timeframe(df) == "1h"


def test_gap_detection(df):
    broken = df.drop(df.index[100:105]).drop(df.index[500])
    rep = check(broken, "X", "1h", now=df.index[-1] + pd.Timedelta(days=1))
    assert rep.missing == 6
    assert len(rep.gaps) == 2
    assert rep.gaps[0][2] == 5


def test_duplicates_zero_volume_invalid_and_future(df):
    d = pd.concat([df, df.iloc[[10, 20]]]).sort_index()
    d.iloc[50, d.columns.get_loc("volume")] = 0
    d.iloc[60, d.columns.get_loc("high")] = d.iloc[60]["low"] - 1  # invalid OHLC
    now = df.index[-10]
    rep = check(d, "X", "1h", now=now)
    assert rep.duplicates == 2
    assert rep.zero_volume >= 1
    assert rep.invalid_ohlc >= 1
    assert rep.future == 9  # bars opening after `now`
    assert not rep.ok


def test_outlier_detection_and_flagging(df):
    d = df.copy()
    d.iloc[700, d.columns.get_loc("close")] *= 3
    rep = check(d, "X", "1h", now=d.index[-1] + pd.Timedelta(hours=1))
    assert rep.outliers >= 1
    fixed = repair(d, "1h", now=d.index[-1] + pd.Timedelta(hours=1))
    assert (fixed["flags"] & FLAG_OUTLIER).any()
    assert fixed.loc[d.index[700], "close"] == d.iloc[700]["close"]  # flagged, never silently altered


def test_repair_fills_gaps_drops_future_and_dupes(df):
    d = pd.concat([df.drop(df.index[200:203]), df.iloc[[5]]])
    now = df.index[-20]
    fixed = repair(d, "1h", now=now)
    assert fixed.index.is_unique and fixed.index.is_monotonic_increasing
    assert fixed.index.max() <= now
    filled = fixed[(fixed["flags"] & FLAG_FILLED) > 0]
    assert len(filled) == 3 and (filled["volume"] == 0).all()
    prev_close = fixed.loc[df.index[199], "close"]
    assert (filled["close"] == prev_close).all()


def test_repair_floors_misaligned(df):
    d = df.iloc[:50].copy()
    d.index = d.index + pd.Timedelta(seconds=7)
    assert check(d, "X", "1h", now=d.index[-1] + pd.Timedelta(hours=1)).misaligned == 50
    fixed = repair(d, "1h", now=d.index[-1] + pd.Timedelta(hours=1))
    assert (fixed.index == df.index[:50]).all()


def test_resample_correct_and_complete_bins_only(df):
    h4 = resample_ohlcv(df, "1h", "4h")
    first = df.iloc[:4] if df.index[0].hour % 4 == 0 else None
    assert first is not None
    row = h4.iloc[0]
    assert row["open"] == first["open"].iloc[0] and row["close"] == first["close"].iloc[-1]
    assert row["high"] == first["high"].max() and row["low"] == first["low"].min()
    assert np.isclose(row["volume"], first["volume"].sum())
    # incomplete trailing bin dropped
    partial = resample_ohlcv(df.iloc[:-2], "1h", "4h")
    assert partial.index[-1] + pd.Timedelta(hours=4) <= df.index[-3] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        resample_ohlcv(df, "4h", "1h")


def test_higher_tf_alignment_has_no_lookahead(df):
    h4 = resample_ohlcv(df, "1h", "4h")["close"]
    aligned = align_higher_tf(df["close"], h4, "1h", "4h")
    # The 4h bar opening at T is known at the close of the 1h bar opening at T+3h, and equals that close.
    t = h4.index[5]
    assert np.isnan(aligned.loc[t + pd.Timedelta(hours=2)]) or aligned.loc[t + pd.Timedelta(hours=2)] != h4.loc[t] \
        or h4.loc[t] == h4.iloc[4]
    assert aligned.loc[t + pd.Timedelta(hours=3)] == h4.loc[t] == df["close"].loc[t + pd.Timedelta(hours=3)]


def test_no_future_timestamps_in_store_roundtrip(tmp_path, df):
    st = OHLCVStore(tmp_path / "m.db")
    st.write("BTC/USDT", "1h", repair(df, "1h", now=df.index[-1] + pd.Timedelta(hours=1)))
    back = st.load("BTC/USDT", "1h")
    assert str(back.index.tz) == "UTC"
    assert (back.index == df.index).all()
    np.testing.assert_allclose(back["close"].to_numpy(), df["close"].to_numpy())
    assert st.last_ts("BTC/USDT", "1h") == df.index[-1]
    sub = st.load("BTC/USDT", "1h", start=df.index[10], end=df.index[20])
    assert len(sub) == 10
    cov = st.coverage()
    assert cov.iloc[0]["bars"] == len(df)


class FakeExchange:
    """ccxt-like exchange serving a fixed candle list, with an injected rate-limit error."""

    rateLimit = 1
    has = {"fetchFundingRateHistory": False}

    def __init__(self, df, fail_first=True):
        self.rows = [[int(t.value // 1_000_000), *r] for t, r in
                     zip(df.index, df[["open", "high", "low", "close", "volume"]].to_numpy().tolist(), strict=True)]
        self.calls = 0
        self.fail_first = fail_first

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            class RateLimitExceeded(Exception):
                pass
            raise RateLimitExceeded("slow down")
        return [r for r in self.rows if r[0] >= since][:limit]


def test_downloader_paginates_resumes_and_skips_forming_candle(tmp_path, df):
    st = OHLCVStore(tmp_path / "m.db")
    ex = FakeExchange(df)
    sleeps = []
    dl = HistoricalDownloader(ex, st, limit=200, sleep=sleeps.append)
    now_ms = int(df.index[-1].value // 1_000_000) + 30 * 60 * 1000  # last candle still forming
    n = dl.download("BTC/USDT", "1h", since=df.index[0], now_ms=now_ms)
    assert n == len(df) - 1  # forming candle excluded
    assert ex.calls >= 5  # paginated (+1 retried call)
    assert sleeps and sleeps[0] > 0  # backed off after the rate-limit error
    calls_before = ex.calls
    n2 = dl.download("BTC/USDT", "1h", since=df.index[0], now_ms=now_ms + 3600 * 1000)
    assert n2 == 1 and ex.calls - calls_before <= 2  # resumed from the last stored bar


def test_feed_emits_only_closed_bars_without_duplicates():
    got = []
    f = Feed(["X"], "1h", lambda s, b: got.append(b.ts))
    now = 10 * 3600 * 1000 + 1800 * 1000
    candles = [[i * 3600 * 1000, 1, 2, 0.5, 1.5, 10] for i in range(8, 11)]
    f._emit_closed("X", candles, now)
    f._emit_closed("X", candles, now)  # reconnect replays the same candles
    assert got == [8 * 3600, 9 * 3600]


def test_replay_feed_orders_by_time(df):
    a, b = df.iloc[:5], df.iloc[:5]
    seen = []
    ReplayFeed({"A": a, "B": b}, "1h", lambda s, bar: seen.append((bar.ts, s))).run()
    assert seen == sorted(seen)
    assert isinstance(Bar(*([0.0] * 6)), Bar)
    assert seen[0][0] == epoch_seconds(df.index[:1])[0]
