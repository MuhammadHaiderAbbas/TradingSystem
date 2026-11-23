"""Deterministic synthetic OHLCV generator for demos and tests.

A regime-switching random walk (bull / bear / sideways / crash regimes with
different drift and volatility) so strategies, risk limits and robustness tests
get exercised on varied conditions. It is *not* real market data and results on
it say nothing about real-world performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.timeframes import tf_offset, tf_seconds

REGIMES = {
    # name: (annualised drift, annualised vol, mean duration in days)
    "bull": (0.8, 0.55, 60),
    "bear": (-0.6, 0.65, 45),
    "sideways": (0.0, 0.35, 50),
    "crash": (-4.0, 1.4, 6),
}
_TRANSITION_WEIGHTS = {"bull": 0.35, "bear": 0.25, "sideways": 0.33, "crash": 0.07}


def generate_ohlcv(
    n_bars: int = 8760,
    timeframe: str = "1h",
    start: str = "2022-01-01",
    start_price: float = 30_000.0,
    seed: int = 42,
    base_volume: float = 500.0,
    return_regimes: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt_years = tf_seconds(timeframe) / (365 * 86400)
    bars_per_day = 86400 / tf_seconds(timeframe)

    names = list(REGIMES)
    weights = np.array([_TRANSITION_WEIGHTS[n] for n in names])
    regime_seq = np.empty(n_bars, dtype=object)
    i = 0
    current = "sideways"
    while i < n_bars:
        _, _, dur_days = REGIMES[current]
        length = max(1, int(rng.exponential(dur_days * bars_per_day)))
        regime_seq[i : i + length] = current
        i += length
        current = rng.choice(names, p=weights / weights.sum())

    mu = np.array([REGIMES[r][0] for r in regime_seq])
    sigma = np.array([REGIMES[r][1] for r in regime_seq])
    # fat tails: student-t shocks scaled to unit variance
    shocks = rng.standard_t(df=4, size=n_bars) / np.sqrt(2.0)
    log_ret = (mu - 0.5 * sigma**2) * dt_years + sigma * np.sqrt(dt_years) * shocks
    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    bar_sigma = sigma * np.sqrt(dt_years)
    wick_hi = np.abs(rng.normal(0, 0.6, n_bars)) * bar_sigma
    wick_lo = np.abs(rng.normal(0, 0.6, n_bars)) * bar_sigma
    high = np.maximum(open_, close) * np.exp(wick_hi)
    low = np.minimum(open_, close) * np.exp(-wick_lo)
    vol_mult = 1 + 3 * np.abs(shocks) + (regime_seq == "crash") * 2
    volume = base_volume * vol_mult * rng.lognormal(0, 0.3, n_bars)

    idx = pd.date_range(start=pd.Timestamp(start, tz="UTC"), periods=n_bars, freq=tf_offset(timeframe))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
    df.index.name = "ts"
    if return_regimes:
        df["regime"] = regime_seq
    return df


def demo_universe(timeframe: str = "1h", n_bars: int = 17520, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Two correlated-ish synthetic assets named like the real ones, clearly marked SYNTH."""
    btc = generate_ohlcv(n_bars, timeframe, start_price=30_000, seed=seed)
    eth_raw = generate_ohlcv(n_bars, timeframe, start_price=2_000, seed=seed + 1)
    # Blend returns to create realistic cross-asset correlation (~0.6)
    r_btc = np.log(btc["close"]).diff().fillna(0)
    r_eth = np.log(eth_raw["close"]).diff().fillna(0)
    r = 0.6 * r_btc * 1.2 + 0.8 * r_eth
    close = 2_000 * np.exp(r.cumsum())
    scale = close / eth_raw["close"]
    eth = eth_raw.copy()
    for c in ("open", "high", "low", "close"):
        eth[c] = eth_raw[c] * scale
    eth["open"] = eth["close"].shift(1).fillna(eth["close"].iloc[0])
    eth["high"] = np.maximum(eth["high"], eth[["open", "close"]].max(axis=1))
    eth["low"] = np.minimum(eth["low"], eth[["open", "close"]].min(axis=1))
    return {"SYNTH-BTC/USDT": btc, "SYNTH-ETH/USDT": eth}
