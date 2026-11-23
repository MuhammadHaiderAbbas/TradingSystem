"""Timeframe helpers. All timestamps are UTC; bars are labelled by their OPEN time."""

from __future__ import annotations

import numpy as np
import pandas as pd

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
TF_PANDAS = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def tf_seconds(tf: str) -> int:
    try:
        return TF_SECONDS[tf]
    except KeyError as e:
        raise ValueError(f"unsupported timeframe {tf!r}; use one of {list(TF_SECONDS)}") from e


def tf_offset(tf: str) -> str:
    return TF_PANDAS[tf]


def periods_per_year(tf: str) -> float:
    """Crypto trades 24/7: 365 days a year."""
    return 365 * 86400 / tf_seconds(tf)


def to_utc_index(idx) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(idx)
    if di.tz is None:
        return di.tz_localize("UTC")
    return di.tz_convert("UTC")


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def infer_timeframe(df: pd.DataFrame) -> str:
    if len(df) < 2:
        raise ValueError("need at least two bars to infer timeframe")
    step = pd.Series(df.index).diff().dropna().mode().iloc[0].total_seconds()
    for tf, s in TF_SECONDS.items():
        if s == step:
            return tf
    raise ValueError(f"cannot infer timeframe from step {step}s")


def epoch_seconds(idx) -> np.ndarray:
    """Epoch seconds (float) for a DatetimeIndex regardless of its unit (pandas 3 uses 'us')."""
    di = to_utc_index(idx)
    return di.as_unit("ns").asi8.astype(np.float64) / 1e9


def from_epoch_seconds(ts: float) -> pd.Timestamp:
    return pd.Timestamp(int(round(ts * 1000)), unit="ms", tz="UTC")
