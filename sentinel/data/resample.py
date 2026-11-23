"""Resampling to higher timeframes.

Bins are left-closed and labelled by their OPEN time (exchange convention). Only
*complete* bins are returned, so a 4h bar never contains information from bars
that hadn't closed yet.
"""

from __future__ import annotations

import pandas as pd

from sentinel.data.timeframes import tf_offset, tf_seconds

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, source_tf: str, target_tf: str, drop_incomplete: bool = True) -> pd.DataFrame:
    s, t = tf_seconds(source_tf), tf_seconds(target_tf)
    if t < s or t % s:
        raise ValueError(f"cannot resample {source_tf} -> {target_tf}")
    if t == s:
        return df[list(AGG)].copy()
    g = df[list(AGG)].resample(tf_offset(target_tf), label="left", closed="left")
    out = g.agg(AGG)
    counts = g["close"].count()
    if drop_incomplete:
        out = out[counts == t // s]
    return out.dropna()


def align_higher_tf(base: pd.DataFrame | pd.Series, higher: pd.DataFrame | pd.Series, base_tf: str,
                    higher_tf: str) -> pd.DataFrame | pd.Series:
    """Map higher-timeframe values onto the base index without lookahead.

    A higher-TF bar opening at T closes at T + higher_tf; its values become known at
    the close of the base bar that *ends* at that moment, i.e. the base bar opening at
    T + higher_tf - base_tf.
    """
    shift = pd.Timedelta(seconds=tf_seconds(higher_tf) - tf_seconds(base_tf))
    shifted = higher.copy()
    shifted.index = shifted.index + shift
    return shifted.reindex(base.index, method="ffill")
