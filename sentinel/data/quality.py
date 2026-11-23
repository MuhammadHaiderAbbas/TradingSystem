"""Data-quality checks and repair.

Checks: missing candles (timestamp gaps), duplicates, misaligned timestamps,
outliers (robust z-score of log returns), zero volume, invalid OHLC relationships,
future timestamps and non-UTC indexes. Every finding is logged; ``repair`` fixes
what can be fixed and *flags* the rest (``flags`` column) rather than silently
hiding it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from sentinel.data.storage import FLAG_FILLED, FLAG_OUTLIER
from sentinel.data.timeframes import tf_offset, tf_seconds, to_utc_index, utc_now
from sentinel.monitoring.logging import get_logger

log = get_logger("data.quality")


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    bars: int = 0
    duplicates: int = 0
    missing: int = 0
    gaps: list[tuple[str, str, int]] = field(default_factory=list)  # (from, to, missing bars)
    misaligned: int = 0
    outliers: int = 0
    zero_volume: int = 0
    invalid_ohlc: int = 0
    future: int = 0
    non_positive: int = 0

    @property
    def ok(self) -> bool:
        return not any((self.duplicates, self.missing, self.misaligned, self.outliers, self.invalid_ohlc,
                        self.future, self.non_positive))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        d["gaps"] = d["gaps"][:20]
        return d


def robust_return_z(close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    med = r.rolling(500, min_periods=50).median()
    mad = (r - med).abs().rolling(500, min_periods=50).median()
    return (r - med) / (1.4826 * mad.replace(0, np.nan))


def check(df: pd.DataFrame, symbol: str, timeframe: str, outlier_z: float = 8.0,
          now: pd.Timestamp | None = None) -> QualityReport:
    rep = QualityReport(symbol=symbol, timeframe=timeframe, bars=len(df))
    if df.empty:
        return rep
    idx = to_utc_index(df.index)
    step = pd.Timedelta(seconds=tf_seconds(timeframe))
    rep.duplicates = int(idx.duplicated().sum())
    uniq = idx[~idx.duplicated()].sort_values()
    rep.misaligned = int((uniq.as_unit("ns").asi8 % step.value != 0).sum())
    diffs = pd.Series(uniq[1:] - uniq[:-1], index=uniq[1:])
    for ts_to, d in diffs[diffs > step].items():
        n_missing = int(d / step) - 1
        rep.missing += n_missing
        rep.gaps.append((str(ts_to - d), str(ts_to), n_missing))
    now = now or utc_now()
    # A bar is "future" if it OPENS after now (a still-forming bar opens <= now).
    rep.future = int((idx > now).sum())
    o, h, lo, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    rep.non_positive = int(((o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)).sum())
    rep.invalid_ohlc = int(((h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (h < lo)).sum())
    rep.zero_volume = int((v <= 0).sum())
    clean_close = c[c > 0]
    z = robust_return_z(clean_close[~clean_close.index.duplicated()])
    rep.outliers = int((z.abs() > outlier_z).sum())
    level = "warning" if not rep.ok else "info"
    getattr(log, level)("data quality check", extra=rep.to_dict())
    return rep


def repair(df: pd.DataFrame, timeframe: str, outlier_z: float = 8.0, now: pd.Timestamp | None = None,
           fill_limit: int | None = 24) -> pd.DataFrame:
    """Return a cleaned copy.

    * duplicates: keep the last observation
    * future / non-positive rows: dropped
    * misaligned timestamps: floored to the bar boundary, then de-duplicated
    * invalid OHLC: high/low widened to contain open/close
    * missing candles: inserted as flat bars (O=H=L=C=previous close, volume 0) and flagged
      ``FLAG_FILLED`` -- up to ``fill_limit`` consecutive bars; longer gaps stay as gaps
    * outliers: flagged ``FLAG_OUTLIER`` (never silently altered)
    """
    out = df.copy()
    out.index = to_utc_index(out.index)
    now = now or utc_now()
    out = out[out.index <= now]
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out.index = out.index.floor(tf_offset(timeframe))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = out["volume"].clip(lower=0)
    flags = out["flags"].astype(int) if "flags" in out else pd.Series(0, index=out.index)
    if len(out) > 1:
        full = pd.date_range(out.index[0], out.index[-1], freq=tf_offset(timeframe))
        out = out.reindex(full)
        flags = flags.reindex(full)
        missing = out["close"].isna()
        prev_close = out["close"].ffill(limit=fill_limit)
        fillable = missing & prev_close.notna()
        for col in ("open", "high", "low", "close"):
            out.loc[fillable, col] = prev_close[fillable]
        out.loc[fillable, "volume"] = 0.0
        flags = flags.fillna(0).astype(int)
        flags[fillable] |= FLAG_FILLED
        out = out[out["close"].notna()]
        flags = flags[out.index]
    z = robust_return_z(out["close"])
    flags[(z.abs() > outlier_z).reindex(flags.index, fill_value=False)] |= FLAG_OUTLIER
    out["flags"] = flags.astype(int)
    out.index.name = "ts"
    return out
