"""Signal combiner (vote / weighted ensemble), regime filter, and the signal pipeline
that turns the config's strategy list into one target-exposure series per symbol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.config import Settings, StrategyEntry
from sentinel.features.indicators import regime_labels
from sentinel.strategies import registry
from sentinel.strategies.base import Strategy

TREND_LABELS = {"trend", "range"}
VOL_LABELS = {"high_vol", "low_vol"}


def combine_signals(signals: dict[str, pd.Series], weights: dict[str, float] | None = None,
                    method: str = "vote", threshold: float = 0.5) -> pd.Series:
    if not signals:
        raise ValueError("no signals to combine")
    df = pd.DataFrame(signals).fillna(0.0)
    if method == "none":
        if df.shape[1] != 1:
            raise ValueError("method 'none' needs exactly one signal")
        return df.iloc[:, 0]
    w = pd.Series({k: (weights or {}).get(k, 1.0) for k in df.columns}, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must sum to > 0")
    w = w / w.sum()
    if method == "vote":
        score = (np.sign(df) * w).sum(axis=1)
        return pd.Series(np.where(score >= threshold, 1.0, np.where(score <= -threshold, -1.0, 0.0)),
                         index=df.index)
    if method == "weighted":
        avg = (df * w).sum(axis=1)
        return avg.where(avg.abs() >= threshold * 0.5, 0.0).clip(-1, 1)
    raise ValueError(f"unknown combiner method {method!r}")


def regime_mask(data: pd.DataFrame, allowed: list[str] | tuple[str, ...] | None) -> pd.Series:
    """True where the current regime is one the strategy is allowed to trade in."""
    if not allowed:
        return pd.Series(True, index=data.index)
    allowed = set(allowed)
    unknown = allowed - TREND_LABELS - VOL_LABELS
    if unknown:
        raise ValueError(f"unknown regimes {sorted(unknown)}")
    lab = regime_labels(data["high"], data["low"], data["close"])
    mask = pd.Series(True, index=data.index)
    for labels, col in ((TREND_LABELS, "regime_trend"), (VOL_LABELS, "regime_vol")):
        wanted = allowed & labels
        if wanted:
            mask &= lab[col].isin(wanted)
    return mask


def apply_regime_filter(signal: pd.Series, mask: pd.Series, mode: str = "entry") -> pd.Series:
    """``entry``: block *new* positions outside allowed regimes but let open ones run to the
    strategy's own exit. ``strict``: force flat whenever the regime is not allowed."""
    if mode == "strict":
        return signal.where(mask, 0.0)
    s = signal.to_numpy(dtype=float)
    m = mask.reindex(signal.index, fill_value=False).to_numpy(bool)
    out = np.zeros(len(s))
    for i in range(len(s)):
        prev = out[i - 1] if i else 0.0
        if s[i] == 0:
            out[i] = 0.0
        elif m[i] or (prev != 0 and np.sign(prev) == np.sign(s[i])):
            out[i] = s[i]
    return pd.Series(out, index=signal.index, name=signal.name)


@dataclass
class PipelineResult:
    target: pd.Series                 # combined target exposure
    per_strategy: dict[str, pd.Series]  # after regime filter
    source: str                       # attribution label


class SignalPipeline:
    def __init__(self, entries: list[StrategyEntry], method: str = "none", threshold: float = 0.5,
                 allow_short: bool = False, timeframe: str | None = None, regime_mode: str = "entry") -> None:
        self.entries = [e for e in entries if e.enabled]
        self.strategies: list[tuple[StrategyEntry, Strategy]] = [
            (e, registry.create(e.name, **e.params)) for e in self.entries
        ]
        self.method = method
        self.threshold = threshold
        self.allow_short = allow_short
        self.timeframe = timeframe
        self.regime_mode = regime_mode

    @classmethod
    def from_settings(cls, s: Settings) -> SignalPipeline:
        return cls(s.strategies, s.combiner.method, s.combiner.threshold, s.backtest.allow_short,
                   s.market.timeframe)

    def warmup(self) -> int:
        return max(st.warmup() for _, st in self.strategies)

    def run(self, ohlcv: pd.DataFrame) -> PipelineResult:
        per: dict[str, pd.Series] = {}
        weights: dict[str, float] = {}
        for entry, strat in self.strategies:
            sig = strat.run(ohlcv, base_tf=self.timeframe, allow_short=self.allow_short)
            allowed = entry.regimes if entry.regimes is not None else None
            if allowed:
                sig = apply_regime_filter(sig, regime_mask(ohlcv, allowed), self.regime_mode)
            per[entry.name] = sig
            weights[entry.name] = entry.weight
        if len(per) == 1:
            name = next(iter(per))
            return PipelineResult(per[name], per, name)
        method = self.method if self.method != "none" else "vote"
        return PipelineResult(combine_signals(per, weights, method, self.threshold), per, f"ensemble:{method}")
