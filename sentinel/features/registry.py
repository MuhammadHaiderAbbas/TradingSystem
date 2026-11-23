"""Feature registry.

Strategies declare the features they need as ``FeatureRequest`` objects; the
registry computes them once into a single frame with deterministic column names.

    >>> reqs = [FeatureRequest("ema", n=12), FeatureRequest("adx", n=14)]
    >>> feats = compute_features(ohlcv, reqs)   # columns: ema_12, adx_14, plus_di_14, ...

Multi-timeframe: ``FeatureRequest("ema", n=50, timeframe="4h")`` is computed on
4h bars resampled from the base data and aligned back with no lookahead (column
``ema_50@4h``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sentinel.data.resample import align_higher_tf, resample_ohlcv
from sentinel.features import indicators as ind


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    func: Callable[..., pd.Series | pd.DataFrame]
    inputs: tuple[str, ...]
    defaults: dict[str, Any] = field(default_factory=dict)
    description: str = ""


REGISTRY: dict[str, FeatureSpec] = {}


def register(name: str, func, inputs: tuple[str, ...], description: str = "", **defaults) -> None:
    REGISTRY[name] = FeatureSpec(name, func, inputs, defaults, description)


register("sma", ind.sma, ("close",), "simple moving average", n=20)
register("ema", ind.ema, ("close",), "exponential moving average", n=20)
register("macd", ind.macd, ("close",), "MACD line/signal/hist", fast=12, slow=26, signal=9)
register("adx", ind.adx, ("high", "low", "close"), "ADX and directional indicators", n=14)
register("supertrend", ind.supertrend, ("high", "low", "close"), "Supertrend", n=10, mult=3.0)
register("ichimoku", ind.ichimoku, ("high", "low"), "Ichimoku (causal)", tenkan=9, kijun=26, senkou=52)
register("donchian", ind.donchian, ("high", "low"), "Donchian channel", n=20)
register("rsi", ind.rsi, ("close",), "RSI (Wilder)", n=14)
register("stoch", ind.stochastic, ("high", "low", "close"), "Stochastic %K/%D", k=14, d=3, smooth_k=3)
register("roc", ind.roc, ("close",), "rate of change %", n=10)
register("willr", ind.williams_r, ("high", "low", "close"), "Williams %R", n=14)
register("atr", ind.atr, ("high", "low", "close"), "average true range", n=14)
register("bbands", ind.bollinger, ("close",), "Bollinger bands", n=20, k=2.0)
register("keltner", ind.keltner, ("high", "low", "close"), "Keltner channel", n=20, mult=2.0, atr_n=10)
register("rvol", ind.realized_vol, ("close",), "realized volatility (per bar)", n=24)
register("pvol", ind.parkinson_vol, ("high", "low"), "Parkinson volatility (per bar)", n=24)
register("obv", ind.obv, ("close", "volume"), "on-balance volume")
register("vwap", ind.vwap, ("high", "low", "close", "volume"), "rolling VWAP", n=24)
register("vol_z", ind.volume_zscore, ("volume",), "volume z-score", n=50)
register("mfi", ind.mfi, ("high", "low", "close", "volume"), "money flow index", n=14)
register("zscore", ind.rolling_zscore, ("close",), "rolling z-score of close", n=50)
register("hurst", ind.hurst, ("close",), "rolling Hurst exponent", n=128, max_lag=20)
register("autocorr", ind.autocorr, ("close",), "rolling return autocorrelation", n=50, lag=1)
register("regime", ind.regime_labels, ("high", "low", "close"), "trend/range and vol regime labels",
         adx_n=14, adx_trend=25.0, vol_n=24, vol_lookback=500)


@dataclass(frozen=True)
class FeatureRequest:
    name: str
    timeframe: str | None = None  # None = base timeframe
    params: tuple[tuple[str, Any], ...] = ()

    def __init__(self, name: str, timeframe: str | None = None, **params: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "params", tuple(sorted(params.items())))

    @property
    def kwargs(self) -> dict[str, Any]:
        spec = REGISTRY[self.name]
        return {**spec.defaults, **dict(self.params)}

    def suffix(self) -> str:
        vals = [str(v) for _, v in sorted(self.kwargs.items())]
        s = "_".join(vals)
        return (f"_{s}" if s else "") + (f"@{self.timeframe}" if self.timeframe else "")


def _compute_one(df: pd.DataFrame, req: FeatureRequest) -> pd.DataFrame:
    if req.name not in REGISTRY:
        raise KeyError(f"unknown feature {req.name!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[req.name]
    res = spec.func(*(df[c] for c in spec.inputs), **req.kwargs)
    sfx = req.suffix()
    if isinstance(res, pd.Series):
        return pd.DataFrame({f"{req.name}{sfx}": res})
    return res.rename(columns=lambda c: f"{c}{sfx}")


def compute_features(df: pd.DataFrame, requests: list[FeatureRequest], base_tf: str | None = None) -> pd.DataFrame:
    parts = []
    seen: set[FeatureRequest] = set()
    for req in requests:
        if req in seen:
            continue
        seen.add(req)
        if req.timeframe and req.timeframe != base_tf:
            if base_tf is None:
                raise ValueError("base_tf is required for multi-timeframe features")
            higher = resample_ohlcv(df, base_tf, req.timeframe)
            parts.append(align_higher_tf(df, _compute_one(higher, req), base_tf, req.timeframe))
        else:
            parts.append(_compute_one(df, req))
    if not parts:
        return pd.DataFrame(index=df.index)
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def column(req: FeatureRequest, sub: str | None = None) -> str:
    """Column name for a request; ``sub`` selects one output of a multi-column feature."""
    return f"{sub or req.name}{req.suffix()}"


def describe() -> list[dict[str, Any]]:
    return [{"name": s.name, "inputs": s.inputs, "defaults": s.defaults, "description": s.description}
            for s in REGISTRY.values()]
