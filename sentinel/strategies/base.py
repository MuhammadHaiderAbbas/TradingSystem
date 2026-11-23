"""Strategy base class.

A strategy is a pure function of past data:

    generate_signals(data) -> pd.Series of target exposure in [-1, 1]

* ``data`` is an OHLCV frame (UTC index, bar open times) already joined with the
  features the strategy declared in ``features()``.
* The signal at bar t may use bar t's close; the engine executes it at the
  **next** bar's open. Lookahead is verified by an automated test for every strategy.
* 1 = fully long (up to the risk manager's sizing), 0 = flat, -1 = short.
  Shorts are clipped to 0 unless ``allow_short`` is enabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from sentinel.features.registry import FeatureRequest, compute_features


@dataclass(frozen=True)
class Param:
    default: Any
    low: float | None = None
    high: float | None = None
    step: float | None = None
    description: str = ""

    def grid(self, n: int | None = 5) -> list[Any]:
        """``n`` evenly spaced values in [low, high], snapped to ``step``; ``n=None`` enumerates every step."""
        import numpy as np

        if self.low is None or self.high is None:
            return [self.default]
        if n is None and self.step:
            raw = np.arange(self.low, self.high + self.step / 2, self.step)
        else:
            raw = np.linspace(self.low, self.high, n or 5)
            if self.step:
                raw = self.low + np.round((raw - self.low) / self.step) * self.step
        out: list[Any] = []
        for x in raw:
            v = type(self.default)(round(float(x), 10))
            if v not in out:
                out.append(v)
        return out


class Strategy(ABC):
    """Base class for all strategies. Subclasses set ``name`` and ``params`` and implement
    ``features`` + ``generate_signals``."""

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    params: ClassVar[dict[str, Param]] = {}
    timeframe: ClassVar[str | None] = None  # preferred timeframe (None = any)
    regimes: ClassVar[tuple[str, ...] | None] = None  # regimes where it's expected to work

    def __init__(self, **overrides: Any) -> None:
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        self.p: dict[str, Any] = {k: overrides.get(k, v.default) for k, v in self.params.items()}
        self.validate()

    def validate(self) -> None:
        for k, spec in self.params.items():
            v = self.p[k]
            if spec.low is not None and v < spec.low:
                raise ValueError(f"{self.name}.{k}={v} below minimum {spec.low}")
            if spec.high is not None and v > spec.high:
                raise ValueError(f"{self.name}.{k}={v} above maximum {spec.high}")

    def features(self) -> list[FeatureRequest]:
        return []

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ...

    def warmup(self) -> int:
        """Bars needed before signals are meaningful."""
        nums = [v for v in self.p.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return int(max(nums, default=0) * 3) + 1

    # ---- helpers ---------------------------------------------------------------------
    def prepare(self, ohlcv: pd.DataFrame, base_tf: str | None = None) -> pd.DataFrame:
        feats = compute_features(ohlcv, self.features(), base_tf=base_tf)
        return ohlcv.join(feats, how="left")

    def run(self, ohlcv: pd.DataFrame, base_tf: str | None = None, allow_short: bool = False) -> pd.Series:
        """Compute features + signals; NaN -> 0 (flat); clip to [-1, 1]; warm-up forced flat."""
        data = self.prepare(ohlcv, base_tf)
        sig = self.generate_signals(data).reindex(ohlcv.index).astype(float).fillna(0.0)
        sig = sig.clip(-1.0 if allow_short else 0.0, 1.0)
        sig.iloc[: min(self.warmup(), len(sig))] = 0.0
        return sig.rename(self.name)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.p.items())
        return f"{type(self).__name__}({args})"


def hold_state(entries: pd.Series, exits: pd.Series, short_entries: pd.Series | None = None,
               short_exits: pd.Series | None = None) -> pd.Series:
    """Turn entry/exit event series into a position-state series (1/0/-1), causally."""
    import numpy as np

    e = entries.fillna(False).to_numpy(bool)
    x = exits.fillna(False).to_numpy(bool)
    se = short_entries.fillna(False).to_numpy(bool) if short_entries is not None else np.zeros(len(e), bool)
    sx = short_exits.fillna(False).to_numpy(bool) if short_exits is not None else np.zeros(len(e), bool)
    out = np.zeros(len(e))
    state = 0.0
    for i in range(len(e)):
        if state > 0 and x[i]:
            state = 0.0
        elif state < 0 and sx[i]:
            state = 0.0
        if state == 0:
            if e[i]:
                state = 1.0
            elif se[i]:
                state = -1.0
        out[i] = state
    return pd.Series(out, index=entries.index)
