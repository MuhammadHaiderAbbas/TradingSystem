"""Automated lookahead-bias detection.

Two complementary tests for any function ``f(df) -> Series/DataFrame``:

1. **Truncation test**: the value at bar t computed on the full history must equal the
   value computed on ``df[:t+1]`` (nothing after t can matter).
2. **Future-perturbation test**: randomly scrambling every bar *after* t must not
   change any value at or before t.

Used in the test suite for every registered feature and every strategy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _as_frame(x: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return x.to_frame() if isinstance(x, pd.Series) else x


def _equal(a: pd.DataFrame, b: pd.DataFrame, rtol: float, atol: float) -> bool:
    a, b = a.align(b, join="inner")
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype == object or y.dtype == object or isinstance(x.dtype, pd.StringDtype):
            if not (x.fillna("<NA>").astype(str) == y.fillna("<NA>").astype(str)).all():
                return False
            continue
        x, y = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
            return False
    return True


def find_lookahead(func: Callable[[pd.DataFrame], pd.Series | pd.DataFrame], df: pd.DataFrame,
                   n_checks: int = 8, min_history: int = 300, seed: int = 0, rtol: float = 1e-7,
                   atol: float = 1e-9) -> list[str]:
    """Return a list of violation descriptions (empty = no lookahead detected)."""
    rng = np.random.default_rng(seed)
    full = _as_frame(func(df))
    problems: list[str] = []
    if len(df) <= min_history + 2:
        raise ValueError("not enough data for a lookahead test")
    cuts = sorted(rng.choice(np.arange(min_history, len(df) - 1), size=min(n_checks, len(df) - min_history - 1),
                             replace=False))
    for t in cuts:
        trunc = _as_frame(func(df.iloc[: t + 1]))
        if not _equal(full.iloc[: t + 1], trunc, rtol, atol):
            problems.append(f"truncation mismatch at bar {t} ({df.index[t]})")
            continue
        pert = df.copy()
        n_after = len(df) - t - 1
        noise = np.exp(rng.normal(0, 0.05, n_after))
        for c in ("open", "high", "low", "close"):
            pert.iloc[t + 1:, pert.columns.get_loc(c)] = df[c].to_numpy()[t + 1:] * noise
        pert.iloc[t + 1:, pert.columns.get_loc("high")] = pert[["open", "high", "low", "close"]].iloc[t + 1:].max(axis=1)
        pert.iloc[t + 1:, pert.columns.get_loc("low")] = pert[["open", "high", "low", "close"]].iloc[t + 1:].min(axis=1)
        pert.iloc[t + 1:, pert.columns.get_loc("volume")] = df["volume"].to_numpy()[t + 1:] * rng.uniform(0.2, 5, n_after)
        perturbed = _as_frame(func(pert))
        if not _equal(full.iloc[: t + 1], perturbed.iloc[: t + 1], rtol, atol):
            problems.append(f"future perturbation changed values at/before bar {t} ({df.index[t]})")
    return problems
