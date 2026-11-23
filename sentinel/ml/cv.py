"""Purged and embargoed K-fold cross-validation for time series (no random shuffling).

Each sample i spans [t0_i, t1_i] (signal bar -> label resolution bar). For a test fold
covering [T0, T1]:

* **purge**: drop training samples whose span overlaps [T0, T1]
* **embargo**: also drop training samples starting within ``embargo`` bars after T1
  (serial correlation leaks information forward)

Folds are contiguous in time.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, t0: np.ndarray, t1: np.ndarray, n_bars: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """t0/t1: integer bar positions of each sample's start/end, sorted by t0."""
        t0, t1 = np.asarray(t0), np.asarray(t1)
        if np.any(np.diff(t0) < 0):
            raise ValueError("samples must be sorted by start time")
        n = len(t0)
        n_bars = n_bars or int(t1.max()) + 1
        embargo = int(np.ceil(n_bars * self.embargo_pct))
        bounds = np.array_split(np.arange(n), self.n_splits)
        for test_idx in bounds:
            if len(test_idx) == 0:
                continue
            T0, T1 = t0[test_idx[0]], t1[test_idx].max()
            overlap = (t0 <= T1) & (t1 >= T0)
            embargoed = (t0 > T1) & (t0 <= T1 + embargo)
            train = np.where(~overlap & ~embargoed)[0]
            train = np.setdiff1d(train, test_idx)
            yield train, test_idx
